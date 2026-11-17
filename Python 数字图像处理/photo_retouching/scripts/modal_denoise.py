"""SCUNet 降噪的 GPU 版本（Modal T4），供 learned_denoise 在“疑难”照片上调用。

不需要 modal deploy：每次用临时 app 运行（app.run()），跑完即释放，不常驻、不产生空闲费用。
上传的是整张 JPEG（只为降噪，不保存），回传降噪后的 PNG。
"""
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("photo-denoise-scunet")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy<2", "opencv-python-headless", "einops")
    .run_commands("mkdir -p /w && python -c \"import urllib.request; urllib.request.urlretrieve("
                  "'https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth', "
                  "'/w/scunet.pth')\"")
    .add_local_file(os.path.join(HERE, "scunet_net.py"), "/w/scunet_net.py")
)


@app.function(image=image, gpu="T4", timeout=1800)
def denoise_remote(jpeg: bytes, todo: list, tile: int, overlap: int) -> bytes:
    import sys

    import cv2
    import numpy as np
    import torch
    sys.path.insert(0, "/w")
    from scunet_net import SCUNet
    m = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
    m.load_state_dict(torch.load("/w/scunet.pth", map_location="cpu"), strict=True)
    m = m.eval().cuda().half()
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    acc = np.zeros((h, w, 3), np.float32)
    wsum = np.zeros((h, w), np.float32)
    win1 = np.hanning(tile + 2)[1:-1].astype(np.float32)
    win = np.outer(win1, win1) + 1e-3
    with torch.inference_mode():
        for y, x in todo:
            p = rgb[y:y + tile, x:x + tile]
            ph, pw = p.shape[:2]
            p = np.pad(p, ((0, tile - ph), (0, tile - pw), (0, 0)), mode="reflect")
            t = torch.from_numpy(p.transpose(2, 0, 1))[None].cuda().half()
            o = m(t)[0].float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)[:ph, :pw]
            acc[y:y + ph, x:x + pw] += o * win[:ph, :pw, None]
            wsum[y:y + ph, x:x + pw] += win[:ph, :pw]
    out = acc / np.maximum(wsum, 1e-6)[..., None]
    out = np.where(wsum[..., None] > 0, out, rgb)
    ok, buf = cv2.imencode(".png", cv2.cvtColor((out * 255 + 0.5).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def run(jobs):
    """jobs: [(jpeg 字节, 块列表), ...] → [png 字节, ...]。一个临时 app 内并行处理一批照片。"""
    from scripts.learned_denoise import OVERLAP, TILE
    with modal.enable_output(), app.run():
        return list(denoise_remote.starmap([(j, t, TILE, OVERLAP) for j, t in jobs]))
