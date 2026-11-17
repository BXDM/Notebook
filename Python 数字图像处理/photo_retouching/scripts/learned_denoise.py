"""学习型降噪：SCUNet（真实相机噪声 + JPEG 压缩的盲降噪，Apache-2.0）。

传统滤波对“烤进 JPEG 的斑驳噪点”只能削弱；SCUNet 在这类退化上训练，能去干净又保留纹理。
默认在本机 CPU 上分块推理；整张 24MP 在 CPU 上较慢，所以：
- 只处理需要的区域（mask，比如非皮肤、暗部），其他块直接跳过；
- 疑难（噪声重、需要处理的块多）时由调用方改走 GPU（modal_denoise.py），规则见 plan()。
结果与原图按 keep 混合，保留一点颗粒，避免塑料感。
"""
import os
import time

import cv2
import numpy as np

WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "models", "scunet", "scunet_color_real_psnr.pth")
WEIGHTS_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth"
TILE, OVERLAP = 512, 32
_model = None
try:  # 有 CUDA（比如在云端 GPU 容器里跑整个流程）就直接用，本机没有则用 CPU
    import torch as _torch
    DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"


def _load():
    global _model
    if _model is None:
        import torch
        from scripts.scunet_net import SCUNet
        if not os.path.exists(WEIGHTS):
            import urllib.request
            os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
            print(f"[denoise] 下载 SCUNet 权重 -> {WEIGHTS}")
            urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS + ".part")
            os.replace(WEIGHTS + ".part", WEIGHTS)
        torch.set_num_threads(os.cpu_count() or 4)
        m = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
        m.load_state_dict(torch.load(WEIGHTS, map_location="cpu"), strict=True)
        _model = m.eval().to(DEVICE)
    return _model


def tiles(h, w, tile=TILE, overlap=OVERLAP):
    step = tile - 2 * overlap
    ys = list(range(0, max(h - tile, 0) + 1, step)) or [0]
    xs = list(range(0, max(w - tile, 0) + 1, step)) or [0]
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    return [(y, x) for y in ys for x in xs]


def needed_tiles(mask, tile=TILE, overlap=OVERLAP, frac=0.02):
    """mask（0~1）在块内占比 >= frac 的块才处理。"""
    h, w = mask.shape
    return [(y, x) for y, x in tiles(h, w, tile, overlap) if mask[y:y + tile, x:x + tile].mean() >= frac]


def denoise_tiles(image, todo, model=None):
    """对 todo 中的块做 SCUNet，返回 (降噪图, 覆盖权重)。块之间用余弦窗羽化拼接。"""
    import torch
    model = model or _load()
    h, w = image.shape[:2]
    acc = np.zeros((h, w, 3), np.float32)
    wsum = np.zeros((h, w), np.float32)
    win1 = np.hanning(TILE + 2)[1:-1].astype(np.float32)
    win = np.outer(win1, win1) + 1e-3
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    with torch.inference_mode():
        for y, x in todo:
            patch = rgb[y:y + TILE, x:x + TILE]
            ph, pw = patch.shape[:2]
            pad = np.pad(patch, ((0, TILE - ph), (0, TILE - pw), (0, 0)), mode="reflect")
            t = torch.from_numpy(pad.transpose(2, 0, 1))[None].to(DEVICE)
            o = model(t)[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)[:ph, :pw]
            ww = win[:ph, :pw]
            acc[y:y + ph, x:x + pw] += o * ww[..., None]
            wsum[y:y + ph, x:x + pw] += ww
    out = acc / np.maximum(wsum, 1e-6)[..., None]
    cover = (wsum > 0).astype(np.float32)   # 被任何一块覆盖到就算处理过（不能按最大值归一：重叠处权重叠加会把其他区域压低）
    return cv2.cvtColor((out * 255 + 0.5).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR), cover


def learned_denoise(image, mask=None, keep=0.15, model=None):
    """返回 (结果, 说明)。mask：需要降噪的区域（0~1），None 表示整张。"""
    t0 = time.time()
    h, w = image.shape[:2]
    m = np.ones((h, w), np.float32) if mask is None else mask.astype(np.float32)
    todo = needed_tiles(m)
    if not todo:
        return image, "无需处理的区域"
    den, cover = denoise_tiles(image, todo, model)
    a = (cv2.GaussianBlur(m, (0, 0), 8) * cover * (1 - keep))[..., None]
    out = (image.astype(np.float32) * (1 - a) + den.astype(np.float32) * a + 0.5).clip(0, 255).astype(np.uint8)
    return out, f"SCUNet（CPU）{len(todo)} 块，{time.time() - t0:.0f}s"


CPU_SEC_PER_TILE = 5.7     # 16 核 CPU 实测（512×512 一块）
GPU_THRESHOLD_SEC = 300    # 预计 CPU 超过 5 分钟才算“疑难”，交给 GPU


def smart_denoise(image, protect=None, gpu="auto", iso=None, min_noise=0.8, keep=0.15, min_iso=800):
    """按需选择：ISO < 800（有 EXIF 时）或噪声不可见 → 跳过；需要的块少 → 本机 CPU；预计 CPU 超过 5 分钟 → Modal GPU（失败退回 CPU）。
    protect（0~1）处不降噪（人像皮肤已由磨皮处理）。没装 torch 时退回传统滤波。返回 (结果, 说明)。"""
    from scripts.sharpening import detail_noise
    try:
        import torch  # noqa: F401
    except ImportError:
        from scripts.denoise import denoise_mottle
        out, how = denoise_mottle(image, protect)
        return out, how + "（未装 torch，用传统滤波）"
    # 先看 ISO：噪声由 ISO 决定，ISO < 800 的相机照片没有需要学习型降噪的噪声。
    # 只看像素统计会把稻田、树叶这类细纹理当成噪声（ISO 100 的风景测出过 4.4），降噪后纹理被抹成油画感
    if iso is not None and iso < min_iso:
        return image, f"ISO {iso:.0f}，跳过"
    Y = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
    sigma = detail_noise(Y - cv2.GaussianBlur(Y, (0, 0), 2.0), Y)
    if sigma < (min_noise if iso is not None else 1.5 * min_noise):  # 没有 EXIF 时门槛提高
        return image, f"噪声低（{sigma:.2f}），跳过"
    h, w = image.shape[:2]
    mask = np.ones((h, w), np.float32) if protect is None else (1 - protect).astype(np.float32)
    todo = needed_tiles(mask)
    if not todo:
        return image, "无需处理的区域"
    est = CPU_SEC_PER_TILE * len(todo)
    use_gpu = DEVICE == "cpu" and (gpu == "always" or (gpu == "auto" and est > GPU_THRESHOLD_SEC))
    t0 = time.time()
    den = None
    where = "本机 GPU" if DEVICE == "cuda" else "CPU"
    if use_gpu:
        try:
            from scripts import modal_denoise
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 98])
            png = modal_denoise.run([(buf.tobytes(), todo)])[0]
            den = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
            where = "GPU"
        except Exception as e:  # 网络、Modal 额度等问题：退回本机
            print(f"[denoise] GPU 失败，改用 CPU（{type(e).__name__}: {e}）")
    if den is None:
        den, _ = denoise_tiles(image, todo)
    a = (cv2.GaussianBlur(mask, (0, 0), 8) * (1 - keep))[..., None]
    out = (image.astype(np.float32) * (1 - a) + den.astype(np.float32) * a + 0.5).clip(0, 255).astype(np.uint8)
    return out, f"噪声 {sigma:.2f}，SCUNet {where} {len(todo)} 块（预计 CPU {est / 60:.0f} 分钟），{time.time() - t0:.0f}s"
