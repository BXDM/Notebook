"""人脸语义分割：BiSeNet（CelebAMask-HQ 19 类，ResNet18，ONNX），逐张人脸裁剪推理。

模型来自 https://github.com/yakhyo/face-parsing ，缺失时自动下载到 models/resnet18.onnx（约 53 MB）。
不可用时返回 None，调用方退回肤色模型。
"""
import os
import urllib.request

import cv2
import numpy as np

MODEL_URL = "https://github.com/yakhyo/face-parsing/releases/download/v0.0.1/resnet18.onnx"
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "resnet18.onnx")

# CelebAMask-HQ 类别下标
SKIN, L_BROW, R_BROW, L_EYE, R_EYE, GLASSES, L_EAR, R_EAR, EARRING, NOSE, MOUTH, U_LIP, L_LIP, \
    NECK, NECKLACE, CLOTH, HAIR, HAT = range(1, 19)
GROUPS = {
    "face": [SKIN, NOSE],                     # 脸部皮肤（含鼻子）
    "ears": [L_EAR, R_EAR],
    "neck": [NECK],
    "features": [L_BROW, R_BROW, L_EYE, R_EYE, GLASSES, MOUTH, U_LIP, L_LIP],
    "hair": [HAIR, HAT],
    "nose": [NOSE],
    "cloth": [CLOTH, NECKLACE, EARRING],
}
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_session = None  # 懒加载；False 表示不可用


def _load():
    global _session
    if _session is not None:
        return _session
    try:
        import onnxruntime as ort
    except ImportError:
        print("[parsing] 未安装 onnxruntime，皮肤识别退回肤色模型")
        _session = False
        return _session
    if not os.path.exists(MODEL_PATH):
        try:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            print(f"[parsing] 下载人脸分割模型 -> {MODEL_PATH}")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH + ".part")
            os.replace(MODEL_PATH + ".part", MODEL_PATH)
        except Exception as e:
            print(f"[parsing] 模型下载失败（{e}），皮肤识别退回肤色模型")
            _session = False
            return _session
    _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return _session


def parse_faces(image, faces):
    """对每张脸的扩展裁剪区做分割，返回 {组名: 0~1 float32 概率图}（整图尺寸），不可用时返回 None。"""
    session = _load()
    if not session or not faces:
        return None
    h, w = image.shape[:2]
    out = {k: np.zeros((h, w), np.float32) for k in GROUPS}
    out["covered"] = np.zeros((h, w), np.float32)  # 被分割覆盖到的区域
    name = session.get_inputs()[0].name
    for f in faces:
        x, y, bw, bh = f.box
        # 以人脸为中心取 2.2 倍框，下移一点以包含额头到脖子
        cx, cy, half = x + bw / 2, y + bh * 0.55, bw * 1.1
        x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
        x1, y1 = int(min(cx + half, w)), int(min(cy + half, h))
        crop = image[y0:y1, x0:x1]
        inp = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (512, 512), interpolation=cv2.INTER_AREA)
        inp = ((inp.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None]
        logits = session.run(None, {name: inp})[0][0]
        prob = np.exp(logits - logits.max(axis=0))
        prob /= prob.sum(axis=0)
        size = (x1 - x0, y1 - y0)
        for k, ids in GROUPS.items():
            p = cv2.resize(prob[ids].sum(axis=0), size, interpolation=cv2.INTER_LINEAR)
            np.maximum(out[k][y0:y1, x0:x1], p, out=out[k][y0:y1, x0:x1])
        out["covered"][y0:y1, x0:x1] = 1.0
    return out
