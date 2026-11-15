"""人像多类分割：mediapipe selfie_multiclass（背景/头发/身体皮肤/脸部皮肤/衣物/其他）。

人脸分割只覆盖脸周围约 2 倍脸宽，脖子以下的衣服没有标签，只靠肤色会把浅色衣服算成皮肤；
这里在每张脸下方取一块包含上半身的区域做分割，给身体皮肤一个上限。缺模型时返回 None，退回原逻辑。
"""
import os
import urllib.request

import cv2
import numpy as np

from scripts.face import _quiet_stderr

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
             "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "selfie_multiclass_256x256.tflite")
CLASSES = {"hair": 1, "body": 2, "face": 3, "cloth": 4}
_options = None


def _load_options():
    global _options
    if _options is not None:
        return _options
    try:
        with _quiet_stderr():
            from mediapipe.tasks.python import BaseOptions, vision
    except ImportError:
        _options = False
        return _options
    if not os.path.exists(MODEL_PATH):
        try:
            print(f"[person] 下载人像分割模型 -> {MODEL_PATH}")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH + ".part")
            os.replace(MODEL_PATH + ".part", MODEL_PATH)
        except Exception as e:
            print(f"[person] 模型下载失败（{e}），身体皮肤只用肤色模型")
            _options = False
            return _options
    _options = vision.ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=MODEL_PATH),
                                            output_confidence_masks=True)
    return _options


def segment_person(image, faces):
    """返回 {类名: 0~1 float32 概率图}（整图尺寸，分割区域外为 0），外加 "covered"；不可用时返回 None。"""
    options = _load_options()
    if not options or not faces:
        return None
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    h, w = image.shape[:2]
    out = {k: np.zeros((h, w), np.float32) for k in CLASSES}
    out["covered"] = np.zeros((h, w), np.float32)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with _quiet_stderr(), vision.ImageSegmenter.create_from_options(options) as segmenter:
        for f in faces:
            x, y, bw, _ = f.box
            # 横向左右各 3 倍脸宽，纵向从额头上 1 倍到下方 7 倍：坐姿、半身都能覆盖到手
            cx = x + bw / 2
            x0, x1 = int(max(cx - 3 * bw, 0)), int(min(cx + 3 * bw, w))
            y0, y1 = int(max(y - bw, 0)), int(min(y + 7 * bw, h))
            crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
            result = segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
            masks = result.confidence_masks
            for k, idx in CLASSES.items():
                p = masks[idx].numpy_view().reshape(y1 - y0, x1 - x0)
                np.maximum(out[k][y0:y1, x0:x1], p, out=out[k][y0:y1, x0:x1])
            out["covered"][y0:y1, x0:x1] = 1.0
    return out
