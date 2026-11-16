import cv2
import numpy as np


def _skin_lab(pixels):
    """N×3 BGR float -> (彩度, 色相角°) 的中位数。"""
    lab = cv2.cvtColor(np.clip(pixels, 0, 255).astype(np.uint8)[None], cv2.COLOR_BGR2LAB)[0].astype(np.float32)
    a, b = np.median(lab[:, 1]) - 128, np.median(lab[:, 2]) - 128
    return float(np.hypot(a, b)), float(np.degrees(np.arctan2(b, a)))


def color_correct(image, strength=0.3, skin_pixels=None):
    """色彩校正：灰度世界白平衡 + 全局色阶拉伸，按 strength 与原图混合。

    用全局色阶而不是 CLAHE，避免局部对比度增强把皮肤颗粒放大。
    灰度世界在暖色灯光为主的夜景里会把整张图往冷拉，皮肤跟着变灰；给了皮肤像素（N×3 BGR）时，
    只保留不会让肤色变灰、变色的那部分白平衡：皮肤彩度不低于原来的 85%，色相偏移不超过 10°。
    """
    if strength <= 0:
        return image
    img = image.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)
    gains = means.mean() / np.maximum(means, 1e-3)

    wb = 1.0
    if skin_pixels is not None and len(skin_pixels) > 100:
        px = skin_pixels.astype(np.float32)
        if len(px) > 5000:
            px = px[np.linspace(0, len(px) - 1, 5000).astype(int)]
        c0, h0 = _skin_lab(px)
        for t in np.linspace(1.0, 0.0, 11):
            g = 1 + (gains - 1) * t * strength
            c, h = _skin_lab(px * g)
            if c >= 0.85 * c0 and abs((h - h0 + 180) % 360 - 180) <= 10:
                wb = t
                break
        else:
            wb = 0.0
    balanced = img * (1 + (gains - 1) * wb)

    lum = balanced.mean(axis=2)
    lo, hi = np.percentile(lum, [0.5, 99.5])
    if hi - lo > 10:
        balanced = (balanced - lo) * (255.0 / (hi - lo))

    result = img * (1 - strength) + balanced * strength
    return np.clip(result, 0, 255).astype(np.uint8)
