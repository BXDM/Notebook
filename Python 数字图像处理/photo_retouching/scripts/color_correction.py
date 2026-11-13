import numpy as np


def color_correct(image, strength=0.3):
    """色彩校正：灰度世界白平衡 + 全局色阶拉伸，按 strength 与原图混合。

    用全局色阶而不是 CLAHE，避免局部对比度增强把皮肤颗粒放大。
    """
    if strength <= 0:
        return image
    img = image.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)
    balanced = img * (means.mean() / np.maximum(means, 1e-3))

    lum = balanced.mean(axis=2)
    lo, hi = np.percentile(lum, [0.5, 99.5])
    if hi - lo > 10:
        balanced = (balanced - lo) * (255.0 / (hi - lo))

    result = img * (1 - strength) + balanced * strength
    return np.clip(result, 0, 255).astype(np.uint8)
