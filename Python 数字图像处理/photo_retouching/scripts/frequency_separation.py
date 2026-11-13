import cv2
import numpy as np


def frequency_separation(image, soft_mask, scale, strength=0.5):
    """高低频分离匀肤：只平滑低频（肤色、明暗斑块），保留高频（毛孔纹理）。"""
    if strength <= 0:
        return image
    img = image.astype(np.float32)
    sigma = max(1.5, scale / 60)
    low = cv2.GaussianBlur(img, (0, 0), sigma)
    high = img - low
    # 低频层用保边滤波压平色块，避免皮肤边缘晕出去；低频信息少，缩小后再滤波以提速
    f = min(1.0, 4.0 / sigma)
    small = cv2.resize(low, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    small = cv2.bilateralFilter(small, d=0, sigmaColor=20, sigmaSpace=max(2.0, scale / 12 * f))
    low_even = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    alpha = (soft_mask * strength)[..., None]
    result = low + (low_even - low) * alpha + high
    return np.clip(result, 0, 255).astype(np.uint8)
