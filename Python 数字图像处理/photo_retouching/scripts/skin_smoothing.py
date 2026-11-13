import cv2
import numpy as np


def skin_smoothing(image, soft_mask, scale, strength=0.6):
    """保边磨皮：双边滤波后按皮肤掩码混合，再补回少量细节防止“塑料感”。"""
    if strength <= 0:
        return image
    img = image.astype(np.float32)
    smooth = cv2.bilateralFilter(image, d=0, sigmaColor=40, sigmaSpace=max(3.0, scale / 40)).astype(np.float32)
    # 只把细节中较弱的部分（瑕疵、噪点）磨掉，保留 15% 纹理
    detail = img - smooth
    smooth = smooth + detail * 0.15
    alpha = (soft_mask * strength)[..., None]
    result = img * (1 - alpha) + smooth * alpha
    return np.clip(result, 0, 255).astype(np.uint8)
