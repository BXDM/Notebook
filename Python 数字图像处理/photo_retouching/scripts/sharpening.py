import cv2
import numpy as np


def sharpen_image(image, soft_mask, scale, amount=0.5):
    """USM 锐化，皮肤区域不锐化（避免放大颗粒），主要锐化眼睛、眉毛、头发等细节。"""
    if amount <= 0:
        return image
    img = image.astype(np.float32)
    blur = cv2.GaussianBlur(img, (0, 0), max(1.0, scale / 400))
    weight = (amount * (1 - soft_mask))[..., None]
    result = img + (img - blur) * weight
    return np.clip(result, 0, 255).astype(np.uint8)
