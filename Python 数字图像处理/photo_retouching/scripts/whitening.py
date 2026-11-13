import cv2
import numpy as np


def whiten_skin(image, soft_mask, strength=0.3):
    """美白提亮：在 LAB 的 L 通道上做对数曲线提亮，按皮肤掩码混合。"""
    if strength <= 0:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    beta = 1 + 4 * strength
    l = lab[..., 0] / 255.0
    l_bright = np.log1p(l * (beta - 1)) / np.log(beta)
    lab[..., 0] = (l + (l_bright - l) * soft_mask) * 255.0
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out
