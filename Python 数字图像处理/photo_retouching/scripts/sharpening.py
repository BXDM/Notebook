import cv2
import numpy as np


def detail_noise(detail, luma, block=16):
    """细节层里的噪声水平：中低亮度（夜景噪点最显眼）平坦块的标准差，取 20 分位。"""
    h, w = detail.shape
    h, w = h - h % block, w - w % block
    st = detail[:h, :w].reshape(h // block, block, w // block, block).std(axis=(1, 3))
    m = luma[:h, :w].reshape(h // block, block, w // block, block).mean(axis=(1, 3))
    sel = st[(m > 15) & (m < 160)]
    return float(np.percentile(sel if sel.size > 50 else st, 20))


def sharpen_image(image, soft_mask, scale, amount=0.5):
    """USM 锐化，皮肤区域不锐化，主要锐化眼睛、眉毛、头发等细节。
    按噪声取芯（coring）：细节幅度接近噪声的地方不锐化，夜景高 ISO 的颗粒不会被放大。"""
    if amount <= 0:
        return image
    img = image.astype(np.float32)
    blur = cv2.GaussianBlur(img, (0, 0), max(1.0, scale / 400))
    d = img - blur
    dl = d.mean(axis=2)
    sigma = detail_noise(dl, img.mean(axis=2))
    mag = cv2.GaussianBlur(np.abs(dl), (0, 0), 1.0)
    t = (3.0 * sigma) ** 2
    core = mag ** 2 / (mag ** 2 + t)
    weight = (amount * (1 - soft_mask) * core)[..., None]
    result = img + d * weight
    return np.clip(result, 0, 255).astype(np.uint8)
