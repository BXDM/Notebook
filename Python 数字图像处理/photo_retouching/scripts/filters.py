"""各模块共用的小工具。"""
import cv2
import numpy as np


def blur(x, sigma):
    """高斯模糊；半径大时先缩小（倍数 = sigma/4）算再放大回原尺寸：结果只含低频，缩小几乎无损，
    4000 万像素上半径几百的模糊从几十秒降到不到一秒。"""
    f = sigma / 4.0
    if f < 2:
        return cv2.GaussianBlur(x, (0, 0), sigma)
    h, w = x.shape[:2]
    small = cv2.resize(x, (max(1, round(w / f)), max(1, round(h / f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.GaussianBlur(small, (0, 0), 4.0), (w, h), interpolation=cv2.INTER_LINEAR)


def grow(mask, size):
    """椭圆核膨胀（直径 size）的快速等价：距离变换，耗时与 size 无关。mask 为 0/非 0 的 uint8，返回 0/255。"""
    if size <= 25:
        k = max(1, int(size)) | 1
        return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    d = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    return ((d <= size / 2) * 255).astype(np.uint8)


def shrink(mask, size):
    """椭圆核腐蚀的快速等价（见 grow）。"""
    if size <= 25:
        k = max(1, int(size)) | 1
        return cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    d = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    return ((d > size / 2) * 255).astype(np.uint8)


def masked_blur(values, mask, sigma):
    """归一化卷积：只用 mask 内的像素做高斯平均，避免五官、背景把皮肤的参考值拉偏。"""
    mask = mask.astype(np.float32)
    num = blur(values * (mask[..., None] if values.ndim == 3 else mask), sigma)
    den = blur(mask, sigma)
    den = np.maximum(den, 1e-4)
    return num / (den[..., None] if values.ndim == 3 else den)


def taper(d, tau):
    """小偏差几乎全部修正，大偏差（五官结构、立体阴影）逐渐不动：返回 d * exp(-(d/tau)²)。"""
    return d * np.exp(-(d / tau) ** 2)


def to_lab(image):
    """BGR uint8 -> float32 Lab（L 0~100，a/b 约 -128~127）。"""
    return cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def from_lab(lab):
    bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return np.clip(bgr * 255.0 + 0.5, 0, 255).astype(np.uint8)
