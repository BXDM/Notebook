"""各模块共用的小工具。"""
import cv2
import numpy as np


def masked_blur(values, mask, sigma):
    """归一化卷积：只用 mask 内的像素做高斯平均，避免五官、背景把皮肤的参考值拉偏。"""
    mask = mask.astype(np.float32)
    num = cv2.GaussianBlur(values * (mask[..., None] if values.ndim == 3 else mask), (0, 0), sigma)
    den = cv2.GaussianBlur(mask, (0, 0), sigma)
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
