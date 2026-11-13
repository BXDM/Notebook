import cv2
import numpy as np


def preprocess_image(image_path, max_side=None):
    """读取图像（支持中文路径），返回 BGR uint8；可按长边等比缩小。"""
    data = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    if max_side and max(image.shape[:2]) > max_side:
        scale = max_side / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image
