import os

import cv2


def save_image(image, output_path, quality=95):
    """保存 BGR 图像（支持中文路径）。"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise ValueError(f"无法编码图像: {output_path}")
    buf.tofile(output_path)
