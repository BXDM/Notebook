import cv2
import numpy as np


def _local_translate(map_x, map_y, center, target, radius):
    """局部平移变形（液化“向前推”）：把 center 附近的像素推向 target。

    位移按 (1 - d²/r²)² 平滑衰减，到半径边缘为 0；位移远小于半径时不会折叠。
    """
    cx, cy = center
    dx, dy = target[0] - cx, target[1] - cy
    x0, x1 = int(max(cx - radius, 0)), int(min(cx + radius + 1, map_x.shape[1]))
    y0, y1 = int(max(cy - radius, 0)), int(min(cy + radius + 1, map_x.shape[0]))
    xs, ys = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
    t = np.clip(((xs - cx) ** 2 + (ys - cy) ** 2) / (radius * radius), 0, 1)
    weight = (1 - t) ** 2
    map_x[y0:y1, x0:x1] -= weight * dx
    map_y[y0:y1, x0:x1] -= weight * dy


def slim_face(image, faces, strength=0.3):
    """瘦脸：在人脸框两侧下颌处向脸中线做局部液化。依赖 Haar 人脸框，位置是估计的。"""
    if strength <= 0 or not faces:
        return image
    h_img, w_img = image.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w_img, dtype=np.float32), np.arange(h_img, dtype=np.float32))
    for x, y, w, h in faces:
        radius = w * 0.28
        shift = w * 0.07 * strength
        cy = y + h * 0.78
        for side in (-1, 1):
            cx = x + w / 2 + side * w * 0.40
            _local_translate(map_x, map_y, (cx, cy), (cx - side * shift, cy), radius)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
