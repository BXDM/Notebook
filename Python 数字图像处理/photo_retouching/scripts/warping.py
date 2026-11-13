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


# 下颌轮廓上的推点（左右成对）：颧骨下 → 下颌角 → 下巴两侧
_JAW_PAIRS = [(132, 361), (58, 288), (172, 397), (136, 365), (150, 379)]


def slim_face(image, faces, strength=0.3):
    """瘦脸：把下颌轮廓两侧向脸中线做局部液化。

    有关键点时沿真实下颌线推、方向指向鼻尖；只有 Haar 框时按框估计下颌位置。
    """
    if strength <= 0 or not faces:
        return image
    h_img, w_img = image.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w_img, dtype=np.float32), np.arange(h_img, dtype=np.float32))
    for f in faces:
        x, y, w, h = f.box
        if f.points is not None:
            pts = f.points
            face_w = np.linalg.norm(pts[454] - pts[234])
            radius = face_w * 0.22
            shift = face_w * 0.035 * strength  # 相邻推点半径重叠，总位移约为单点的 2 倍
            center = pts[4]  # 鼻尖
            for pair in _JAW_PAIRS:
                for i in pair:
                    p = pts[i]
                    d = center - p
                    target = p + d / (np.linalg.norm(d) + 1e-6) * shift
                    _local_translate(map_x, map_y, tuple(p), tuple(target), radius)
        else:
            radius = w * 0.28
            shift = w * 0.07 * strength
            cy = y + h * 0.78
            for side in (-1, 1):
                cx = x + w / 2 + side * w * 0.40
                _local_translate(map_x, map_y, (cx, cy), (cx - side * shift, cy), radius)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
