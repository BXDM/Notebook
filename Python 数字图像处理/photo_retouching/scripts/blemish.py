import cv2
import numpy as np


# 人脸框内不做祛痘的区域（相对坐标 x0, y0, x1, y1）：眉眼、鼻孔、嘴
_FACE_EXCLUDE = [(0.0, 0.12, 1.0, 0.52),
                 (0.30, 0.52, 0.70, 0.74), (0.22, 0.68, 0.78, 0.95)]


def remove_blemishes(image, hard_mask, faces, scale, sensitivity=1.0):
    """祛痘祛痣：在皮肤区域内找比周围暗的小斑点，用 inpaint 修补。

    返回 (修复后图像, 斑点掩码)。sensitivity 越大检出越多，0 关闭。
    """
    if sensitivity <= 0:
        return image, np.zeros(image.shape[:2], np.uint8)

    # 先轻微模糊，避免把噪点、胶片颗粒当成斑点
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (0, 0), max(1.0, scale / 250))
    k = max(5, int(scale / 18) | 1)  # 斑点直径上限约为脸宽的 1/18
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # 只看完全处于皮肤内部的点，避开眉毛、眼睛、鼻孔、发际线等边缘
    inner = cv2.erode(hard_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1)))
    for x, y, w, h in faces:
        for rx0, ry0, rx1, ry1 in _FACE_EXCLUDE:
            inner[int(y + ry0 * h):int(y + ry1 * h), int(x + rx0 * w):int(x + rx1 * w)] = 0
    vals = blackhat[inner > 0]
    if vals.size == 0:
        return image, np.zeros(image.shape[:2], np.uint8)
    # 斑点要比周围暗得足够明显：取皮肤内 blackhat 分布的高分位，并设绝对下限
    thresh = max(12.0, float(np.percentile(vals, 99.5))) / sensitivity
    spots = ((blackhat > thresh) & (inner > 0)).astype(np.uint8) * 255

    # 过滤：面积不能太大，形状要接近圆点（排除皱纹、眉毛边缘这类细长或弯曲的形状）
    n, labels, stats, _ = cv2.connectedComponentsWithStats(spots)
    max_area = np.pi * (k / 2) ** 2
    min_area = max(3.0, (scale / 150) ** 2)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        w, h, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        keep[i] = (min_area <= area <= max_area and max(w, h) <= 2.5 * min(w, h)
                   and area >= 0.4 * w * h)
    spots = np.where(keep[labels], 255, 0).astype(np.uint8)
    spots = cv2.dilate(spots, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)

    radius = max(3, k // 3)
    return cv2.inpaint(image, spots, radius, cv2.INPAINT_TELEA), spots
