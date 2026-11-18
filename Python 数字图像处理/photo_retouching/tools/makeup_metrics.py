"""妆容保真指标：腮红区域的色差是否保住、位置是否不变。

    python tools/makeup_metrics.py 原图.jpg 结果1.jpg [结果2.jpg ...]

腮红区域在原图上定：脸部皮肤内，大尺度（脸宽 1/30 平滑）色度相对脸部中位色更红（a* 高出 >= 3）的连片区域。
- 腮红色差：腮红区与其余脸部皮肤的平均色差 ΔE00（只用色度，亮度置为相同，排除提亮的影响），结果/原图；
- 位置相关：原图与结果的“a*、b* 相对脸部中位色的偏离图”的空间相关系数（1 = 分布完全一致）；
- 腮红色准：腮红区内结果与原图的 ΔE00 中位数（同样只比色度）。
蒙版取自原图；结果与原图须同尺寸（人像流程不裁剪）。
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.face import detect_faces  # noqa: E402
from scripts.preprocess import preprocess_image  # noqa: E402
from scripts.skin_mask import skin_mask  # noqa: E402


def de00(L1, a1, b1, L2, a2, b2):
    """CIEDE2000（向量化）。"""
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p, h2p = np.degrees(np.arctan2(b1, a1p)) % 360, np.degrees(np.arctan2(b2, a2p)) % 360
    dL, dC = L2 - L1, C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dH = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lb, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    hs = h1p + h2p
    hb = np.where(np.abs(h1p - h2p) > 180, np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2), hs / 2)
    T = 1 - 0.17 * np.cos(np.radians(hb - 30)) + 0.24 * np.cos(np.radians(2 * hb)) \
        + 0.32 * np.cos(np.radians(3 * hb + 6)) - 0.20 * np.cos(np.radians(4 * hb - 63))
    SL = 1 + 0.015 * (Lb - 50) ** 2 / np.sqrt(20 + (Lb - 50) ** 2)
    SC, SH = 1 + 0.045 * Cbp, 1 + 0.015 * Cbp * T
    RT = -2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7)) * np.sin(np.radians(60 * np.exp(-((hb - 275) / 25) ** 2)))
    return np.sqrt((dL / SL) ** 2 + (dC / SC) ** 2 + (dH / SH) ** 2 + RT * (dC / SC) * (dH / SH))


def blush_regions(img, faces, masks):
    w = faces[0].box[2]
    lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    face = cv2.erode(masks.face_hard, np.ones((9, 9), np.uint8)) > 0
    a = cv2.GaussianBlur(lab[..., 1], (0, 0), w / 30)
    blush = face & (a - np.median(a[face]) >= 3)
    return face, blush, w


def measure(img, ref_lab, face, blush, w):
    lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    g = lambda x: cv2.GaussianBlur(x, (0, 0), w / 30)
    a, b = g(lab[..., 1]), g(lab[..., 2])
    rest = face & ~blush
    am, bm = np.median(a[rest]), np.median(b[rest])
    contrast = float(de00(60, a[blush].mean(), b[blush].mean(), 60, am, bm))
    da, db = (a - np.median(a[face]))[face], (b - np.median(b[face]))[face]
    fa, fb = ref_lab
    corr = float(np.corrcoef(np.r_[da, db], np.r_[fa, fb])[0, 1])
    return contrast, corr, a, b


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ref = preprocess_image(sys.argv[1])
    s = 3000 / max(ref.shape[:2])
    small = lambda x: cv2.resize(x, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    r = small(ref)
    faces = detect_faces(r)
    face, blush, w = blush_regions(r, faces, skin_mask(r, faces))
    lab0 = cv2.cvtColor(r.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    g = lambda x: cv2.GaussianBlur(x, (0, 0), w / 30)
    a0, b0 = g(lab0[..., 1]), g(lab0[..., 2])
    ref_dev = ((a0 - np.median(a0[face]))[face], (b0 - np.median(b0[face]))[face])
    c0, _, _, _ = measure(r, ref_dev, face, blush, w)
    print(f"腮红区占脸部 {blush.sum() / face.sum():.0%}，原图腮红色差 ΔE00 {c0:.2f}")
    for p in sys.argv[2:]:
        c, corr, a, b = measure(small(preprocess_image(p)), ref_dev, face, blush, w)
        acc = float(np.median(de00(60, a0[blush], b0[blush], 60, a[blush], b[blush])))
        print(f"{os.path.basename(os.path.dirname(p)) or p:<12} 腮红色差保留 {c / c0:5.0%}  位置相关 {corr:.2f}  腮红色准 ΔE00 {acc:.2f}")


if __name__ == "__main__":
    main()
