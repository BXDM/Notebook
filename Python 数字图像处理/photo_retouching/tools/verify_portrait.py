"""人像精修验收：同一张照片的原图与成片逐像素比较（不跨照片比较），三项：

1. 腮红保护：腮红区（原图上定，见 makeup_metrics）的颜色 ΔE00（只比色度）、位置相关、边缘渐变保留
   （腮红边界带内 a*/b* 低频梯度幅度，成片/原图）。
2. 中频纹理：脸颊（脸部皮肤去掉瑕疵点及其周围）中频亮度起伏保留（亮度先按均值归一，排除提亮）；
   瑕疵抑制：每个检出瑕疵中心与周围环带的亮度差，成片/原图（越小越好）。
3. 分区肤色连续性：脸↔脖子/身体、身体↔手的交界两侧各取一条带，比较两侧平均色差 ΔE00，
   成片 − 原图 = 蒙版造成的新增跳变（光源造成的差异原图里就有，不算）。

    python tools/verify_portrait.py 原图目录 成片目录 [调试目录]
调试目录给出时读取 report.txt 里的瑕疵坐标，并以“曝光之后”（01_exposure.jpg）为参照——三项都只考核
修图步骤带来的变化；曝光本身会把脸、身体原有的亮度差和色差在 Lab 里等比放大（提亮 4 倍约放大 1.6 倍），
不能算成蒙版问题。没有调试目录时以原图为参照。
"""
import glob
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from makeup_metrics import de00  # noqa: E402
from scripts.face import detect_faces  # noqa: E402
from scripts.hands import hand_mask  # noqa: E402
from scripts.skin_mask import skin_mask  # noqa: E402

WORK = 3000


def lab_of(img):
    return cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)


def blush_items(l0, l1, face, w):
    g = lambda x: cv2.GaussianBlur(x, (0, 0), w / 30)
    a0, b0, a1, b1 = g(l0[..., 1]), g(l0[..., 2]), g(l1[..., 1]), g(l1[..., 2])
    blush = face & (a0 - np.median(a0[face]) >= 3)
    if blush.sum() < 200:
        return None
    color = float(np.median(de00(60, a0[blush], b0[blush], 60, a1[blush], b1[blush])))
    d0 = np.r_[(a0 - np.median(a0[face]))[face], (b0 - np.median(b0[face]))[face]]
    d1 = np.r_[(a1 - np.median(a1[face]))[face], (b1 - np.median(b1[face]))[face]]
    corr = float(np.corrcoef(d0, d1)[0, 1])
    k = np.ones((max(3, int(w / 25)) | 1,) * 2, np.uint8)
    edge = (cv2.dilate(blush.astype(np.uint8), k) - cv2.erode(blush.astype(np.uint8), k)).astype(bool) & face
    grad = lambda a, b: np.hypot(*np.gradient(a)) + np.hypot(*np.gradient(b))
    gr = float(grad(a1, b1)[edge].mean() / max(grad(a0, b0)[edge].mean(), 1e-6))
    return color, corr, gr


def blemish_points(report, s):
    if not report or not os.path.exists(report):
        return []
    pts = []
    for line in open(report, encoding="utf-8"):
        m = re.match(r"\s+(acne|spot|freckle|mole|mark)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+([\d.]+)", line)
        if m and float(m.group(5)) > 0:
            pts.append((int(m.group(2)) * s, int(m.group(3)) * s, float(m.group(4)) * s))
    return pts


def texture_items(l0, l1, face, w, pts):
    L0, L1 = l0[..., 0], l1[..., 0]
    cheek = face.copy()
    for x, y, r in pts:
        cv2.circle(cheek.view(np.uint8), (int(x), int(y)), int(r * 3 + 4), 0, -1)
    g = lambda x, s: cv2.GaussianBlur(x, (0, 0), max(1.0, s))
    n0, n1 = L0[cheek].mean(), L1[cheek].mean()
    mid = lambda L, n: ((g(L, w / 250) - g(L, w / 40)) / n)[cheek].std()
    keep = float(mid(L1, n1) / max(mid(L0, n0), 1e-6))
    sup = []
    for x, y, r in pts:
        x, y, r = int(x), int(y), max(2, int(r))
        yy, xx = np.ogrid[:L0.shape[0], :L0.shape[1]]
        dist = np.hypot(xx - x, yy - y)
        core, ring = dist <= r, (dist > 1.8 * r) & (dist <= 3 * r)
        if ring.sum() < 8 or core.sum() < 3:
            continue
        c0 = abs(L0[ring].mean() / n0 - L0[core].mean() / n0)
        c1 = abs(L1[ring].mean() / n1 - L1[core].mean() / n1)
        if c0 > 1e-3:
            sup.append(c1 / c0)
    return keep, (float(np.median(sup)) if sup else None), len(sup)


def seam_items(l0, l1, regions, w):
    """regions: {名字: bool 蒙版}。两两相邻的交界带两侧的色差：成片 − 原图（ΔE00，越接近 0 越好）。"""
    out = {}
    k = np.ones((max(3, int(w / 20)) | 1,) * 2, np.uint8)
    g = lambda x: cv2.GaussianBlur(x, (0, 0), w / 60)
    L = [[g(l[..., i]) for i in range(3)] for l in (l0, l1)]
    names = list(regions)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A, B = regions[a].astype(np.uint8), regions[b].astype(np.uint8)
            touch = cv2.dilate(A, k) & cv2.dilate(B, k)
            if touch.sum() < 50:
                continue
            ba, bb = (A > 0) & (touch > 0), (B > 0) & (touch > 0)
            if ba.sum() < 20 or bb.sum() < 20:
                continue
            d = []
            for c in L:
                d.append(float(de00(c[0][ba].mean(), c[1][ba].mean(), c[2][ba].mean(),
                                    c[0][bb].mean(), c[1][bb].mean(), c[2][bb].mean())))
            out[f"{a}↔{b}"] = (d[0], d[1])
    return out


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    dbg = sys.argv[3] if len(sys.argv) > 3 else None
    for f in sorted(glob.glob(os.path.join(src, "*.jpg")) + glob.glob(os.path.join(src, "*.JPG"))):
        n = os.path.basename(f)
        p1 = os.path.join(dst, n)
        if not os.path.exists(p1):
            continue
        im0, im1 = cv2.imread(f), cv2.imread(p1)
        ref = "原图"
        if dbg:
            ex = glob.glob(os.path.join(dbg, os.path.splitext(n)[0], "*_exposure.jpg"))
            if ex:
                im0, ref = cv2.imread(ex[0]), "曝光后"
        s = WORK / max(im0.shape[:2])
        r0 = cv2.resize(im0, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        r1 = cv2.resize(im1, (r0.shape[1], r0.shape[0]), interpolation=cv2.INTER_AREA)
        faces = detect_faces(r0)
        if not faces:
            print(n, "没检测到人脸")
            continue
        w = faces[0].box[2]
        hand, _ = hand_mask(r0)
        m = skin_mask(r0, faces, hand=hand)
        face = cv2.erode(m.face_hard, np.ones((9, 9), np.uint8)) > 0
        l0, l1 = lab_of(r0), lab_of(r1)
        tag = n[-10:-4]
        b = blush_items(l0, l1, face, w)
        pts = blemish_points(os.path.join(dbg, os.path.splitext(n)[0], "report.txt") if dbg else None, s)
        keep, sup, ns = texture_items(l0, l1, face, w, pts)
        regions = {"脸": face, "身体": (m.skin_hard > 0) & ~(m.face_hard > 0)}
        if hand is not None:
            regions["手"] = hand > 0
        seams = seam_items(l0, l1, regions, w)
        bl = (f"腮红 色差 {b[0]:.2f} 位置 {b[1]:.2f} 渐变 {b[2]:.0%}" if b else "腮红 无")
        tx = f"中频 {keep:.0%}" + (f" 瑕疵对比 {sup:.0%}（{ns} 个）" if sup is not None else "")
        sm = "；".join(f"{k} {v[0]:.1f}→{v[1]:.1f}" for k, v in seams.items()) or "无交界"
        print(f"{tag} [参照{ref}] | {bl} | {tx} | 交界 ΔE00 参照→成片 {sm}")


if __name__ == "__main__":
    main()
