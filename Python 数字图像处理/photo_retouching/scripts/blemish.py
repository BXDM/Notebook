"""瑕疵检测、分类与纹理感知修复。

检测：皮肤内比周围暗（L 通道 blackhat）或更红（a 通道 tophat）的小块。
分类（按大小、明暗差 dL、红度差 da）：
  acne    痘痘/红点      红度占主导                       -> 修复
  mole    痣             很暗且有一定大小                 -> 默认保留（--moles 去除）
  freckle 雀斑           小而浅                           -> 按 --freckles 减淡，默认保留
  spot    小色斑/暗斑    其余                             -> 减淡 70%
  mark    痘印           小而浅、但不成“雀斑分布”或偏红    -> 按 marks 减淡（人像配置 0.6；默认 0 = 1.x 行为，仍算雀斑）
“雀斑分布”：小而浅的斑点 ≥ 6 个、鼻梁两侧都有且较少一侧 ≥ 较多一侧的 30%——真雀斑（含妆面画的雀斑）成片对称分布，
算身份特征保留；普通人的痘印、小色斑零散、常集中在一侧（2026-11-19 IMG_5870：10 个“雀斑”有 9 个在同一侧脸颊），
按痘印减淡。成片雀斑里偏红的（红度差 ≥ MARK_RED）也按痘印。
修复：低频用 inpaint 补平，高频纹理从邻近干净皮肤拷贝过来（类似修复画笔），避免出现光滑补丁。
"""
from dataclasses import dataclass

import cv2
import numpy as np

from scripts.filters import grow, shrink

# 没有分割和关键点时，人脸框内不做祛痘的区域（相对坐标 x0, y0, x1, y1）：眉眼、鼻孔、嘴
_FACE_EXCLUDE = [(0.0, 0.12, 1.0, 0.52), (0.30, 0.52, 0.70, 0.74), (0.22, 0.68, 0.78, 0.95)]
_DIRS = [(np.cos(a), np.sin(a)) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]


@dataclass
class Blemish:
    kind: str
    center: tuple
    radius: float
    dL: float
    da: float
    ring_std: float  # 周围皮肤亮度起伏；褶皱、凹陷处很大
    weight: float  # 实际修复强度，0 表示保留


def _hat(x, op, k, scale, face_px=700):
    """顶帽/黑帽：脸宽超过 face_px 时缩小到约 face_px 再算、放大回来——只是找斑点的对比，
    缩小后斑点仍有好几个像素；大核形态学在 4000 万像素原图上要几十秒。"""
    f = scale / face_px
    if f <= 1.5:
        return cv2.morphologyEx(x, op, _ellipse(k))
    h, w = x.shape
    small = cv2.resize(x, (round(w / f), round(h / f)), interpolation=cv2.INTER_AREA)
    r = cv2.morphologyEx(small, op, _ellipse(max(3, int(k / f) | 1)))
    return cv2.resize(r, (w, h), interpolation=cv2.INTER_LINEAR)


def _ellipse(k):
    k = max(1, int(k)) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _classify(d, dL, da, scale):
    """返回类别；对比太弱的返回 None（噪声或正常肤色起伏）。dL、da 为 Lab uint8 单位。"""
    if da >= 5.0 and da >= dL * 0.35:
        return "acne"
    if dL < 5.0:
        return None
    if dL >= 40.0 and d >= scale / 90:
        return "mole"
    if dL < 14.0 and d <= scale / 45:
        return "freckle"
    return "spot"


def _compact(area, bw, bh, min_area, max_area):
    """形状要接近圆点：排除皱纹、发丝、眉毛边缘这类细长或弯曲的形状。"""
    return min_area <= area <= max_area and max(bw, bh) <= 2.5 * min(bw, bh) and area >= 0.4 * bw * bh


def _candidates(dark, red, inner, t_dark, t_red, min_area, max_area, trace):
    """返回候选斑点 [(x, y, w, h, roi 掩码, 质心, 面积)]。

    连在大片阴影（鼻翼旁、法令纹）上的斑点会和阴影连成一大块；对这类不合格的大块，
    在它内部逐级提高阈值，把更深、更紧凑的核心分离出来。
    """
    out = []
    stack = [(0, 0, dark.shape[1], dark.shape[0], 1.0)]
    while stack:
        X, Y, Wd, Ht, level = stack.pop()
        d_roi, r_roi = dark[Y:Y + Ht, X:X + Wd], red[Y:Y + Ht, X:X + Wd]
        m = (((d_roi > t_dark * level) | (r_roi > t_red * level)) & (inner[Y:Y + Ht, X:X + Wd] > 0))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        for i in range(1, n):
            bx, by, bw, bh, area = stats[i]
            if area < min_area:
                continue
            if _compact(area, bw, bh, min_area, max_area):
                mask = (labels[by:by + bh, bx:bx + bw] == i).astype(np.uint8) * 255
                out.append((X + bx, Y + by, bw, bh, mask, (X + cents[i][0], Y + cents[i][1]), area))
            elif level < 4.0:
                stack.append((X + bx, Y + by, bw, bh, level * 1.6))
            elif trace is not None:
                trace.append(((X + cents[i][0], Y + cents[i][1]), f"形状/大小 area={area} {bw}x{bh}"))
    return out


def _heal(img, high, region, blob, cand, d, weight):
    """修复一个斑点：低频 inpaint + 邻近皮肤的高频纹理，原地写入 img（float32）。"""
    h, w = region.shape
    bx, by, bw, bh = cv2.boundingRect(blob)
    pad = int(d * 3) + 4
    Y0, Y1, X0, X1 = max(0, by - pad), min(h, by + bh + pad), max(0, bx - pad), min(w, bx + bw + pad)
    roi_m = grow(blob[Y0:Y1, X0:X1], max(3, d * 0.6))
    roi = img[Y0:Y1, X0:X1]

    base = cv2.inpaint(np.clip(roi, 0, 255).astype(np.uint8), roi_m, max(3, int(d)), cv2.INPAINT_TELEA)
    base = cv2.GaussianBlur(base.astype(np.float32), (0, 0), max(1.0, d / 5))

    # 找一个纹理供体：沿 12 个方向平移，要求整块落在干净皮肤内，取亮度最接近的
    my, mx = np.nonzero(roi_m)
    best, best_cost = None, None
    ring_l = roi[grow(roi_m, d) > roi_m].mean()
    for dist in (1.6 * d + 3, 2.4 * d + 4):
        for cx, cy in _DIRS:
            oy, ox = int(round(cy * dist)), int(round(cx * dist))
            ty, tx = my + Y0 + oy, mx + X0 + ox
            if ty.min() < 0 or tx.min() < 0 or ty.max() >= h or tx.max() >= w:
                continue
            if region[ty, tx].min() == 0 or cand[ty, tx].max() > 0:
                continue
            cost = abs(img[ty, tx].mean() - ring_l)
            if best_cost is None or cost < best_cost:
                best, best_cost = (oy, ox), cost
        if best is not None:
            break

    tex = np.zeros_like(roi)
    if best is not None:
        oy, ox = best
        tex[my, mx] = high[my + Y0 + oy, mx + X0 + ox]
    healed = base + tex
    alpha = cv2.GaussianBlur(roi_m.astype(np.float32) / 255, (0, 0), max(1.0, d / 4)) * weight
    alpha = np.minimum(alpha * 1.5, weight)[..., None]  # 斑点内部全量替换，边缘过渡
    img[Y0:Y1, X0:X1] = roi * (1 - alpha) + healed * alpha


MARK_MIN_PATTERN = 6
MARK_RED = 3.0


def _freckle_pattern(found, faces):
    """小而浅的斑点是否成“雀斑分布”（够多、鼻梁两侧都有）。"""
    if len(found) < MARK_MIN_PATTERN:
        return False
    f = next((q for q in faces if q.points is not None), None)
    mid = float(f.points[1][0]) if f is not None else float(np.median([b.center[0] for b in found]))
    left = sum(b.center[0] < mid for b in found)
    right = len(found) - left
    return min(left, right) >= max(2, 0.3 * max(left, right))


def remove_blemishes(image, masks, faces, scale, sensitivity=1.0, freckles=0.0, moles=False, trace=None, marks=0.0):
    """返回 (修复后图像, [Blemish, ...])。sensitivity 越大检出越多，0 关闭。

    trace 传入 list 时记录每个被排除的候选及原因，用于调参。
    """
    if sensitivity <= 0:
        return image, []
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L = cv2.GaussianBlur(lab[..., 0].astype(np.float32), (0, 0), max(1.0, scale / 300))
    A = cv2.GaussianBlur(lab[..., 1].astype(np.float32), (0, 0), max(1.0, scale / 300))
    k = max(7, int(scale / 14) | 1)  # 形态学核要明显大于斑点，才能完整取出斑点的对比
    max_d = scale / 16                 # 斑点直径上限，更大的多是皱纹、褶皱阴影

    region = masks.face_hard.copy()  # 只在脸上找，脖子下颌的阴影不算
    if masks.method != "parsing" and not any(f.points is not None for f in faces):
        for x, y, w, h in (f.box for f in faces):
            for rx0, ry0, rx1, ry1 in _FACE_EXCLUDE:
                region[int(y + ry0 * h):int(y + ry1 * h), int(x + rx0 * w):int(x + rx1 * w)] = 0
    # 眼周（鱼尾纹、眼袋）交给 Dodge & Burn，不当作瑕疵
    for f in faces:
        if f.points is not None:
            region[f.part_mask(image.shape, ["left_eye", "right_eye"], grow=f.box[2] * 0.09) > 0] = 0
            region[f.under_eye_mask(image.shape) > 0.1] = 0
            # 眉间（皱眉纹）
            glabella = cv2.convexHull(f.points[[55, 65, 285, 295, 168, 6]].astype(np.int32))
            g = np.zeros(region.shape, np.uint8)
            cv2.fillConvexPoly(g, glabella, 255)
            region[grow(g, f.box[2] * 0.05) > 0] = 0
            # 嘴角阴影
            region[f.part_mask(image.shape, ["lips"], grow=f.box[2] * 0.06) > 0] = 0
            # 鼻子侧面的轮廓阴影
            nose = f.part_mask(image.shape, ["nose"])
            band = grow(nose, f.box[2] * 0.06) & ~shrink(nose, f.box[2] * 0.06)
            region[band > 0] = 0
    # 离五官、头发、轮廓边缘至少一个核的距离
    inner = shrink(region, k)
    if not np.any(inner):
        return image, []

    dark = _hat(L, cv2.MORPH_BLACKHAT, k, scale)
    red = _hat(A, cv2.MORPH_TOPHAT, k, scale)
    # 阈值 = 固定下限 与 皮肤内噪声水平 取大，再除以灵敏度
    t_dark = max(6.0, 4 * float(np.median(dark[inner > 0]))) / sensitivity
    t_red = max(3.5, 4 * float(np.median(red[inner > 0]))) / sensitivity
    cand = (((dark > t_dark) | (red > t_red)) & (inner > 0)).astype(np.uint8) * 255

    min_area = max(4.0, (scale / 150) ** 2)
    max_area = np.pi * (max_d / 2) ** 2
    img = image.astype(np.float32)
    high = img - cv2.GaussianBlur(img, (0, 0), max(1.0, scale / 250))
    # 去掉大尺度明暗（侧光、脸颊转折）后的亮度细节，用来衡量斑点周围皮肤的起伏
    L_detail = L - cv2.GaussianBlur(L, (0, 0), scale / 25)
    found, todo = [], []
    weights = {"acne": 1.0, "spot": 0.7, "freckle": float(freckles), "mole": 1.0 if moles else 0.0, "mark": float(marks)}
    H, W = L.shape
    for bx, by, bw, bh, mask, center, area in _candidates(dark, red, inner, t_dark, t_red,
                                                          min_area, max_area, trace):
        d = 2 * np.sqrt(area / np.pi)
        # 只在斑点周围的局部窗口里计算
        pad = int(d * 2.5) + 10
        y0, y1, x0, x1 = max(0, by - pad), min(H, by + bh + pad), max(0, bx - pad), min(W, bx + bw + pad)
        blob = np.zeros((y1 - y0, x1 - x0), np.uint8)
        blob[by - y0:by - y0 + bh, bx - x0:bx - x0 + bw] = mask
        # 周围环带：离开斑点自身的渐变边缘再取
        ring = ((grow(blob, d * 2.2 + 6) > 0) & (grow(blob, d * 0.9 + 5) == 0)
                & (region[y0:y1, x0:x1] > 0))
        if ring.sum() < 8:
            continue
        inside = blob > 0
        Lr, Ar = L[y0:y1, x0:x1], A[y0:y1, x0:x1]
        ring_d = L_detail[y0:y1, x0:x1][ring]
        ring_std = 1.4826 * float(np.median(np.abs(ring_d - np.median(ring_d))))  # 稳健的起伏估计
        # 用峰值对比（斑点最深/最红处 vs 周围），不受阈值切出的斑点大小影响
        dL = float(np.median(Lr[ring]) - np.percentile(Lr[inside], 10))
        da = float(np.percentile(Ar[inside], 90) - np.median(Ar[ring]))
        kind = _classify(d, dL, da, scale)
        # 孤立度 1：斑点要明显突出于周围的起伏，否则是褶皱、酒窝、微笑纹这类结构
        if kind is None or max(dL, da * 2) < 3.0 * ring_std:
            if trace is not None:
                trace.append((center, f"对比不足 dL={dL:.1f} da={da:.1f} ring_std={ring_std:.1f}"))
            continue
        # 孤立度 2：周围也在发暗/发红（处于一片阴影或褶皱里，只是其中较深的一点）则不算
        resp = red if kind == "acne" else dark
        r_roi = resp[y0:y1, x0:x1]
        spread = float(np.median(r_roi[ring])) / (float(np.percentile(r_roi[inside], 90)) + 1e-6)
        if spread > 0.3:
            if trace is not None:
                trace.append((center, f"处于阴影/褶皱中 spread={spread:.2f}"))
            continue
        found.append(Blemish(kind, center, d / 2, dL, da, ring_std, weights[kind]))
        todo.append((y0, y1, x0, x1, blob, d))
    # 痘印：小而浅、但不成雀斑分布（或偏红）的，从“雀斑”改判为痘印
    if marks > 0:
        fr = [b for b in found if b.kind == "freckle"]
        pattern = _freckle_pattern(fr, faces)
        for i, b in enumerate(found):
            if b.kind == "freckle" and (not pattern or b.da >= MARK_RED):
                found[i] = Blemish("mark", b.center, b.radius, b.dL, b.da, b.ring_std, weights["mark"])
    # 统一修复（顺序与检出顺序相同：不开 marks 时与逐个修复的旧行为逐像素一致）
    for b, (y0, y1, x0, x1, blob, d) in zip(found, todo):
        if b.weight > 0:
            full = np.zeros((H, W), np.uint8)
            full[y0:y1, x0:x1] = blob
            _heal(img, high, region, full, cand, d, b.weight)
    return np.clip(img + 0.5, 0, 255).astype(np.uint8), found
