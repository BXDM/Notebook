"""建筑校正：识别主体、取主体的关键竖直边，估计相机的歪斜（roll）和仰俯（pitch），用相机旋转一次纠正。

1. 主体：建筑、窗框是人造物，竖直长直线密集；树、草、天空几乎没有。接近竖直的长线段按长度画成密度图，
   最密的几块（质量 >= 最大块 30%）合并成主体框。
2. 关键边：在主体框内用全分辨率 LSD 取接近竖直（±10°）的长线段。
3. 竖直消失点：关键边延长后交于一点（平拍时在无穷远），按长度加权最小二乘求解，再剔除离群边重算。
4. 相机旋转：d = K⁻¹·消失点 是现实中“竖直”方向在相机坐标里的方向；绕光轴转 α 消掉左右歪（全额），
   再绕水平轴转 β 消掉仰拍的上窄下宽（按 keystone 比例，默认 80%：完全拉直后楼顶显大、不自然）。
   单应 H = K·R·K⁻¹，最后裁出原画幅比例的最大内接矩形。
焦距像素数取自 EXIF（焦平面分辨率 × 焦距），没有就按 35mm 等效焦距或默认 1.2 倍长边。
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Straighten:
    ok: bool
    reason: str
    roll: float = 0.0       # 度，正 = 画面里竖线上端偏右（相机顺时针歪）
    pitch: float = 0.0      # 度，正 = 仰拍
    edges: np.ndarray = None   # N×4 关键边（原图坐标）
    box: tuple = None       # 主体框 (x0, y0, x1, y1)
    rms: float = 0.0        # 关键边相对消失点的残差（度）
    H: np.ndarray = None    # 原图 -> 校正后（已含裁剪平移）
    size: tuple = None      # 输出 (宽, 高)


def _lsd(gray):
    lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    if lines is None:
        return np.zeros((0, 4), np.float32)
    return lines.reshape(-1, 4)


def _lean(l):
    """线段相对竖直的倾斜（度，正 = 上端偏右）。"""
    top = l[:, 1] < l[:, 3]
    xt, yt = np.where(top, l[:, 0], l[:, 2]), np.where(top, l[:, 1], l[:, 3])
    xb, yb = np.where(top, l[:, 2], l[:, 0]), np.where(top, l[:, 3], l[:, 1])
    return np.degrees(np.arctan2(xt - xb, yb - yt))


def subject_box(image, work_side=3000):
    """竖直长线密度最高的区域（原图坐标 x0, y0, x1, y1），找不到返回 None。"""
    h, w = image.shape[:2]
    s = min(1.0, work_side / max(h, w))
    g = cv2.cvtColor(cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    l = _lsd(g)
    L = np.hypot(l[:, 2] - l[:, 0], l[:, 3] - l[:, 1])
    vert = (np.abs(_lean(l)) < 15) & (L > max(H, W) * 0.015)
    if vert.sum() < 4:
        return None
    dens = np.zeros((H, W), np.float32)
    for (x1, y1, x2, y2), ll in zip(l[vert], L[vert]):
        cv2.line(dens, (int(x1), int(y1)), (int(x2), int(y2)), float(ll), 3)
    dens = cv2.GaussianBlur(dens, (0, 0), max(H, W) / 40)
    n, lab, st, _ = cv2.connectedComponentsWithStats((dens > 0.2 * dens.max()).astype(np.uint8))
    mass = np.array([dens[lab == i].sum() for i in range(1, n)])
    keep = np.where(mass >= 0.3 * mass.max())[0] + 1
    x0, y0 = st[keep, 0].min(), st[keep, 1].min()
    x1, y1 = (st[keep, 0] + st[keep, 2]).max(), (st[keep, 1] + st[keep, 3]).max()
    return int(x0 / s), int(y0 / s), int(x1 / s), int(y1 / s)


def key_edges(image, box, max_lean=10.0):
    """主体框内的关键竖直边（全分辨率，原图坐标）。"""
    x0, y0, x1, y1 = box
    g = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    l = _lsd(g)
    L = np.hypot(l[:, 2] - l[:, 0], l[:, 3] - l[:, 1])
    k = (np.abs(_lean(l)) < max_lean) & (L > max(image.shape[:2]) * 0.02)
    return l[k] + np.float32([x0, y0, x0, y0])


def vertical_vp(edges, K):
    """竖直消失点在相机坐标里的方向 d（单位向量，指向上方），以及各边残差（度）。"""
    Ki = np.linalg.inv(K)
    p1 = (Ki @ np.c_[edges[:, :2], np.ones(len(edges))].T).T
    p2 = (Ki @ np.c_[edges[:, 2:], np.ones(len(edges))].T).T
    lines = np.cross(p1, p2)
    lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    length = np.hypot(edges[:, 2] - edges[:, 0], edges[:, 3] - edges[:, 1])
    w = length.copy()
    for _ in range(3):  # 加权最小二乘，再按残差剔除离群边
        M = (lines * w[:, None]).T @ lines
        d = np.linalg.eigh(M)[1][:, 0]
        if d[1] > 0:
            d = -d
        # 残差：线段方向与“线段中点指向消失点”方向的夹角
        mid = (p1 + p2) / 2
        to_vp = d[None, :2] - mid[:, :2] * d[2]
        seg = p2[:, :2] - p1[:, :2]
        cos = np.abs(np.sum(to_vp * seg, axis=1)) / (np.linalg.norm(to_vp, axis=1) * np.linalg.norm(seg, axis=1) + 1e-12)
        res = np.degrees(np.arccos(np.clip(cos, 0, 1)))
        w = length * (res < max(1.5, 3 * np.median(res)))
    return d / np.linalg.norm(d), res, w > 0


def camera_matrix(shape, focal_px=None, focal35=None):
    h, w = shape[:2]
    if focal_px:
        f = focal_px
    elif focal35:
        f = focal35 * np.hypot(w, h) / np.hypot(36, 24)  # 等效焦距按对角线换算
    else:
        f = 1.2 * max(w, h)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)


def _inner_rect(H, w, h):
    """单应变形后原画幅比例的最大内接矩形 (x, y, 宽, 高)，以变形后四边形的中心为中心。"""
    corners = cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]])[None], H)[0]
    c = corners.mean(axis=0)
    lo, hi = 0.0, 1.5
    for _ in range(30):
        s = (lo + hi) / 2
        rw, rh = w * s / 2, h * s / 2
        pts = [(c[0] - rw, c[1] - rh), (c[0] + rw, c[1] - rh), (c[0] + rw, c[1] + rh), (c[0] - rw, c[1] + rh)]
        if all(cv2.pointPolygonTest(corners.reshape(-1, 1, 2), p, False) >= 0 for p in pts):
            lo = s
        else:
            hi = s
    rw, rh = w * lo, h * lo
    return c[0] - rw / 2, c[1] - rh / 2, rw, rh


def estimate(image, exif=None, keystone=0.8, min_edges=6, max_rms=1.0):
    """估计校正；不可信（关键边太少、彼此不一致、角度过大）时 ok=False，不改图。"""
    exif = exif or {}
    box = subject_box(image)
    if box is None:
        return Straighten(False, "没有找到竖直结构（不是建筑照片）")
    edges = key_edges(image, box)
    if len(edges) < min_edges:
        return Straighten(False, f"主体关键边只有 {len(edges)} 条", box=box, edges=edges)
    K = camera_matrix(image.shape, exif.get("focal_px"), exif.get("focal35"))
    d, res, inl = vertical_vp(edges, K)
    length = np.hypot(edges[:, 2] - edges[:, 0], edges[:, 3] - edges[:, 1])
    rms = float(np.sqrt(np.average(res[inl] ** 2, weights=length[inl])))
    # 绕光轴：把 d 的画面内分量转到正上方；绕水平轴：把 d 转进画面平面（消掉上窄下宽）
    alpha = np.arctan2(-d[0], -d[1])
    r = np.hypot(d[0], d[1])
    beta = np.arctan2(d[2], r)
    roll, pitch = float(-np.degrees(alpha)), float(np.degrees(beta))  # 竖线上端偏右时 alpha 为负
    st = Straighten(True, "", roll=roll, pitch=pitch, edges=edges[inl], box=box, rms=rms)
    if inl.sum() < min_edges:
        st.ok, st.reason = False, f"一致的关键边只有 {inl.sum()} 条"
    elif rms > max_rms:
        st.ok, st.reason = False, f"关键边彼此不一致（残差 {rms:.2f}°）"
    elif abs(roll) > 8 or abs(pitch) > 25:
        st.ok, st.reason = False, f"角度过大（歪 {roll:.1f}°，仰俯 {pitch:.1f}°），可能是误判"
    if not st.ok:
        return st
    ca, sa = np.cos(alpha), np.sin(alpha)
    Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    b = beta * keystone
    cb, sb = np.cos(b), np.sin(b)
    Rx = np.array([[1, 0, 0], [0, cb, -sb], [0, sb, cb]])
    h, w = image.shape[:2]
    H = K @ Rx @ Rz @ np.linalg.inv(K)
    x, y, rw, rh = _inner_rect(H, w, h)
    T = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]])
    st.H, st.size = T @ H, (int(rw), int(rh))
    return st


def apply(image, st):
    if not st.ok:
        return image
    return cv2.warpPerspective(image, st.H, st.size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def overlay(image, st):
    """调试图：主体框（黄）、关键边（红）。"""
    vis = image.copy()
    t = max(2, max(image.shape[:2]) // 800)
    if st.box:
        x0, y0, x1, y1 = st.box
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2 * t)
    if st.edges is not None:
        for x1, y1, x2, y2 in st.edges.astype(int):
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2 * t)
    return vis
