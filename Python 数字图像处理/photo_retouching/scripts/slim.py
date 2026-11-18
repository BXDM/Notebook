"""瘦脸（可选模块，默认关闭）：只给确实显宽、瘦了能更上镜的脸做轻度轮廓修饰，宁可不瘦也不瘦错。

一、判断（face_shape_metrics，只描述脸型比例，不是 BMI、不是脂肪厚度）
- 宽度比 = 两侧脸缘（234、454）距离 / 两眼外角距离，用 MediaPipe 的相对三维坐标算（z 是模型推断的相对深度，
  只用来抵消一点转头，不当作真实脸型）。卡上 264 张人脸：中位 1.56、90% 分位 1.66；1.66 附近胖瘦混杂
  （单张测量误差约 ±0.04），1.73 以上基本是真正偏宽的脸：从 1.70 开始，1.80 满额。
- 跳过：脸宽 < MIN_FACE_PX；头部姿态（人脸变换矩阵）偏航 > 25° 或俯仰 > 20°；不像真脸（肤色、眼睛比脸颊暗：
  排除毛绒玩具、花、手臂误检）；表情系数（blendshapes）鼓腮、嘟嘴、张嘴、歪嘴笑（这时的脸型不作数）；
  儿童（年龄模型 ≤ 12 岁）。笑的时候脸颊变宽，量减半。
二、形变（MLS 移动最小二乘，相似变换）
- 控制点：脸颊（93/132、323/361）每侧最多内收 CHEEK × 脸宽，下颌（58/172/136/150、288/397/365/379）最多 JAW × 脸宽，
  方向指向脸部中线；眼、眉、鼻、嘴、下巴、太阳穴（耳朵高度 127/234、356/454）、额头和区域边框是固定锚点
  ——五官、下巴长度、耳朵不动，形变只发生在脸颊和下颌附近。
- 左右按可见半宽分配（转头时靠镜头一侧轮廓更宽）；区域内的长直线（门框、墙线）附近位移衰减到 0。
- 收窄脸部轮廓时，轮廓外的背景被相应拉进来一点；位移只有脸宽的 1~1.5%，不会出现空洞。
三、检查形变场：五官处位移超过 0.3% 脸宽、下颌没按预期移动、或位移越界时回退原图。
返回 (结果, 每张脸的说明)。
"""
import os

import cv2
import numpy as np

MIN_FACE_PX = 250                    # 脸宽（像素）小于这个不瘦：成片上看不出差别
MAX_YAW, MAX_PITCH = 25.0, 20.0      # 偏航 30° 左右时三维比例已不稳（R10 的 5246 偏航 −31° 时被判成显宽）
WE_RANGE = (1.70, 1.80)              # 宽度比：(起点, 满额)，起点以下一律不瘦
CHEEK, JAW = 0.010, 0.015            # 每侧最大内收 / 脸宽

_CHEEK = [(93, 323), (132, 361)]
_JAW = [(58, 288), (172, 397), (136, 365), (150, 379)]
_JAW_TAPER = [1.0, 1.0, 0.8, 0.5]    # 越靠近下巴收得越少，下巴本身不动
_ANCHORS = [127, 234, 356, 454,                      # 太阳穴 / 耳朵高度
            33, 133, 263, 362, 159, 386, 70, 300,    # 眼、眉
            1, 4, 98, 327, 168,                      # 鼻
            61, 291, 0, 17,                          # 嘴
            149, 176, 148, 152, 377, 400, 378, 10]   # 下巴、额头

AGE_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "age_googlenet.onnx")
_age_sess = None


def face_shape_metrics(face):
    """脸型比例与姿态；没有关键点时返回 None。"""
    if face.points is None or face.depth is None:
        return None
    p = np.c_[face.points, face.depth]
    d = lambda i, j: float(np.linalg.norm(p[i] - p[j]))
    yaw, pitch, _ = face.pose if face.pose else (0.0, 0.0, 0.0)
    return {"width_ratio": d(234, 454) / d(33, 263),
            "W2d": float(np.linalg.norm(face.points[454] - face.points[234])), "yaw": yaw, "pitch": pitch}


def _patch_lab(lab, pt, r):
    x, y = int(pt[0]), int(pt[1])
    h, w = lab.shape[:2]
    p = lab[max(y - r, 0):min(y + r + 1, h), max(x - r, 0):min(x + r + 1, w)].reshape(-1, 3)
    return np.median(p, axis=0) if len(p) else None


def _face_like(image, face):
    """排除误检：脸颊颜色像皮肤（Lab 色相 15~75°、彩度 ≥ 8），且虹膜（468、473）比脸颊暗 ≥ 10 L*。返回 (是否, 原因)。"""
    x0, y0 = face.points.min(axis=0).astype(int)
    x1, y1 = face.points.max(axis=0).astype(int) + 1
    x0, y0 = max(x0, 0), max(y0, 0)
    lab = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2Lab).astype(np.float32)
    pts = face.points - [x0, y0]
    r = max(2, int((x1 - x0) / 80))
    cheeks = [c for c in (_patch_lab(lab, pts[i], r) for i in (50, 280, 205, 425)) if c is not None]
    if not cheeks:
        return False, "看不到脸颊"
    L, a, b = np.median(cheeks, axis=0)
    hue, chroma = np.degrees(np.arctan2(b - 128, a - 128)), np.hypot(a - 128, b - 128)
    if not (15 <= hue <= 75 and chroma >= 8):
        return False, "颜色不像人脸，可能是误检"
    eyes = [c for c in (_patch_lab(lab, pts[i], r) for i in (468, 473)) if c is not None] if len(pts) > 473 else []
    if not eyes or (L - np.median([e[0] for e in eyes])) / 2.55 < 10:
        return False, "看不出眼睛（误检或闭眼）"
    return True, ""


def is_child(image, face):
    """年龄模型（ONNX 模型库 age_googlenet，Adience 8 个年龄段，Apache-2.0）判为 0~12 岁返回 True；模型缺失返回 None。
    人脸网格会把孩子的脸往成人比例上拟合，几何特征分不出孩子（实测），所以用模型。"""
    global _age_sess
    if _age_sess is None:
        if not os.path.exists(AGE_MODEL):
            return None
        import onnxruntime as ort
        _age_sess = ort.InferenceSession(AGE_MODEL, providers=["CPUExecutionProvider"])
    x0, y0 = face.points.min(axis=0)
    x1, y1 = face.points.max(axis=0)
    w, h = x1 - x0, y1 - y0
    crop = image[max(int(y0 - 0.25 * h), 0):int(y1 + 0.125 * h), max(int(x0 - 0.25 * w), 0):int(x1 + 0.25 * w)]
    if crop.size == 0:
        return None
    blob = (cv2.resize(crop, (224, 224)).astype(np.float32) - np.float32([104, 117, 123])).transpose(2, 0, 1)[None]
    prob = _age_sess.run(None, {_age_sess.get_inputs()[0].name: blob})[0][0]
    return int(np.argmax(prob)) <= 2


def need(face, image=None, strength=1.0):
    """返回 (瘦脸量 0~1, 说明)。"""
    s = face_shape_metrics(face)
    if s is None:
        return 0.0, "没有三维关键点"
    if s["W2d"] < MIN_FACE_PX:
        return 0.0, f"脸宽 {s['W2d']:.0f}px，太小"
    if abs(s["yaw"]) > MAX_YAW or abs(s["pitch"]) > MAX_PITCH:
        return 0.0, f"姿态（偏航 {s['yaw']:+.0f}°、俯仰 {s['pitch']:+.0f}°）不瘦"
    tag = f"宽度比 {s['width_ratio']:.2f}"
    if s["width_ratio"] < WE_RANGE[0]:
        return 0.0, f"不显宽（{tag}）"
    if image is not None:
        ok, why = _face_like(image, face)
        if not ok:
            return 0.0, why
    b = face.blend or {}
    roll = np.radians(face.pose[2]) if face.pose else 0.0
    v = face.points[291] - face.points[61]
    tilt = abs(float(-np.sin(roll) * v[0] + np.cos(roll) * v[1])) / float(np.linalg.norm(face.points[263] - face.points[33]))
    if tilt > 0.12 or (b.get("cheekPuff", 0) > 0.3 or b.get("mouthPucker", 0) > 0.6 or b.get("jawOpen", 0) > 0.4
            or abs(b.get("mouthSmileLeft", 0) - b.get("mouthSmileRight", 0)) > 0.3):
        return 0.0, f"鼓腮/嘟嘴/张嘴/歪嘴笑（{tag} 不作数）"
    if image is not None and is_child(image, face):
        return 0.0, "儿童，不瘦"
    amt = float(np.clip((s["width_ratio"] - WE_RANGE[0]) / (WE_RANGE[1] - WE_RANGE[0]), 0, 1)) * strength
    smile = (b.get("mouthSmileLeft", 0) + b.get("mouthSmileRight", 0)) / 2
    if smile > 0.5:
        amt *= 0.5
        tag += "，在笑减半"
    return amt, f"瘦 {amt:.0%}（{tag}，偏航 {s['yaw']:+.0f}°）"


def _mls_similarity(grid, src, dst):
    """MLS 相似变换（Schaefer 2006）：grid (N,2) 上每点在控制点 src→dst 下的映射。"""
    d2 = ((grid[:, None, :] - src[None]) ** 2).sum(-1)
    w = 1.0 / np.maximum(d2, 1e-6)
    ws = w.sum(1, keepdims=True)
    ps = (w @ src) / ws
    qs = (w @ dst) / ws
    ph = src[None] - ps[:, None]
    qh = dst[None] - qs[:, None]
    mu = (w * (ph ** 2).sum(-1)).sum(1)
    a = (w * (ph * qh).sum(-1)).sum(1) / mu
    b = (w * (ph[..., 0] * qh[..., 1] - ph[..., 1] * qh[..., 0])).sum(1) / mu
    v = grid - ps
    return np.stack([a * v[:, 0] - b * v[:, 1], b * v[:, 0] + a * v[:, 1]], 1) + qs


def _line_guard(gray, radius):
    """长直线附近为 0、远处为 1 的权重（形变不推弯门框墙线）。"""
    lines = cv2.createLineSegmentDetector().detect(gray)[0]
    guard = np.ones(gray.shape, np.float32)
    if lines is None:
        return guard
    mask = np.zeros(gray.shape, np.uint8)
    for x0, y0, x1, y1 in lines.reshape(-1, 4):
        if np.hypot(x1 - x0, y1 - y0) >= radius:
            cv2.line(mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, 3)
    if mask.any():
        t = np.clip(cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3) / (radius * 0.5), 0, 1)
        guard = (t * t * (3 - 2 * t)).astype(np.float32)
    return guard


def warp_face(image, face, amt):
    """对一张脸做 MLS 瘦脸，返回 (结果, 期望的单侧下颌位移像素)。只改脸周围的局部区域。"""
    pts = face.points
    fw = float(np.linalg.norm(pts[454] - pts[234]))
    H, W = image.shape[:2]
    x0, y0 = (pts.min(axis=0) - fw * 0.35).astype(int)
    x1, y1 = (pts.max(axis=0) + fw * 0.35).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    top, bot = pts[168], pts[152]                    # 脸部中线：鼻根 → 下巴
    axis = (bot - top) / (np.linalg.norm(bot - top) + 1e-6)

    def inward(p):
        foot = top + axis * np.dot(p - top, axis)
        v = foot - p
        return v / (np.linalg.norm(v) + 1e-6)

    half_l = float(np.linalg.norm(pts[234] - pts[4]))
    half_r = float(np.linalg.norm(pts[454] - pts[4]))
    side_w = [2 * half_l / (half_l + half_r), 2 * half_r / (half_l + half_r)]
    src, dst = [], []
    for pairs, frac, tapers in ((_CHEEK, CHEEK, [1.0, 1.0]), (_JAW, JAW, _JAW_TAPER)):
        for (i, j), t in zip(pairs, tapers):
            for k, idx in enumerate((i, j)):
                p = pts[idx]
                src.append(p)
                dst.append(p + inward(p) * fw * frac * amt * t * min(side_w[k], 1.3))
    for idx in _ANCHORS:
        src.append(pts[idx])
        dst.append(pts[idx])
    for t in np.linspace(0, 1, 8):                   # 区域边框固定
        for q in ((x0 + t * (x1 - x0), y0), (x0 + t * (x1 - x0), y1 - 1),
                  (x0, y0 + t * (y1 - y0)), (x1 - 1, y0 + t * (y1 - y0))):
            src.append(np.float32(q))
            dst.append(np.float32(q))
    src, dst = np.float32(src), np.float32(dst)
    # 逆映射（输出像素从哪里取样）= 控制点 dst→src 的 MLS；在 8 像素网格上算再插值
    step = 8
    gx, gy = np.meshgrid(np.arange(x0, x1, step, dtype=np.float32), np.arange(y0, y1, step, dtype=np.float32))
    m = _mls_similarity(np.c_[gx.ravel(), gy.ravel()], dst, src).reshape(gx.shape + (2,))
    dx = cv2.resize(m[..., 0] - gx, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(m[..., 1] - gy, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    guard = _line_guard(gray, fw * 0.2)
    inside = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(inside, cv2.convexHull((pts - [x0, y0]).astype(np.int32)), 1)
    guard = np.maximum(guard, cv2.GaussianBlur(inside.astype(np.float32), (0, 0), fw * 0.04))
    xs, ys = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
    dx, dy = dx * guard, dy * guard
    patch = cv2.remap(image, xs + dx, ys + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out = image.copy()
    out[y0:y1, x0:x1] = patch
    mag = np.hypot(dx, dy)

    def at(idx):
        q = np.clip((pts[idx] - [x0, y0]).astype(int), 0, [x1 - x0 - 1, y1 - y0 - 1])
        return mag[q[:, 1], q[:, 0]]
    stats = {"expect": fw * JAW * amt, "feat": float(at([33, 263, 1, 61, 291, 152, 159, 386]).max()),
             "jaw": float(np.median(at([58, 172, 288, 397]))), "max": float(mag.max()), "fw": fw}
    return out, stats


def _check(stats):
    """形变场检查（不用重新检测关键点：MediaPipe 的轮廓点按人脸先验拟合，跟不上几个像素的形变，
    轮廓一变还会带着五官点一起微调，实测会把正常形变误判成“下颌没收”“五官移动 0.5~1.5%”）：
    五官处位移 ≤ 0.3% 脸宽；下颌处实际位移 ≥ 期望的一半（没被直线保护抵消）；最大位移 ≤ 期望的 2 倍。"""
    fw, e = stats["fw"], stats["expect"]
    if stats["feat"] > 0.003 * fw:
        return False, f"五官处位移 {stats['feat'] / fw:.2%} 脸宽"
    if stats["jaw"] < 0.5 * e:
        return False, f"下颌只移动 {stats['jaw']:.1f}px（期望约 {e:.1f}px，可能被背景直线保护抵消）"
    if stats["max"] > 2 * e:
        return False, f"最大位移 {stats['max']:.1f}px，超过期望 2 倍"
    return True, ""


def slim(image, faces, strength=1.0, check=True):
    out, notes = image, []
    for f in faces:
        amt, why = need(f, image, strength)
        if amt <= 0:
            notes.append(why)
            continue
        fw = float(np.linalg.norm(f.points[454] - f.points[234]))
        warped, stats = warp_face(out, f, amt)
        if check:
            ok, reason = _check(stats)
            if not ok:
                notes.append(f"{why} → 检查不通过，回退（{reason}）")
                continue
        out = warped
        notes.append(why)
    return out, notes
