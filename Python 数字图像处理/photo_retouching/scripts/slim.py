"""自适应瘦脸：只瘦确实显大、瘦了能更好看的脸；本来就瘦、脸太小、侧得太厉害的不动（意义不大、风险高）。

判断（用三维关键点，转头不影响）：
- 丰满度 = 脸宽（两侧脸缘 234、454 的三维距离）/ 两眼外角距离。卡上 264 张可用人脸（2026-11-18 扫描）：
  中位 1.56、90% 分位 1.66、97% 分位 1.72；从 1.66 开始瘦，1.76 满额，所以绝大多数脸不动。
  （脸宽 / 脸高试过，排在最前的全是被误检成人脸的毛绒玩具，低头抬头也让它波动很大，不用。）
- 跳过：没有关键点；脸宽 < MIN_FACE_PX（小脸瘦了看不出）；|偏航| > MAX_YAW（侧脸只看得到一侧轮廓，推错风险大）；
  不像真脸（脸颊颜色不像皮肤、眼睛不比脸颊暗：玩具、花、手臂被误检成人脸）；表情夸张（嘟嘴、闭眼时眼角定位不准，嘴宽 / 眼距 > 0.8；
  歪嘴笑，两嘴角高差 / 眼距 > 0.12）——这些情况比值偏大不是因为脸大。
形变：沿下颌轮廓（颧骨下 → 下颌角 → 下巴两侧）向脸中线局部液化，位移上限 MAX_SHIFT × 脸宽。
- 左右分开：转头时靠近镜头的一侧轮廓在画面里更宽，按两侧可见半宽分配位移。
- 背景保护：形变区域里的长直线（门框、墙线、栏杆）附近位移衰减到 0，直线不会被推弯。
- 只在脸周围的局部区域重映射。返回 (结果, 每张脸的说明)。
"""
import cv2
import numpy as np

MIN_FACE_PX = 250          # 脸宽（像素）小于这个不瘦：成片上看不出差别
MAX_YAW = 25.0             # 偏航超过这个角度不瘦（30° 左右时三维测量已不稳，R10 的 5246 偏航 −31° 时被判成显大）
MAX_SHIFT = 0.04           # 单侧最大位移 / 脸宽
WE_RANGE = (1.66, 1.76)    # 丰满度（脸宽 / 两眼外角距离）：(起点, 满额)，起点以下一律不瘦

_JAW_L = [132, 58, 172, 136, 150]      # 画面左侧（人物的右脸）下颌轮廓，自上而下
_JAW_R = [361, 288, 397, 365, 379]


def shape_of(face):
    """三维脸型度量；没有关键点或深度时返回 None。"""
    if face.points is None or face.depth is None:
        return None
    p = np.c_[face.points, face.depth]
    d = lambda i, j: float(np.linalg.norm(p[i] - p[j]))
    v = p[454] - p[234]
    W = d(234, 454)
    eye = d(33, 263)
    return {"W": W, "W2d": float(np.linalg.norm(face.points[454] - face.points[234])),
            "we": W / eye, "mouth": d(61, 291) / eye,
            "tilt": abs(float(face.points[61][1] - face.points[291][1])) / eye,
            "yaw": float(np.degrees(np.arctan2(v[2], v[0])))}


def _patch_lab(lab, pt, r):
    x, y = int(pt[0]), int(pt[1])
    h, w = lab.shape[:2]
    p = lab[max(y - r, 0):min(y + r + 1, h), max(x - r, 0):min(x + r + 1, w)].reshape(-1, 3)
    return np.median(p, axis=0) if len(p) else None


def _face_like(image, face):
    """排除误检（毛绒玩具、花、手臂手肘）：脸颊颜色像皮肤（Lab 色相 15~75°、彩度 ≥ 8），
    且两眼虹膜中心（468、473）比脸颊暗 ≥ 10 L*。返回 (是否像真脸, 原因)。"""
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
    a, b = a - 128, b - 128
    hue, chroma = np.degrees(np.arctan2(b, a)), np.hypot(a, b)
    if not (15 <= hue <= 75 and chroma >= 8):
        return False, "颜色不像人脸，可能是误检"
    eyes = [c for c in (_patch_lab(lab, pts[i], r) for i in (468, 473)) if c is not None] if len(pts) > 473 else []
    if not eyes or (L - np.median([e[0] for e in eyes])) / 2.55 < 10:
        return False, "看不出眼睛（误检或闭眼），不瘦"
    return True, ""


def need(face, image=None, strength=1.0):
    """返回 (瘦脸量 0~1, 说明)。"""
    s = shape_of(face)
    if s is None:
        return 0.0, "没有三维关键点"
    if s["W2d"] < MIN_FACE_PX:
        return 0.0, f"脸宽 {s['W2d']:.0f}px，太小"
    if abs(s["yaw"]) > MAX_YAW:
        return 0.0, f"侧脸 {s['yaw']:+.0f}°"
    tag = f"丰满度 {s['we']:.2f}"
    if s["we"] < WE_RANGE[0]:
        return 0.0, f"不显大（{tag}）"
    if image is not None:
        ok, why = _face_like(image, face)
        if not ok:
            return 0.0, why
    if s["mouth"] > 0.8 or s["tilt"] > 0.12:
        return 0.0, f"表情夸张（{tag} 不可信）"
    amt = float(np.clip((s["we"] - WE_RANGE[0]) / (WE_RANGE[1] - WE_RANGE[0]), 0, 1)) * strength
    return amt, f"瘦 {amt:.0%}（{tag}，偏航 {s['yaw']:+.0f}°）"


def _line_guard(gray, radius):
    """长直线附近为 0、远处为 1 的权重（形变不推弯门框墙线）。"""
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(gray)[0]
    guard = np.ones(gray.shape, np.float32)
    if lines is None:
        return guard
    mask = np.zeros(gray.shape, np.uint8)
    for x0, y0, x1, y1 in lines.reshape(-1, 4):
        if np.hypot(x1 - x0, y1 - y0) >= radius:
            cv2.line(mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, 3)
    if mask.any():
        dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
        t = np.clip(dist / (radius * 0.5), 0, 1)
        guard = (t * t * (3 - 2 * t)).astype(np.float32)
    return guard


def slim(image, faces, strength=1.0):
    out = image
    notes = []
    H, W = image.shape[:2]
    for f in faces:
        amt, why = need(f, image, strength)
        notes.append(why)
        if amt <= 0:
            continue
        pts = f.points
        fw = float(np.linalg.norm(pts[454] - pts[234]))
        radius = fw * 0.22
        # 局部区域：脸周围 0.6 个脸宽
        x0, y0 = (pts.min(axis=0) - fw * 0.6).astype(int)
        x1, y1 = (pts.max(axis=0) + fw * 0.6).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        dx = np.zeros((y1 - y0, x1 - x0), np.float32)
        dy = np.zeros_like(dx)
        xs, ys = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
        # 两侧可见半宽（鼻尖到脸缘）分配位移
        nose = pts[4]
        half_l = float(np.linalg.norm(pts[234] - nose))
        half_r = float(np.linalg.norm(pts[454] - nose))
        for idx, half in ((_JAW_L, half_l), (_JAW_R, half_r)):
            shift = fw * MAX_SHIFT * amt * (2 * half / (half_l + half_r)) / 2   # 相邻推点半径重叠，合计约 2 倍
            for i in idx:
                p = pts[i]
                v = nose - p
                v = v / (np.linalg.norm(v) + 1e-6) * shift
                t = np.clip(((xs - p[0]) ** 2 + (ys - p[1]) ** 2) / radius ** 2, 0, 1)
                wgt = (1 - t) ** 2
                dx -= wgt * v[0]
                dy -= wgt * v[1]
        gray = cv2.cvtColor(out[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        guard = _line_guard(gray, radius)
        # 脸部轮廓内部（皮肤上不会有要保护的直线）不衰减
        hull = cv2.convexHull((pts - [x0, y0]).astype(np.int32))
        inside = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(inside, hull, 1)
        guard = np.maximum(guard, cv2.GaussianBlur(inside.astype(np.float32), (0, 0), radius * 0.2))
        mx, my = xs + dx * guard, ys + dy * guard
        patch = cv2.remap(out, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        out = out.copy()
        out[y0:y1, x0:x1] = patch
    return out, notes
