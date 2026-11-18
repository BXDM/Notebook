"""去双下巴（可选模块，默认关闭、按需调用）：光影优先、几何保守，宁可不做也不做错。

没有找到可直接用的开源实现：公开的只有 StyleGAN 隐空间编辑类方法（如 coarse-to-fine-chin-editing，
数据集仅限非商业研究，且要整张脸重新生成，会改变身份细节），不适合修图流程。这里全部用传统图像处理自己算。

一、分析（在人脸摆正、缩到脸宽约 ANALYSIS_FW 像素的局部图上）
- 下颌线：MediaPipe 下颌轮廓点 172/136/150/149/176/148/152/377/400/378/379/365/397，插值成 y_j(x)；
  每个像素到下颌线的竖直距离 d（以脸宽 fw 为单位，向下为正）。
- 皮肤：BiSeNet 人脸分割的 neck + face 类，去掉衣服、头发；下颌线以下 0~0.40 fw、下颌两端之间是分析区。
- 剖面：中间段每个 d 的亮度中位数 P(d)。由此量出
  · 阴影深度：下颌保护带以下（0.07~0.25 fw）最暗处比下颈部参考亮度暗多少（L*）；
  · 褶痕/第二道轮廓：0.09~0.22 fw 内的横向暗谷（比上下各 0.035 fw 处暗多少；更近是下颌自己的投影边缘，
    更远多是颈纹），看左/中/右三段是否至少两段在相近位置都有（跟着下颌走的弧线，而不是一段局部皱纹），
    以及暗谷上方有没有亮的软组织鼓起。
  · 脖子露出宽度：下颌线下 0.1~0.2 fw 处从中线往两侧连续皮肤的宽度。
二、分类
- 不做（按顺序判断）：脸宽 < MIN_FACE_PX、偏航 > MAX_YAW、张嘴；脸型不显饱满（scripts.slim 的宽度比 < FULL_MIN，
  笑时再加 SMILE_EXTRA；瘦的人下颌下的投影就是下颌轮廓，不该动）；下颌下皮肤露出不足 60% 或脖子露出宽度
  < NECK_MIN 脸宽（衣领、头发、手挡住；手用 MediaPipe 手部关键点排除）；阴影 > SHADOW_MAX（整片背光/被挡，
  补光就成了重新打光）；没有褶痕（暗谷 < CREASE_MIN，或不连贯且 < CREASE_SOLO）——只有阴影没有褶痕的，
  多半就是下颌在脖子上的正常投影（圆脸女生、低头时都有），不动；有褶痕但阴影 < SHADOW_MIN 时暗谷要 ≥ 2×CREASE_MIN。
- 光影型：有褶痕（和阴影），但没有成形的第二道轮廓，或者是低头（俯仰 > HEAD_DOWN，MediaPipe 俯仰正值为低头）
  造成的 → 只做局部光影。
- 轮廓重叠型：第二道轮廓连贯（三段至少两段）、离下颌线 OVERLAP_D 范围内、暗谷 ≥ OVERLAP_CREASE、上方鼓起
  ≥ OVERLAP_BULGE、不是明显低头 → 光影之外再做很小的几何上收。
三、处理
- 光影：只抬低频亮度（σ≈4% fw 的阴影亏量的一部分 + 横向暗谷），保留皮肤纹理；
  下颌线下方 0~2.5% fw 完全不动、到 7% fw 渐入——下颌轮廓本身的明暗是脸型边界，保留。
- 几何：只有竖直位移，下颌线以下的软组织向上收 ≤ GEO_MAX × fw，峰值在第二道轮廓附近，
  下颌线处为 0（下巴不动、不悬空），往下渐弱到 0；离皮肤边界（衣领、脖子两侧、头发）0.05 fw 内渐弱到 0，
  所以脖子宽度、领口都不动。
  检查形变场：下颌线以上位移必须为 0，最大位移不超过设定。
返回 (结果, 每张脸的说明)。
"""
import cv2
import numpy as np

from scripts.face import Face
from scripts.face_parsing import parse_faces
from scripts.hands import hand_mask
from scripts.slim import face_shape_metrics

JAW = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397]
MIN_FACE_PX = 200          # 原图上脸宽小于这个不处理
MAX_YAW = 20.0             # 转头后宽度比偏大、下颌下剖面也歪（富士 0853 偏航 30° 被误判），只做近正脸
ANALYSIS_FW = 320          # 分析图上的脸宽
SHADOW_MIN = 8.0           # 阴影深度（L*）起点
CREASE_MIN = 1.8           # 暗谷（下颌线下 0.09~0.22 fw 的横向褶痕）起点：光有阴影不做，那多半是下颌的投影
CREASE_SOLO = 4.0          # 暗谷不连贯时，要这么深才算
OVERLAP_CREASE = 3.0       # 判为轮廓重叠所需的暗谷深度
OVERLAP_BULGE = 2.0        # 暗谷上方鼓起比暗谷亮多少（L*）
OVERLAP_D = (0.09, 0.18)   # 第二道轮廓离下颌线的距离范围（fw）
HEAD_DOWN = 18.0           # 俯仰（低头为正）超过这个角度：双下巴多半是姿态造成的，只做光影
NECK_MIN = 0.45            # 下颌下 0.1~0.2 fw 处连续皮肤宽度 / fw：更窄说明脖子被衣领、头发挡住大半
FULL_MIN = 1.64            # 脸型饱满度（scripts.slim 的宽度比）起点：瘦脸的人下颌下的阴影是脸型轮廓，不动
GEO_MAX = 0.010            # 几何上收最大量 / 脸宽
PROTECT = (0.025, 0.07)    # 下颌线保护带：(完全不动, 完全生效)，单位 fw
MAX_LIFT = 14.0            # 单点最多提亮（L*）
SHADOW_MAX = 28.0          # 阴影比这还深：整片脖子背光或被手/物体挡住，补光就成了重新打光，不做
SMILE_EXTRA = 0.04         # 笑的时候脸颊横向变宽，饱满度门槛相应提高


def _smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def _head_down(pitch):
    """MediaPipe 人脸变换矩阵的俯仰：正值是低头（卡上 2026-11-08 抬着下巴的照片约 −15°~−22°，
    梦露经典仰拍 −18°；普通平视/相机略高时多在 +5°~+15°）。"""
    return pitch


def _prep(image, face):
    """取下巴到脖子的分析区域：摆正（按翻滚角旋转）、缩放到脸宽 ANALYSIS_FW。
    返回 dict：crop 区域、旋转仿射矩阵（原图 crop 坐标 → 分析图坐标）、分析图、关键点（分析图坐标）等。"""
    pts = face.points
    H, W = image.shape[:2]
    fw = float(np.linalg.norm(pts[454] - pts[234]))
    cx = float((pts[234][0] + pts[454][0]) / 2)
    x0, x1 = int(max(cx - 0.85 * fw, 0)), int(min(cx + 0.85 * fw, W))
    y0, y1 = int(max(pts[10][1] - 0.1 * fw, 0)), int(min(pts[152][1] + 0.6 * fw, H))
    crop = image[y0:y1, x0:x1]
    k = min(1.0, ANALYSIS_FW / fw)
    # 摆正：让 鼻根168→下巴152 竖直向下
    v = pts[152] - pts[168]
    ang = np.degrees(np.arctan2(v[0], v[1]))      # 向右偏为正
    c0 = (float(pts[152][0] - x0), float(pts[152][1] - y0))
    M = cv2.getRotationMatrix2D(c0, -ang, k)      # crop 坐标 → 分析图坐标（绕下巴旋转并缩放）
    M[:, 2] += np.array(c0) * k - np.array(c0)    # 下巴点落在缩放后的原位置
    sw, sh = int((x1 - x0) * k), int((y1 - y0) * k)
    A = np.vstack([M, [0, 0, 1]])
    # 先在 crop 坐标里算缩放+旋转，输出尺寸用 crop 缩放后的大小（摆正角度很小，边角丢一点无所谓）
    small = cv2.warpAffine(crop, M, (sw, sh), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
    p = (np.c_[pts - [x0, y0], np.ones(len(pts))] @ M.T).astype(np.float32)
    return dict(box=(x0, y0, x1, y1), M=A, small=small, pts=p, fw=fw, fw_a=fw * k, k=k)


def _parse(small, pts, face):
    """在分析图上跑人脸分割；返回 (皮肤 0/1, 衣服/头发 0/1)；模型不可用返回 None。"""
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    bw = (x1 - x0) * 1.2
    cx = (x0 + x1) / 2
    box = (int(cx - bw / 2), int(y0 + (y1 - y0) * 0.08), int(bw), int(bw))
    f = Face(box, pts, None, face.pose, face.blend)
    parsed = parse_faces(small, [f])
    if parsed is None:
        return None
    skin = ((parsed["neck"] + parsed["face"]) > 0.5) & (parsed["cloth"] < 0.5) & (parsed["hair"] < 0.5) \
        & (parsed["features"] < 0.4)
    hand, _ = hand_mask(small)                       # 托腮、手挡下巴：人脸分割没有手这一类，手会被当成皮肤
    if hand is not None:
        skin &= cv2.dilate(hand, np.ones((15, 15), np.uint8)) == 0
    return skin.astype(np.float32)


def _jaw_line(pts, w):
    """下颌线 y_j(x)，x 在两端之外不外推（返回有效范围）。"""
    j = pts[JAW]
    order = np.argsort(j[:, 0])
    jx, jy = j[order, 0], j[order, 1]
    xs = np.arange(w, dtype=np.float32)
    yj = np.interp(xs, jx, jy).astype(np.float32)
    return yj, float(jx[0]), float(jx[-1])


def _profile(L, skin, d, sel, fw, d_max=0.40, step=0.01):
    """分析区 sel 内按 d（fw 单位）分箱的亮度中位数；皮肤像素不足的箱为 nan。"""
    bins = np.arange(0, d_max + 1e-6, step)
    P = np.full(len(bins), np.nan, np.float32)
    for i, b in enumerate(bins):
        m = sel & (skin > 0.5) & (d >= b) & (d < b + step)
        if m.sum() >= 8:
            P[i] = np.median(L[m])
    return bins, P


def _valley(bins, P, lo=0.09, hi=0.22, delta=0.035):
    """剖面里 [lo, hi] 内最深的暗谷：返回 (深度 L*, 位置 d, 上方鼓起比谷底亮多少)。"""
    step = bins[1] - bins[0]
    n = int(round(delta / step))
    best = (0.0, None, 0.0)
    Ps = P.copy()
    ok = ~np.isnan(Ps)
    if ok.sum() < 10:
        return best
    Ps = np.interp(np.arange(len(P)), np.flatnonzero(ok), Ps[ok])
    Ps = np.convolve(np.pad(Ps, 1, mode="edge"), np.ones(3) / 3, "valid")
    for i in range(len(bins)):
        if not (lo <= bins[i] <= hi) or i - n < 0 or i + n >= len(bins) or not ok[i]:
            continue
        depth = (Ps[i - n] + Ps[i + n]) / 2 - Ps[i]
        if depth > best[0]:
            above = Ps[max(1, i - 3 * n):i]
            bulge = float(above.max() - Ps[i]) if len(above) else 0.0
            best = (float(depth), float(bins[i]), bulge)
    return best


def _neck_width(skin, d, xc, lo=0.10, hi=0.20):
    """中线处下颌线以下 lo~hi fw 的各行上，从中线往两侧、在下颌线以下的连续皮肤宽度（像素，各行中位数）。"""
    xc = int(round(xc))
    if not 0 <= xc < skin.shape[1]:
        return 0.0
    ws = []
    for r in np.flatnonzero((d[:, xc] >= lo) & (d[:, xc] <= hi)):
        row = (skin[r] > 0.5) & (d[r] > 0)
        if not row[xc]:
            continue
        off = np.flatnonzero(~row)
        left = off[off < xc]
        right = off[off > xc]
        x0 = left.max() + 1 if len(left) else 0
        x1 = right.min() if len(right) else len(row)
        ws.append(x1 - x0)
    return float(np.median(ws)) if ws else 0.0


def analyze(image, face):
    """量测下巴下方。返回 (结果 dict 或 None, 原因)。dict 里有类型 kind：none / light / overlap。"""
    if face.points is None:
        return None, "没有关键点"
    pts = face.points
    fw = float(np.linalg.norm(pts[454] - pts[234]))
    if fw < MIN_FACE_PX:
        return None, f"脸宽 {fw:.0f}px，太小"
    yaw, pitch, _ = face.pose if face.pose else (0.0, 0.0, 0.0)
    if abs(yaw) > MAX_YAW:
        return None, f"侧脸（偏航 {yaw:+.0f}°）"
    b = face.blend or {}
    if b.get("jawOpen", 0) > 0.35:
        return None, "张嘴，下颌位置不作数"
    sm = face_shape_metrics(face)
    if sm is None:
        return None, "没有三维关键点，无法判断脸型"
    b = face.blend or {}
    smile = (b.get("mouthSmileLeft", 0) + b.get("mouthSmileRight", 0)) / 2
    if sm["width_ratio"] < FULL_MIN + (SMILE_EXTRA if smile > 0.5 else 0):
        return None, f"脸型不显饱满（宽度比 {sm['width_ratio']:.2f}），下颌下的阴影是脸型轮廓，不做"
    a = _prep(image, face)
    small, p, fa = a["small"], a["pts"], a["fw_a"]
    h, w = small.shape[:2]
    if p[152][1] + 0.15 * fa > h:
        return None, "下巴下方出画"
    skin = _parse(small, p, face)
    if skin is None:
        return None, "人脸分割模型不可用"
    L = cv2.cvtColor(small, cv2.COLOR_BGR2Lab)[..., 0].astype(np.float32) * (100 / 255)
    yj, jl, jr = _jaw_line(p, w)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    d = (ys - yj[None, :]) / fa
    cxj = float(p[152][0])
    half = max(cxj - jl, jr - cxj, 1.0)
    u = (xs - cxj) / half                              # -1 ~ 1：下颌两端
    lat = 1 - _smooth((np.abs(u) - 0.55) / 0.35)       # 中间 55% 满额，到下颌端点渐出
    lat[np.abs(u) > 0.95] = 0
    roi = (d > 0) & (d <= 0.40) & (lat > 0)
    core = (d > 0.03) & (d <= 0.20) & (np.abs(u) < 0.5)
    cover = float((skin[core] > 0.5).mean()) if core.any() else 0.0
    if cover < 0.6:
        return None, f"下颌下方皮肤只露出 {cover:.0%}（衣领/手/胡子/出画）"
    neck = _neck_width(skin, d, cxj) / fa
    if neck < NECK_MIN:
        return None, f"脖子露出太窄（{neck:.2f} 脸宽，衣领/头发挡住）"
    center = np.abs(u) < 0.5
    bins, P = _profile(L, skin, d, center, fa)
    lower = (d >= 0.24) & (d <= 0.40) & center & (skin > 0.5)
    upper = (d >= -0.12) & (d <= -0.03) & center & (skin > 0.5)
    if lower.sum() < 30 or upper.sum() < 30:
        return None, "脖子露出太少"
    ref_neck = float(np.percentile(L[lower], 60))
    ref_chin = float(np.median(L[upper]))
    band = (bins >= PROTECT[1]) & (bins <= 0.25) & ~np.isnan(P)   # 保护带以下才算：紧贴下颌的投影是轮廓
    shadow = float(max(0.0, min(ref_neck, ref_chin) - np.nanmin(P[band]))) if band.any() else 0.0
    crease, d2, bulge = _valley(bins, P)
    # 左/中/右三段各自找暗谷，看是不是跟着下颌走的连贯弧线
    seg_hits = []
    for lo_u, hi_u in ((-0.75, -0.25), (-0.25, 0.25), (0.25, 0.75)):
        sb, sP = _profile(L, skin, d, (u >= lo_u) & (u < hi_u), fa)
        cdep, cd, _ = _valley(sb, sP)
        seg_hits.append(cd if cdep >= 1.4 else None)
    coherent = d2 is not None and sum(1 for s in seg_hits if s is not None and abs(s - d2) <= 0.05) >= 2
    down = _head_down(pitch)
    info = dict(a, skin=skin, L=L, d=d, u=u, lat=lat, roi=roi, shadow=shadow, crease=crease, d2=d2,
                bulge=bulge, coherent=coherent, ref_neck=ref_neck, ref_chin=ref_chin, pitch=pitch, yaw=yaw,
                segs=seg_hits)
    tag = (f"颈宽 {neck:.2f}fw、阴影 {shadow:.1f}L*、暗谷 {crease:.1f}L*" + (f"@{d2:.2f}fw" if d2 is not None else "")
           + f"、鼓起 {bulge:.1f}、三段连贯 {'是' if coherent else '否'}、俯仰 {pitch:+.0f}°")
    info["tag"] = tag
    if shadow > SHADOW_MAX:
        info["kind"] = "none"
        return info, f"下颌下整片很暗（{tag}），像是背光或被遮挡，不做"
    fold = crease >= CREASE_MIN and (coherent or crease >= CREASE_SOLO)
    if not fold or (shadow < SHADOW_MIN and crease < 2 * CREASE_MIN):
        info["kind"] = "none"
        return info, f"没有明显双下巴（{tag}）"
    # 第二道轮廓要离开下颌线一段（< 0.09 fw 的暗谷是下颌自己的投影边缘），又不能太远（> 0.18 fw 的多是颈纹，
    # 2026-11-08 IMG_5847 的 0.24 fw 暗谷就是颈纹，按它上收会把领口一起推上去）
    if coherent and crease >= OVERLAP_CREASE and bulge >= OVERLAP_BULGE and OVERLAP_D[0] <= d2 <= OVERLAP_D[1] \
            and down <= HEAD_DOWN:
        info["kind"] = "overlap"
        return info, f"轮廓重叠型：光影 + 小量上收（{tag}）"
    info["kind"] = "light"
    why = "低头造成" if down > HEAD_DOWN else ("没有成形的第二道轮廓" if not coherent else "第二道轮廓不明显")
    return info, f"光影型（{why}）：只做局部光影（{tag}）"


def _light_delta(info, strength):
    """分析图上的提亮量 ΔL（L*）。"""
    L, d, skin, lat, fa = info["L"], info["d"], info["skin"], info["lat"], info["fw_a"]
    # 皮肤内的归一化模糊（避免把衣服、背景的亮度混进来）
    m = cv2.GaussianBlur(skin, (0, 0), 0.02 * fa)
    sk = cv2.GaussianBlur(skin, (0, 0), 0.04 * fa)
    Lb = cv2.GaussianBlur(L * skin, (0, 0), 0.04 * fa) / np.maximum(sk, 1e-3)
    # 1) 低频阴影：补上亏量的一部分，目标是下颈部参考亮度与下巴亮度的较小者
    target = min(info["ref_neck"], info["ref_chin"])
    deficit = np.maximum(0, target - Lb)
    # 2) 横向暗谷（褶痕/第二道轮廓）：竖直方向大模糊 − 小模糊
    s1, s2 = 0.008 * fa, 0.03 * fa
    Ls = cv2.GaussianBlur(L * skin, (0, 0), s1) / np.maximum(cv2.GaussianBlur(skin, (0, 0), s1), 1e-3)
    Lv = cv2.GaussianBlur(Ls * skin, (1, 0), sigmaX=0.004 * fa, sigmaY=s2) \
        / np.maximum(cv2.GaussianBlur(skin, (1, 0), sigmaX=0.004 * fa, sigmaY=s2), 1e-3)
    Lv = cv2.GaussianBlur(Lv, (0, 0), sigmaX=0.02 * fa, sigmaY=0.004 * fa)
    valley = np.maximum(0, Lv - cv2.GaussianBlur(Ls, (0, 0), sigmaX=0.02 * fa, sigmaY=0.004 * fa))
    prot = _smooth((d - PROTECT[0]) / (PROTECT[1] - PROTECT[0]))
    fade = 1 - _smooth((d - 0.32) / 0.08)               # 0.32~0.40 fw 渐出
    W = prot * fade * lat * (d > 0)
    W = W * np.clip(m, 0, 1) * skin
    W = cv2.GaussianBlur(W, (0, 0), 0.01 * fa)
    dL = (0.6 * deficit + 0.7 * valley) * W * strength
    return np.minimum(dL, MAX_LIFT), W


def _geo_shift(info, strength):
    """分析图上的竖直上收量 s（像素，≥0：输出像素从下方 s 处取样，即软组织向上收）。"""
    d, lat, fa, d2 = info["d"], info["lat"], info["fw_a"], info["d2"]
    amt = GEO_MAX * min(1.0, info["crease"] / (2 * OVERLAP_CREASE)) * strength
    up = _smooth(d / max(d2, 0.05))                       # 下颌线处 0 → 第二道轮廓处满额
    dn = 1 - _smooth((d - d2) / 0.15)                     # 往下 0.15 fw 渐出
    s = amt * fa * up * dn * lat * (d > 0)
    s = cv2.GaussianBlur(s.astype(np.float32), (0, 0), 0.01 * fa)
    # 只动皮肤内部：离衣服/背景/头发 0.05 fw 以内渐弱到 0，领口、脖子两侧边缘不动（脖子不会变细、领口不会被推）
    inside = cv2.distanceTransform((info["skin"] > 0.5).astype(np.uint8), cv2.DIST_L2, 5) / fa
    return s * _smooth(d / 0.01) * _smooth(inside / 0.05)  # 下颌线以上清零


def _to_full(field, info, shape_crop):
    """分析图上的场 → 原图 crop 坐标（逆仿射）。"""
    Minv = cv2.invertAffineTransform(info["M"][:2])
    return cv2.warpAffine(field, Minv, (shape_crop[1], shape_crop[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def apply(image, info, strength=1.0):
    """按分析结果处理一张脸，返回 (结果, 附加说明)。"""
    x0, y0, x1, y1 = info["box"]
    crop = image[y0:y1, x0:x1]
    dL, _ = _light_delta(info, strength)
    note = f"最大提亮 {dL.max():.1f}L*"
    dLf = _to_full(dL, info, crop.shape)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab).astype(np.float32)
    lab[..., 0] = np.clip(lab[..., 0] + dLf * (255 / 100), 0, 255)
    res = cv2.cvtColor(np.round(lab).astype(np.uint8), cv2.COLOR_Lab2BGR)
    # 只在改动处替换，未改动像素逐位不变
    touched = (dLf > 0.05)[..., None]
    res = np.where(touched, res, crop)
    if info["kind"] == "overlap":
        s = _geo_shift(info, strength)
        # 形变场检查：下颌线以上必须不动；最大位移不超过设定
        above = info["d"] < 0
        if float(np.abs(s[above]).max(initial=0)) > 0.05 or float(s.max()) > GEO_MAX * info["fw_a"] * strength + 0.5:
            note += "，几何检查不通过，只做光影"
        else:
            # s 是分析图像素；分析图坐标系的“竖直”就是摆正后的脸部竖直方向，换回原图 crop 坐标
            k = info["k"]
            M = info["M"][:2]
            dirv = cv2.invertAffineTransform(M)[:, 1] * k   # 分析图 +y 在 crop 坐标里的方向（单位向量；逆变换本身带 1/k）
            sf = _to_full(s, info, crop.shape) / k
            hh, ww = crop.shape[:2]
            gx, gy = np.meshgrid(np.arange(ww, dtype=np.float32), np.arange(hh, dtype=np.float32))
            mx = (gx + sf * dirv[0]).astype(np.float32)
            my = (gy + sf * dirv[1]).astype(np.float32)
            moved = cv2.remap(res, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
            res = np.where((sf > 0.02)[..., None], moved, res)
            note += f"，上收最多 {sf.max():.1f}px（{sf.max() / info['fw']:.2%} 脸宽）"
    out = image.copy()
    out[y0:y1, x0:x1] = res
    return out, note


def reduce_double_chin(image, faces, strength=1.0):
    """去双下巴。返回 (结果, 每张脸的说明)。没有需要处理的脸时返回原图对象本身。"""
    out, notes = image, []
    for f in faces:
        info, why = analyze(image, f)
        if info is None or info["kind"] == "none":
            notes.append(why)
            continue
        out, extra = apply(out, info, strength)
        notes.append(f"{why}；{extra}")
    return out, notes
