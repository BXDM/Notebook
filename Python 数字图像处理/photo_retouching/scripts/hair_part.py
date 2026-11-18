"""发缝填充（原型，默认关闭）：降低发缝/头顶稀疏处露出的头皮与周围发色的对比。

只做“颜色 + 纹理迁移”，不生成新发丝、不涂黑：
1. 头发区域：BiSeNet 人脸分割的 hair 类（复用 scripts/face_parsing 的模型会话，裁剪框上移以覆盖头顶），
   闭运算 + 填洞得到“头发包络”，头皮只在包络内部（离包络边缘、额头皮肤都有一段距离）找；
2. 头皮判别（Lab）：比局部发色亮一截、色度接近额头肤色、局部纹理弱于头发——受光头发虽然亮，
   但色度贴着发色、而且有密集发丝纹理，不会被当成头皮；
3. 连通域分类：细长的算“发缝”（正常发型，只减淡一半对比），成片的算“稀疏区”（减淡更多）；
   面积太小、对比不够、发色与肤色太接近（浅色/染发）等情况一律不处理；
4. 修补：亮度底色换成周围头发的局部均值（归一化卷积，保留邻近光照），
   纹理沿发丝方向（结构张量方向场）从周围头发传播进来，并保留原图中横跨发缝的细发丝；
   色度迁移到局部发色；软掩码羽化后与原图混合。

接口：fill_hair_part(image, faces, strength=1.0) -> (结果, 说明列表)；没处理时返回原图对象本身。
"""
import cv2
import numpy as np

from scripts import face_parsing as fp

_ANA = 512          # 分析分辨率（头部裁剪的边长）
PART_STRENGTH = 0.55     # 正常细发缝：只减淡约一半，保留发型本身的分界
SPARSE_STRENGTH = 0.85   # 稀疏区：减淡更多


def _parse_head(crop):
    """在头部裁剪上跑 BiSeNet，返回 (hair, skin) 概率图（与裁剪同尺寸）；模型不可用时返回 None。"""
    session = fp._load()
    if not session:
        return None
    inp = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (512, 512), interpolation=cv2.INTER_AREA)
    inp = ((inp.astype(np.float32) / 255.0 - fp._MEAN) / fp._STD).transpose(2, 0, 1)[None]
    logits = session.run(None, {session.get_inputs()[0].name: inp})[0][0]
    prob = np.exp(logits - logits.max(axis=0))
    prob /= prob.sum(axis=0)
    size = (crop.shape[1], crop.shape[0])
    hair = cv2.resize(prob[[fp.HAIR]].sum(axis=0), size, interpolation=cv2.INTER_LINEAR)
    skin = cv2.resize(prob[[fp.SKIN, fp.NOSE]].sum(axis=0), size, interpolation=cv2.INTER_LINEAR)
    hat = cv2.resize(prob[fp.HAT], size, interpolation=cv2.INTER_LINEAR)
    return hair, skin, hat


def _lab(bgr):
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def _norm_blur(val, weight, sigma):
    """归一化卷积：只用 weight>0 的像素求局部加权均值，返回 (均值, 权重和)。"""
    w = cv2.GaussianBlur(weight, (0, 0), sigma)
    if val.ndim == 3:
        v = cv2.GaussianBlur(val * weight[..., None], (0, 0), sigma)
        return v / np.maximum(w, 1e-6)[..., None], w
    v = cv2.GaussianBlur(val * weight, (0, 0), sigma)
    return v / np.maximum(w, 1e-6), w


def _fill_holes(mask):
    m = mask.copy()
    h, w = m.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    inv = m.copy()
    cv2.floodFill(inv, ff, (0, 0), 255)       # 左上角一般在头外（裁剪框比头大）
    return m | cv2.bitwise_not(inv)


def _ray_hits(mask, radius):
    """对每个像素统计 8 个方向中，在 radius 距离内碰到 mask 的方向数（0~8）。"""
    h, w = mask.shape
    hits = np.zeros((h, w), np.float32)
    steps = np.arange(2, int(radius) + 1, 2)
    for ang in np.arange(8) * np.pi / 4:
        dx, dy = np.cos(ang), np.sin(ang)
        acc = np.zeros((h, w), np.uint8)
        for r in steps:
            M = np.float32([[1, 0, -dx * r], [0, 1, -dy * r]])     # 把 (x+dx*r, y+dy*r) 的值搬到 (x, y)
            acc |= cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        hits += acc
    return hits


def _ray_hits_dir(mask, d, radius):
    """沿单个方向 d（单位向量）radius 距离内是否碰到 mask，返回 0/1 float32。"""
    h, w = mask.shape
    acc = np.zeros((h, w), np.uint8)
    for r in np.arange(2, int(radius) + 1, 2):
        M = np.float32([[1, 0, -d[0] * r], [0, 1, -d[1] * r]])
        acc |= cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return acc.astype(np.float32)


def _head_box(face, shape):
    """以额头顶点为参考取正方形头部裁剪框（按脸的上方向外扩，覆盖整个头顶）。"""
    p = face.points
    fw = float(np.linalg.norm(p[454] - p[234]))
    up = p[10] - p[152]
    up /= np.linalg.norm(up) + 1e-6
    c = p[10] + up * 0.2 * fw
    half = 0.95 * fw
    h, w = shape[:2]
    x0, y0 = int(max(c[0] - half, 0)), int(max(c[1] - half, 0))
    x1, y1 = int(min(c[0] + half, w)), int(min(c[1] + half, h))
    return (x0, y0, x1, y1), fw, up


def analyze(image, face):
    """分析一张脸的头顶，返回 dict（含 status/说明/掩码等），供 fill 和开发工具使用。"""
    if face.points is None:
        return {"status": "skip", "reason": "没有关键点"}
    (x0, y0, x1, y1), fw, up = _head_box(face, image.shape)
    if fw < 120:
        return {"status": "skip", "reason": f"脸太小（脸宽 {fw:.0f}px）"}
    if abs(face.pose[1]) > 35 if face.pose else False:
        return {"status": "skip", "reason": "俯仰角过大"}
    crop = image[y0:y1, x0:x1]
    s = _ANA / max(crop.shape[:2])
    small = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    parsed = _parse_head(small)
    if parsed is None:
        return {"status": "skip", "reason": "人脸分割模型不可用"}
    hair_p, skin_p, hat_p = parsed
    fws = fw * s                                      # 分析分辨率下的脸宽
    h, w = small.shape[:2]
    if (hat_p > 0.5).mean() > 0.02:
        return {"status": "skip", "reason": "戴帽子"}

    hair = (hair_p > 0.5).astype(np.uint8) * 255
    if hair.mean() / 255 < 0.03:
        return {"status": "skip", "reason": "头发区域太小（光头或被遮挡）"}
    k = int(fws * 0.12) | 1
    env = cv2.morphologyEx(hair, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    env = _fill_holes(env)
    # 离包络边缘有一段距离（排除发际轮廓外的背景透光），离额头皮肤也有一段距离（排除发际线本身）
    e = int(fws * 0.03) | 1
    inner = cv2.erode(env, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e)))
    skin = (skin_p > 0.5).astype(np.uint8) * 255
    d = int(fws * 0.12) | 1
    near_skin = cv2.dilate(skin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    # 只看头顶：沿脸的上方向，10 号点（额头上沿）以上 0.08 倍脸宽才算；
    # 再往下是额头，刘海缝隙里透出的额头皮肤最容易被误判成头皮
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    p10 = (face.points[10] - (x0, y0)) * s
    height = (xx - p10[0]) * up[0] + (yy - p10[1]) * up[1]
    # 手/身体（摸头发的手指等）：mediapipe 多类人像分割的身体皮肤
    body = _person_body(small)
    region = (inner > 0) & (near_skin == 0) & (height > 0.08 * fws) & (body < 0.3)

    lab = _lab(small)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    # 额头肤色：分割为皮肤、且在 10 号点附近（眉上额头）
    pts = (face.points - (x0, y0)) * s
    brow_h = ((pts[[105, 334], 0] - p10[0]) * up[0] + (pts[[105, 334], 1] - p10[1]) * up[1]).mean()
    fore = (skin_p > 0.8) & (height < -0.02 * fws) & (height > brow_h + 0.03 * fws) \
        & (np.abs((xx - p10[0]) * up[1] - (yy - p10[1]) * up[0]) < 0.25 * fws)
    if fore.sum() < 30:             # 刘海遮住额头：退回裁剪内所有确定是皮肤的像素（脸颊上部）
        fore = skin_p > 0.85
    if fore.sum() < 30:
        return {"status": "skip", "reason": "取不到肤色"}
    skin_ab = np.array([np.median(A[fore]), np.median(B[fore])])
    skin_L = float(np.median(L[fore]))

    # 局部发色：确定是头发且不太亮（排除可能的头皮）的像素做归一化卷积
    hair_core = (hair_p > 0.7) & region
    if hair_core.sum() < 200:
        return {"status": "skip", "reason": "头顶头发区域太少"}
    hair_L_med = float(np.median(L[hair_core]))
    hair_L_p90 = float(np.percentile(L[hair_core], 90))
    hair_ab = np.array([np.median(A[hair_core]), np.median(B[hair_core])])
    dark = (hair_p > 0.5) & (L < np.percentile(L[hair_core], 70))
    sig = fws * 0.05
    loc, wsum = _norm_blur(np.dstack([L, A, B]), dark.astype(np.float32), sig)
    ab_gap = float(np.linalg.norm(skin_ab - hair_ab))
    if skin_ab[0] < 4:
        return {"status": "skip", "reason": f"肤色偏离正常（a={skin_ab[0]:.1f}，单色/偏色照片）"}
    if hair_L_med > 40:
        return {"status": "skip", "reason": f"浅色头发（L={hair_L_med:.0f}），第一版只处理深色头发"}
    if ab_gap < 5 or skin_L - hair_L_med < 15:
        return {"status": "skip", "reason": f"发色与肤色太接近（色度差 {ab_gap:.1f}，亮度差 {skin_L - hair_L_med:.0f}）"}

    # 公共条件：被头发包围、不在发际线上、不像背景
    # 被头发包围：从像素出发向 8 个方向看 R 距离内是否碰到深色头发，碰不到的方向多于 2 个就不算头皮
    hair_hit = ((hair_p > 0.5) & (L < hair_L_med + 12)).astype(np.uint8)
    enclosed = np.clip(_ray_hits(hair_hit, fws * 0.12) - 5.0, 0, 1)   # ≥6 个方向（发缝沿线两个方向本来就碰不到）
    # 发际线：沿脸的下方向 0.12 倍脸宽内碰到额头皮肤的，算发际线附近（稀发透出额头），不处理
    down_skin = _ray_hits_dir((skin_p > 0.5).astype(np.uint8), -up, fws * 0.12)
    # 背景透光：头发轮廓外一圈（不含皮肤）的局部颜色
    ring = (cv2.dilate(env, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(fws * 0.2) | 1,) * 2)) > 0) \
        & (env == 0) & (skin_p < 0.3)
    bg, bw_ = _norm_blur(lab, ring.astype(np.float32), fws * 0.15)
    has_bg = bw_ > 1e-3
    d_bg = np.sqrt((0.5 * (L - bg[..., 0])) ** 2 + (A - bg[..., 1]) ** 2 + (B - bg[..., 2]) ** 2)
    d_sk = np.sqrt((0.5 * (L - skin_L)) ** 2 + (A - skin_ab[0]) ** 2 + (B - skin_ab[1]) ** 2)
    common = region * enclosed * (1 - down_skin)

    dL = L - loc[..., 0]
    hp = L - cv2.GaussianBlur(L, (0, 0), max(1.0, fws * 0.006))
    tex = np.sqrt(cv2.GaussianBlur(hp * hp, (0, 0), max(1.5, fws * 0.012)))
    tex_hair = float(np.median(tex[hair_core]))
    smooth = np.clip(1.6 - tex / max(tex_hair, 0.5) * 0.6, 0, 1)
    # A 类“透肤色”：比局部发色亮、色度在“发色→肤色”方向上偏向肤色（稀疏处透出的头皮）
    ab = np.dstack([A, B])
    d_skin = np.linalg.norm(ab - skin_ab, axis=2)
    v = skin_ab - hair_ab
    t = ((ab - loc[..., 1:]) * v).sum(axis=2) / (v @ v)
    chroma = np.clip((t - 0.15) / 0.3, 0, 1) * (d_skin < 14)   # 色度必须真的接近肤色（排除发饰/花朵等）
    not_bg = np.where(has_bg, np.clip((d_bg - d_sk) / 6, 0, 1), 1)
    score_a = np.clip((dL - 6) / 10, 0, 1) * chroma * (0.5 + 0.5 * smooth) * common * not_bg
    # B 类“灰白发缝”：实拍里密发的发缝常是偏灰白的一条线（发根 + 头皮反光），色度接近中性甚至和发色一样，
    # 只比周围头发明显亮；颜色上和高光分不开（只排除明显偏蓝的天光高光），靠后面的几何条件：
    # 细长线形、大致沿脸的上下方向、横穿发丝方向（高光和发丝间隙都是顺着发丝的）
    warm = np.clip((B - loc[..., 2] + 3) / 2, 0, 1) * ((A - loc[..., 1]) > -6)
    score_b = np.clip((dL - 15) / 10, 0, 1) * warm * common * np.where(has_bg, d_bg > 12, 1)
    score_a = cv2.GaussianBlur(score_a, (0, 0), 0.8)
    score_b = cv2.GaussianBlur(score_b, (0, 0), 0.8)
    # 发丝方向场（单位向量，已归一化卷积延伸进候选区）
    sdx, sdy = _orientation(L, dark.astype(np.float32), max(0.7, fws * 0.003), fws * 0.03)
    dbg = dict(warm=warm, dL=dL / 50, locb=loc[..., 2], score_a=score_a, score_b=score_b, smooth=smooth, region=region.astype(np.float32),
               enclosed=enclosed, hair=hair_p, body=body)

    keep = np.zeros((h, w), np.uint8)
    final = np.zeros((h, w), np.float32)
    comps, rejected = [], []
    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(fws * 0.04) | 1,) * 2)
    for mode, sc in (("a", score_a), ("b", score_b)):
        m = (sc > 0.35).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, lab_cc, st, _ = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            area = st[i, cv2.CC_STAT_AREA]
            if area < (fws * 0.02) ** 2:
                continue
            ci = (lab_cc == i).astype(np.uint8)
            rg = cv2.dilate(ci, ring_k) & (1 - ci)
            enc = float((hair_p[rg > 0] > 0.5).mean())
            # 形状：主轴方向、长、宽（PCA）
            ys, xs = np.nonzero(ci)
            cov = np.cov(np.vstack([xs, ys]).astype(np.float32))
            ev, evec = np.linalg.eigh(cov)
            axis = evec[:, 1]
            length = 4 * np.sqrt(max(ev[1], 1e-6))
            width = area / max(length, 1)
            # 两侧头发的发丝方向与候选主轴的夹角：分别取主轴两侧平移出去的一条带，倍角平均（方向无符号）。
            # 发缝两侧的发丝呈 V 形分开，两侧合起来平均会恰好和发缝平行，所以必须分侧算
            nrm = np.array([-axis[1], axis[0]])
            off = width / 2 + fws * 0.03
            grown = cv2.dilate(ci, ring_k)
            angs, cohs = [], []
            for sg in (1, -1):
                M = np.float32([[1, 0, sg * nrm[0] * off], [0, 1, sg * nrm[1] * off]])
                side = (cv2.warpAffine(grown, M, (w, h), flags=cv2.INTER_NEAREST) > 0) & (grown == 0)
                if side.sum() < 5:
                    continue
                c2 = float(np.mean(sdx[side] ** 2 - sdy[side] ** 2))
                s2 = float(np.mean(2 * sdx[side] * sdy[side]))
                th = 0.5 * np.arctan2(s2, c2)
                angs.append(np.degrees(np.arccos(min(1.0, abs(float(axis @ (np.cos(th), np.sin(th))))))))
                cohs.append(np.hypot(c2, s2))
            ang_strand = float(np.mean(angs)) if angs else 90.0
            coh = float(np.mean(cohs)) if cohs else 0.0
            ang_up = np.degrees(np.arccos(min(1.0, abs(float(axis @ up)))))
            elong = length > 2.5 * width
            c = dict(mode=mode, area=int(area), length=length / fws, width=width / fws,
                     contrast=float(dL[ci > 0].mean()), enclosed=enc, ang_strand=float(ang_strand),
                     ang_up=float(ang_up), coh=float(coh), smooth=float(smooth[ci > 0].mean()))
            why = None
            if mode == "a" and area < (fws * 0.03) ** 2:
                why = "零星小亮点（逆光碎发/发丝间隙）"
            elif mode == "a" and c["smooth"] < 0.3:
                why = "纹理和头发一样密（是亮发丝，不是头皮）"
            elif mode == "a" and elong and ang_up > 60:
                why = "横向亮带（挑染/发色过渡/高光带），发缝应大致前后走向"
            elif enc < 0.65:
                why = "周围不是头发"
            elif length > 1.5 * width and ang_strand < 30 and coh > 0.3:
                why = "顺着发丝的亮条（高光/发丝间隙）"
            elif mode == "b" and not (elong and length > 0.10 * fws and width < 0.07 * fws
                                      and ang_up < 50 and ang_strand > 30):
                why = "灰白亮区不是线形发缝"
            if why:
                c["why"] = why
                rejected.append(c)
                continue
            c["kind"] = "part" if (mode == "b" or (width < fws * 0.035 and elong)) else "sparse"
            comps.append(c)
            keep[ci > 0] = 1
            np.maximum(final, sc * cv2.dilate(ci, np.ones((3, 3), np.uint8)), out=final)
    total = keep.sum() / fws ** 2
    info = dict(status="skip", box=(x0, y0, x1, y1), scale=s, fw=fw, comps=comps, rejected=rejected,
                dbg=dbg, small=small, area=total, hair_p90=hair_L_p90, skin_L=skin_L, hair_L=hair_L_med, ab_gap=ab_gap)
    if not comps:
        info["reason"] = "未见明显发缝/稀疏"
        return info
    contrast = float(np.average([c["contrast"] for c in comps], weights=[c["area"] for c in comps]))
    info["contrast"] = contrast
    if total < 0.0015:
        info["reason"] = f"露头皮面积太小（{total * 1e4:.0f}‱ 脸宽²）"
        return info
    if contrast < 12:
        info["reason"] = f"头皮与发色对比不明显（ΔL {contrast:.0f}）"
        return info
    if total > 0.12:
        info["reason"] = f"露头皮面积过大（{total:.2f} 脸宽²），超出第一版范围"
        return info
    sparse_area = sum(c["area"] for c in comps if c["kind"] == "sparse") / fws ** 2
    info.update(status="do", kind="sparse" if sparse_area > total * 0.4 else "part",
                score=final, hair_dark=dark.astype(np.float32))
    return info


def _person_body(small):
    """mediapipe 多类人像分割（复用 scripts/person_seg 的模型配置）里“身体皮肤”的概率；不可用时全 0。"""
    try:
        from scripts import person_seg as ps
        options = ps._load_options()
        if not options:
            return np.zeros(small.shape[:2], np.float32)
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        with ps._quiet_stderr(), vision.ImageSegmenter.create_from_options(options) as seg:
            masks = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).confidence_masks
            body = masks[ps.CLASSES["body"]].numpy_view().reshape(small.shape[:2]).copy()
        return cv2.dilate(body, np.ones((5, 5), np.uint8))
    except Exception:
        return np.zeros(small.shape[:2], np.float32)


def _orientation(L, weight, sigma_g, sigma_t):
    """结构张量求发丝方向（单位向量场），张量用 weight 归一化卷积，延伸进头皮区域。"""
    g = cv2.GaussianBlur(L, (0, 0), sigma_g)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    J, _ = _norm_blur(np.dstack([gx * gx, gx * gy, gy * gy]), weight, sigma_t)
    jxx, jxy, jyy = J[..., 0], J[..., 1], J[..., 2]
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)       # 梯度主方向
    # 发丝方向与梯度垂直
    return -np.sin(theta), np.cos(theta)


def _propagate(val, known, dx, dy, step, iters):
    """沿方向场双向传播：未知像素反复取 ±step 方向上邻居的加权均值（类似线积分卷积的扩散）。"""
    h, w = val.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    v = val * known
    k = known.astype(np.float32).copy()
    maps = [(xx + dx * step * sgn, yy + dy * step * sgn) for sgn in (1, -1)]
    for _ in range(iters):
        acc_v, acc_k = v.copy(), k.copy()
        for mx, my in maps:
            acc_v += cv2.remap(v, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            acc_k += cv2.remap(k, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        unknown = known < 0.5
        v = np.where(unknown, acc_v / 3, v)
        k = np.where(unknown, acc_k / 3, k)
    return v / np.maximum(k, 1e-4), k


def _apply(image, info, strength):
    x0, y0, x1, y1 = info["box"]
    crop = image[y0:y1, x0:x1]
    H, W = crop.shape[:2]
    fw = info["fw"]
    up = lambda a: cv2.resize(a.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)  # noqa: E731
    base_k = PART_STRENGTH if info["kind"] == "part" else SPARSE_STRENGTH
    # 软掩码：得分图上采样后羽化，边缘约 0.6% 脸宽
    score = np.clip(up(info["score"]) * 1.6, 0, 1)
    mask = cv2.GaussianBlur(score, (0, 0), max(1.0, fw * 0.004))
    alpha = np.clip(mask * base_k * strength, 0, 1)
    if alpha.max() < 0.05:
        return image, False
    lab = _lab(crop)
    L = lab[..., 0]
    hair_w = up(info["hair_dark"]) * (1 - np.clip(cv2.GaussianBlur(score, (0, 0), fw * 0.01) * 3, 0, 1))
    hair_w = (hair_w > 0.5).astype(np.float32)
    # 低频底色：周围头发的局部均值（保留邻近的光照走势）
    loc, _ = _norm_blur(lab, hair_w, fw * 0.03)
    # 细节层：原图（含横跨发缝的细发丝）与头发的纹理
    sig_d = max(1.0, fw * 0.004)
    detail = L - cv2.GaussianBlur(L, (0, 0), sig_d)
    # 方向场（在下采样上算更稳，再放大；方向是无符号的，插值用张量更好，这里分辨率够用直接放大单位向量）
    sa = 0.5
    Ls = cv2.resize(L, None, fx=sa, fy=sa, interpolation=cv2.INTER_AREA)
    ws = cv2.resize(hair_w, (Ls.shape[1], Ls.shape[0]), interpolation=cv2.INTER_AREA)
    dx, dy = _orientation(Ls, (ws > 0.5).astype(np.float32), max(1.0, fw * 0.003 * sa), fw * 0.02 * sa)
    dx, dy = up(dx), up(dy)
    nrm = np.sqrt(dx * dx + dy * dy) + 1e-6
    dx, dy = dx / nrm, dy / nrm
    known = (alpha < 0.02).astype(np.float32) * hair_w
    # 传播区域只限掩码附近，节省时间
    ys, xs = np.where(alpha > 0.02)
    pad = int(fw * 0.08)
    by0, by1 = max(ys.min() - pad, 0), min(ys.max() + pad, H)
    bx0, bx1 = max(xs.min() - pad, 0), min(xs.max() + pad, W)
    step = max(1.5, fw * 0.003)
    width_px = max(c["width"] for c in info["comps"]) * fw
    iters = int(np.clip(width_px / step * 1.5, 6, 60))
    tex = np.zeros_like(L)
    sub = (slice(by0, by1), slice(bx0, bx1))
    t, kk = _propagate(detail[sub], known[sub], dx[sub], dy[sub], step, iters)
    # 扩散会把纹理抹平：按已知头发的细节强度把幅度补回来
    amp_hair = np.sqrt(_norm_blur(detail[sub] ** 2, known[sub], fw * 0.03)[0])
    amp_t = np.sqrt(cv2.GaussianBlur(t * t, (0, 0), fw * 0.01)) + 1e-3
    tex[sub] = t * np.clip(amp_hair / amp_t, 0, 2.5) * (kk > 0.05)
    # 目标：底色 = 头发局部均值；细节 = 迁移纹理 + 部分保留原图细节（原有发丝）
    tgt_L = loc[..., 0] + 0.75 * tex + 0.5 * detail
    new = lab.copy()
    new[..., 0] = L + alpha * (tgt_L - L)
    for c in (1, 2):
        new[..., c] = lab[..., c] + alpha * (loc[..., c] - lab[..., c])
    out_bgr = cv2.cvtColor(new, cv2.COLOR_LAB2BGR)
    res = image.copy()
    res[y0:y1, x0:x1] = np.clip(out_bgr * 255 + 0.5, 0, 255).astype(np.uint8)
    return res, True


def fill_hair_part(image, faces, strength=1.0, infos=None):
    """发缝填充。返回 (结果, 说明列表)；没有处理时结果就是传入的 image 对象。
    infos：可选的列表，开发工具用来拿到每张脸的分析结果。"""
    notes = []
    if not faces:
        return image, ["没有人脸，跳过发缝填充"]
    out = image
    for i, f in enumerate(faces):
        info = analyze(image, f)
        if infos is not None:
            infos.append(info)
        tag = f"脸{i + 1}"
        if info["status"] != "do":
            notes.append(f"{tag}: 跳过（{info['reason']}）")
            continue
        out, done = _apply(out, info, strength)
        kind = "细发缝" if info["kind"] == "part" else "头顶稀疏"
        notes.append(f"{tag}: {kind}，面积 {info['area'] * 1e4:.0f}‱ 脸宽²，ΔL {info['contrast']:.0f}"
                     + ("" if done else "（掩码太弱，未改动）"))
    return out, notes
