"""光影增强（人像配置，按条件触发，不造新光源）。

只做低风险的两层，都顺着原图已有的光线：
1. 低频明暗塑形：脸上大尺度的明暗（只塑脸，身体和手不动，避免分区交界出现跳变）（受光面/背光面）按比例拉开，不估计光源方向——
   方向本来就写在原图的明暗里。强度随“脸有多平”自适应：脸部大尺度明暗起伏（L 标准差）越小拉得越多，
   起伏已经大于 FLAT_HI（强侧光，如 153228）就不做。单像素改动 ≤ max_dL。
   主体分离：人像分割得到人物，背景亮度最多压 0.5 档；人物本来就比背景暗很多（逆光剪影）时不做。
2. 高光柔辉：只在白天户外（EV ≥ 10）且画面已有明亮高光（L ≥ 90 占比 ≥ 0.3%）时，给高光加一层很淡的
   大半径辉光（只叠加、不压暗），室内、夜景不做。
方向性轮廓光、光束、假镜头光晕这类“造光”不在这里做。返回 (结果, 说明)。
"""
import cv2
import numpy as np

from scripts.filters import blur, masked_blur

FLAT_LO, FLAT_HI = 6.0, 12.0     # 脸部大尺度明暗起伏（L 标准差）：≤6 视为平，≥12 已有足够立体感


def _person_mask(image):
    """人像分割的人物概率（0~1，原图尺寸）；模型不可用时返回 None。"""
    from scripts.person_seg import _load_options
    from scripts.face import _quiet_stderr
    opts = _load_options()
    if not opts:
        return None
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    h, w = image.shape[:2]
    s = 1024 / max(h, w)
    small = np.ascontiguousarray(cv2.cvtColor(cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                                              cv2.COLOR_BGR2RGB))
    with _quiet_stderr(), vision.ImageSegmenter.create_from_options(opts) as seg:
        res = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=small))
    bg = res.confidence_masks[0].numpy_view().reshape(small.shape[:2])
    person = cv2.GaussianBlur(1 - bg, (0, 0), 3)
    return cv2.resize(person, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def light_enhance(image, masks, roi, ev100=None, strength=1.0, max_dL=14.0):
    if strength <= 0:
        return image, ""
    x0, y0, x1, y1 = roi
    notes = []
    lab = cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    L = lab[..., 0]
    face = masks.face_hard > 0
    if face.sum() < 100:
        return image, ""
    fw = float(np.sqrt(face.sum()))
    Lc = L[y0:y1, x0:x1]

    # 1a. 低频明暗塑形（只塑脸；明暗基底在皮肤上求）
    region = (masks.skin_hard > 0).astype(np.float32)
    base = masked_blur(Lc, region, fw / 8)
    spread = float(base[face].std())
    flat = float(np.clip((FLAT_HI - spread) / (FLAT_HI - FLAT_LO), 0, 1))
    if flat > 0:
        mean = float(base[face].mean())
        amt = 0.8 * flat * strength
        # 只塑脸（含五官），并且在脸部轮廓上为 0、向内按距离渐入（fw/5 处满值）：身体和手不动；
        # 轮廓处若仍满值，紧挨身体的受光侧下颌被提亮，脸↔身体交界色差跳变 +2.5~3（验收实测）
        inner = ((masks.tone > 0.5) & (masks.face > 0.05)).astype(np.uint8)
        t = np.clip(cv2.distanceTransform(inner, cv2.DIST_L2, 5) / (fw / 5), 0, 1)
        weight = cv2.GaussianBlur((t * t * (3 - 2 * t)).astype(np.float32), (0, 0), fw / 40)
        dL = np.clip((base - mean) * amt, -max_dL, max_dL) * weight
        L[y0:y1, x0:x1] = Lc + dL
        notes.append(f"塑形（脸部明暗起伏 {spread:.1f}，拉开 {amt:.2f}）")
    else:
        notes.append(f"已有立体感（起伏 {spread:.1f}），不塑形")

    # 1b. 主体分离：背景轻压暗
    person = _person_mask(image)
    if person is not None:
        pm = person > 0.5
        bm = person < 0.2
        if pm.any() and bm.any():
            Lp, Lb = float(np.median(L[pm])), float(np.median(L[bm]))
            if Lp >= Lb - 15:                       # 人物不比背景暗太多（不是逆光剪影）
                k = 2 ** (-0.5 * strength)          # 背景最多压 0.5 档（线性亮度）
                lin = ((L + 16) / 116) ** 3
                L_dark = 116 * np.cbrt(lin * k) - 16
                a = (1 - person) * 1.0
                L = L * (1 - a) + L_dark * a
                notes.append(f"背景压暗 0.5 档（人物 L {Lp:.0f}，背景 L {Lb:.0f}）")
            else:
                notes.append("人物明显比背景暗，背景不压")

    # 2. 高光柔辉：白天户外且已有高光
    if ev100 is not None and ev100 >= 10:
        hi = L >= 90
        if hi.mean() >= 0.003:
            src = np.clip((L - 85) / 15, 0, 1)
            glow = blur(src, max(L.shape) / 120)
            L = L + 12.0 * strength * glow * np.clip((100 - L) / 30, 0, 1)
            notes.append("高光柔辉")
    lab[..., 0] = np.clip(L, 0, 100)
    out = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR) * 255 + 0.5, 0, 255).astype(np.uint8)
    return out, "，光影：" + "、".join(notes)
