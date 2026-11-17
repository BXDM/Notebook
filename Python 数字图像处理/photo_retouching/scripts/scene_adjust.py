"""按场景的光线与色彩调整（只改 Lab 的 L 或彩度，色相不动）。

night_light（只对夜景）：
1. 影调：参考作品（8 张 Unsplash 橙青夜景）中位 L≈15、黑位 L≈2。整体很暗时用幂曲线把中位提到 15，
   再把第 1 百分位压回 L≈2；高光端不动。
2. 灯光光晕：灯芯（L>95）周围一圈 L 60~95 的散射光晕按距离减暗最多 25%，灯芯本身不动——灯显得利落，不糊成一团。
3. 被照亮的结构（L 30~80）加一点局部反差：保边滤波取基底（不出光晕），只放大中间尺度。

enhance_glow（晚霞，强度 = scene.glow 0~1，按 scene.glow_mask 作用）：
彩度按“自然饱和度”加：原本越淡加得越多（最多 +40%），已经很浓的几乎不加；
亮的霞光（L>60）稍微压暗最多 6，颜色显得更浓，不发白。地面不受影响（mask 只在画面上 60% 的平滑暖色区）。
"""
import cv2
import numpy as np

from scripts.exposure import guided_coeffs


def _to_lab(img):
    return cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)


def _to_bgr(lab):
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2BGR) * 255 + 0.5, 0, 255).astype(np.uint8)


def _fast_guided(I, r, eps, work=1500):
    H, W = I.shape
    q = min(1.0, work / max(H, W))
    s = cv2.resize(I, None, fx=q, fy=q, interpolation=cv2.INTER_AREA)
    a, b = guided_coeffs(s, s, r, eps)
    return cv2.resize(a, (W, H)) * I + cv2.resize(b, (W, H))


def night_light(image, strength=1.0, mid_target=15.0, halo=0.25, clarity=0.3):
    if strength <= 0:
        return image, "关闭"
    lab = _to_lab(image)
    L = lab[..., 0]
    notes = []
    # 1. 影调
    med = float(np.median(L[::4, ::4]))
    if 0.5 < med < mid_target:
        g = np.log(mid_target / 100.0) / np.log(med / 100.0)
        g = 1 - (1 - max(g, 0.6)) * strength
        L = 100.0 * (np.clip(L, 0, 100) / 100.0) ** g
        notes.append(f"中位 L {med:.0f}→{np.median(L[::4, ::4]):.0f}")
    p1 = float(np.percentile(L[::4, ::4], 1))
    if p1 > 2.0:
        L = np.maximum((L - (p1 - 2.0)) * 100.0 / (100.0 - (p1 - 2.0)), 0)
    # 2. 灯光光晕
    core = (L > 95).astype(np.uint8)
    if core.mean() > 1e-5:
        H, W = L.shape
        q = 0.25
        dist = cv2.distanceTransform(1 - cv2.resize(core, None, fx=q, fy=q, interpolation=cv2.INTER_NEAREST), cv2.DIST_L2, 5) / q
        dist = cv2.resize(dist, (W, H))
        reach = max(H, W) / 60                                  # 光晕范围约画幅 1.7%
        ring = np.exp(-(dist / reach) ** 2) * (dist > 0)
        band = np.clip((L - 60) / 15, 0, 1) * np.clip((97 - L) / 5, 0, 1)
        L = L * (1 - halo * strength * ring * band)
        notes.append("压灯光晕")
    # 3. 局部反差（中间尺度）
    y = L / 100.0
    fine = _fast_guided(y, 2, 1e-3)
    coarse = _fast_guided(y, 15, 1e-3)
    w = np.clip((L - 30) / 15, 0, 1) * np.clip((80 - L) / 15, 0, 1)
    L = np.clip(L + 100.0 * (fine - coarse) * clarity * strength * w, 0, 100)
    lab[..., 0] = L
    return _to_bgr(lab), "夜景光线：" + "，".join(notes + ["建筑局部反差"])


def enhance_glow(image, scene, strength=1.0, max_sat=0.4, darken=6.0):
    k = scene.glow * strength
    if k <= 0 or scene.glow_mask is None:
        return image, "无晚霞"
    lab = _to_lab(image)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    m = scene.glow_mask * k
    gain = 1 + max_sat * m * np.clip(1 - C / 70.0, 0, 1)        # 自然饱和度：越淡加得越多
    lab[..., 1], lab[..., 2] = a * gain, b * gain
    lab[..., 0] = L - darken * m * np.clip((L - 60) / 30, 0, 1)
    return _to_bgr(lab), f"晚霞增强 {k:.2f}"
