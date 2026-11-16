"""天空高光恢复：渐变滤镜 × 亮度范围蒙版（摄影师在 Lightroom 里的做法），不划天空边界。

先粗判天空（偏亮、几乎无纹理、白/灰/蓝、与上边缘连通），只用来得到三个数：有没有天空、天空延伸到多低、
天空和地面各自多亮。处理强度 = 竖直渐变（顶部 1，到天空最低处降到 0）× 亮度范围（像素越接近天空亮度
越接近 1，暗的树枝、山体自动为 0）。两者都是连续函数，所以不会在树冠周围留亮圈、在雾里留斑块
——之前“先划出天空再修边界”的做法在这两处都失败过。

天空没有云层次（阴天一片白）时不处理：压暗不会带出任何东西，只会变灰变脏。
恢复：
1. 渐变减光 + 按最亮通道的软肩：接近溢出的云亮部还有层次，压下来就显出轮廓。
2. 三个通道都溢出（>= 254）的像素没有任何信息，不假装恢复细节：用周围未溢出天空的颜色大尺度插值，
   补成平滑渐变，否则压暗后会是一块死平的灰斑。
"""
from dataclasses import dataclass

import cv2
import numpy as np

from scripts.analyze import to_linear, to_srgb
from scripts.exposure import shoulder


@dataclass
class SkyInfo:
    found: bool
    reason: str
    coverage: float = 0.0   # 天空占画面比例（粗判）
    y_end: float = 0.0      # 渐变结束的位置（画面高度的比例）
    L_sky: float = 0.0      # 天空亮度 L*（中位数）
    L_ground: float = 0.0   # 天空以下非天空的亮度 L*（中位数）
    structure: float = 0.0  # 天空里的云层次：去掉大尺度渐变后的 L* 标准差


def analyze_sky(image, work_side=1200):
    h, w = image.shape[:2]
    s = work_side / max(h, w)
    small = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    Lb = cv2.GaussianBlur(L, (0, 0), 1.0)
    texture = cv2.blur(np.hypot(cv2.Sobel(Lb, cv2.CV_32F, 1, 0), cv2.Sobel(Lb, cv2.CV_32F, 0, 1)), (7, 7))
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a)) % 360
    cand = (L > 55) & (texture < 2.5) & ((chroma < 12) | ((hue > 200) & (hue < 300)))
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, cc, st, _ = cv2.connectedComponentsWithStats(cand)
    top_ids = [i for i in set(np.unique(cc[:max(2, small.shape[0] // 50)])) - {0}
               if st[i, cv2.CC_STAT_AREA] > cc.size * 0.01]
    sky = np.isin(cc, top_ids)
    coverage = float(sky.mean())
    if coverage < 0.03:
        return SkyInfo(False, "没有天空")
    rows = sky.mean(axis=1)
    # 渐变结束：天空在这一行已占不到 10% 的宽度
    below = np.where(rows < 0.1)[0]
    below = below[below > np.argmax(rows)]
    y_end = (below[0] if len(below) else len(rows)) / len(rows)
    ground = (~sky) & (np.arange(len(rows))[:, None] < (y_end + 0.15) * len(rows))
    L_sky = float(np.median(L[sky]))
    L_ground = float(np.median(L[ground])) if ground.any() else 0.0
    # 只在天空内部取大尺度平均（不让树、楼的暗色混进来），并且只统计离边界够远的天空
    skyf = sky.astype(np.float32)
    base = cv2.GaussianBlur(L * skyf, (0, 0), 40) / np.maximum(cv2.GaussianBlur(skyf, (0, 0), 40), 1e-4)
    inner = cv2.erode(sky.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    structure = float((L - base)[inner].std()) if inner.sum() > 1000 else 0.0
    return SkyInfo(True, "", coverage, float(y_end), L_sky, L_ground, structure)


def sky_weight(image, info):
    """渐变 × 亮度范围，0~1 float32（原图尺寸）。"""
    h, w = image.shape[:2]
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    end = min(1.0, info.y_end + 0.05)
    t = np.clip(1 - yy / end, 0, 1)
    grad = t * t * (3 - 2 * t)          # 顶部 1，天空最低处平滑降到 0
    L = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)[..., 0].astype(np.float32) / 2.55
    lo = info.L_ground + 0.35 * (info.L_sky - info.L_ground)
    hi = info.L_ground + 0.75 * (info.L_sky - info.L_ground)
    r = np.clip((L - lo) / max(hi - lo, 1e-3), 0, 1)
    r = r * r * (3 - 2 * r)
    return (grad * r).astype(np.float32)


def recover_sky(image, info, strength=1.0, darken=0.6, knee=0.55):
    """返回 (结果, 权重图, 说明)。darken：顶部减光档数（随渐变减弱），knee：最亮通道软肩起点（线性）。"""
    if strength <= 0 or not info.found:
        return image, None, "跳过" if info.found else info.reason
    # 没有云层次的天空（阴天一片白）压暗只会变脏，不处理；有层次才按层次多少加强
    strength *= float(np.clip((info.structure - 0.3) / 0.7, 0, 1))
    if strength <= 0:
        return image, None, f"天空没有层次（{info.structure:.2f}），压暗只会变灰，不处理"
    h, w = image.shape[:2]
    wgt = sky_weight(image, info) * strength
    lin = to_linear(image)
    clipped = (image.min(axis=2) >= 254) & (wgt > 0.3)

    sky = lin
    if clipped.any():
        # 完全溢出：用周围未溢出天空的颜色插值（小图上做归一化卷积）
        s = 0.25
        small = cv2.resize(lin, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        good = cv2.resize(((wgt > 0.3) & ~clipped).astype(np.float32), (small.shape[1], small.shape[0]),
                          interpolation=cv2.INTER_AREA)
        good = (good > 0.99).astype(np.float32)
        sigma = max(small.shape) / 15
        num = cv2.GaussianBlur(small * good[..., None], (0, 0), sigma)
        den = cv2.GaussianBlur(good, (0, 0), sigma)[..., None]
        fill = num / np.maximum(den, 1e-4)
        if good.sum() > 0:
            mean_sky = (small * good[..., None]).sum(axis=(0, 1)) / good.sum()
            fill = np.where(den > 0.02, fill, mean_sky)
        fill = cv2.resize(fill, (w, h), interpolation=cv2.INTER_LINEAR)
        cl = np.clip(cv2.GaussianBlur(clipped.astype(np.float32), (0, 0), max(h, w) / 300) * 1.5, 0, 1)[..., None]
        sky = lin * (1 - cl) + np.minimum(fill, 1.0) * cl

    sky = sky * (2.0 ** (-darken * wgt))[..., None]
    m = sky.max(axis=2)
    white = max(float(np.percentile(m[wgt > 0.3], 99.9)) if (wgt > 0.3).any() else 1.0, 1.0) / 0.92
    m_new = shoulder(m, knee, white)
    sky = sky * (m_new / np.maximum(m, 1e-6))[..., None]
    out = to_srgb(lin * (1 - wgt[..., None]) + sky * wgt[..., None])
    region = wgt > 0.3
    before = float(((image.max(axis=2) >= 254) & region).sum()) / (h * w)
    after = float(((out.max(axis=2) >= 254) & region).sum()) / (h * w)
    return out, wgt, (f"天空约 {info.coverage:.0%}，层次 {info.structure:.1f}，渐变到 {info.y_end:.0%} 高度；"
                      f"溢出 {before:.1%} -> {after:.1%}"
                      f"（其中 {clipped.sum() / (h * w):.1%} 完全溢出，补成平滑渐变）")
