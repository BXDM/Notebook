"""场景分类：白天 / 黄昏 / 夜景 / 室内，外加晚霞强度（0~1）。

依据物理量而不是像素统计阈值：
- 现场亮度 EV100（EXIF 光圈、快门、ISO 算出）：白天户外 10.5 以上，黄昏 6~10，夜景通常 < 5；
- 画面顶部 30% 的亮度中位数：夜景天空是黑的（L ≤ 20），黄昏、白天天空还亮（L ≥ 30）。
  灯火通明的夜景 EV 能到 7（和黄昏重叠），靠这一条分开。
- 晚霞：画面上 60% 范围内、平滑（不是纹理）、偏暖（Lab 色相 320°~80°）、有彩度的区域。
  强度按面积和彩度连续计算（面积 1%→4%、彩度 p90 20→26 平滑过渡），不做硬阈值：
  面积大、颜色浓的晚霞增强多，边缘情况只轻微增强。室内暖色墙面彩度低（p90≈20），自然落到 0。
标定：卡上 55 张人工标注照片（白天 17、晚霞 10、无霞黄昏 7、夜景 14、室内 7）。
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Scene:
    kind: str            # "day" / "dusk" / "night" / "indoor"
    ev100: float         # 可能为 None
    top_L: float
    sky: float           # 平滑、与上边连通的区域占比
    glow: float          # 晚霞强度 0~1
    glow_area: float
    glow_chroma: float
    glow_mask: np.ndarray = None   # 原图尺寸 0~1（只在 glow > 0 时给）

    @property
    def label(self):
        name = {"day": "白天", "dusk": "黄昏", "night": "夜景", "indoor": "室内/近景"}[self.kind]
        ev = f"EV {self.ev100:.1f}，" if self.ev100 is not None else ""
        g = f"，晚霞 {self.glow:.2f}（面积 {self.glow_area:.1%}）" if self.glow > 0 else ""
        return f"{name}（{ev}顶部 L {self.top_L:.0f}）{g}"


def _smooth(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return float(t * t * (3 - 2 * t))


def classify(image, ev100=None, work=600):
    h0, w0 = image.shape[:2]
    s = work / max(h0, w0)
    sm = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(sm.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    H = L.shape[0]
    C = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a)) % 360
    Lb = cv2.GaussianBlur(L, (0, 0), 1.5)
    tex = cv2.blur(np.hypot(cv2.Sobel(Lb, cv2.CV_32F, 1, 0), cv2.Sobel(Lb, cv2.CV_32F, 0, 1)), (7, 7))
    top_L = float(np.median(L[:int(H * 0.3)]))
    smooth = ((tex < 3.0) & (L > 12)).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(smooth)
    top_ids = [i for i in set(np.unique(cc[:3])) - {0} if st[i, cv2.CC_STAT_AREA] > cc.size * 0.01]
    sky = float(np.isin(cc, top_ids).mean())

    if (ev100 is not None and (ev100 < 4.5 or (ev100 < 8 and top_L < 25))) or \
            (ev100 is None and top_L < 25 and np.median(L) < 20):
        kind = "night"
    elif ev100 is not None and ev100 >= 10.5:
        kind = "day"
    elif sky >= 0.03 or top_L >= 30:
        kind = "dusk" if ev100 is not None else ("day" if top_L >= 60 else "dusk")
    else:
        kind = "indoor"

    up = np.zeros_like(L, bool)
    up[:int(H * 0.6)] = True
    warm = ((hue >= 320) | (hue <= 80)) & (C > 12) & (L > 15) & (tex < 4) & up
    area = float(warm.mean())
    chroma = float(np.percentile(C[warm], 90)) if warm.sum() > 20 else 0.0
    glow = 0.0 if kind in ("night", "indoor") else _smooth(area, 0.01, 0.04) * _smooth(chroma, 20, 26)
    mask = None
    if glow > 0:
        m = cv2.GaussianBlur(warm.astype(np.float32), (0, 0), 4)
        mask = cv2.resize(np.clip(m * 1.5, 0, 1), (w0, h0), interpolation=cv2.INTER_LINEAR)
    return Scene(kind, ev100, top_L, sky, glow, area, chroma, mask)
