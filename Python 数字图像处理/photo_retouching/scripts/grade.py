"""风格调色：橙青（orange & teal）。所有参数都是感知量（Lab 色相角、L*、彩度），对任何照片同一套规则。

1. 色相归拢：暖色族（橙、黄，及偏暖的黄绿；正红保持不动）往橙色（Lab 色相约 60°）转，冷色族（绿、青、蓝）往青色
   （约 210°）转；品红一带不动，避免肤色、霓虹粉被拉坏。转动量 = pull × 隶属度，族边界平滑过渡。
2. 分离色调：暗部和中性色（天空、墙、玻璃）染青，亮部染橙；按“1 − 现有彩度”加权，已经很鲜艳的颜色不再叠色。
   明暗分界取本张照片亮度的第 60 百分位（L 12~35），夜景整体很暗时分界随之下移。
3. 饱和度：橙、青两族各加一点，彩度上限 cap，暗部（L<15）的彩度不加（那里多是噪点，加了就是彩色颗粒）。
4. 影调：整体很暗的夜景用幂曲线把中位亮度提到 mid_target（参考作品中位 L≈15），黑点、高光不动。
参数来源：8 张 Unsplash 橙青夜景的统计（暗部色 8.0/244°、亮部近中性、黑位 L1≈2、中位 L≈15）。
夜景暗部噪点多：调用方应先降噪（denoise_mottle）。
"""
import cv2
import numpy as np

ORANGE, TEAL = 60.0, 235.0   # 参考作品统计：暗部青色色相中位 244°（Lab），橙色在 35~85°


def _hdiff(h, c):
    return (h - c + 180.0) % 360.0 - 180.0


def _member(h, center, half, soft=25.0):
    """色相 h 到 center 的距离在 half 内为 1，向外 soft 度平滑降到 0。"""
    d = np.abs(_hdiff(h, center))
    t = np.clip((half + soft - d) / soft, 0, 1)
    return t * t * (3 - 2 * t)


def orange_teal(image, strength=1.0, pull=0.6, split=11.0, warm_split=0.3, sat=0.2, cap=60.0, mid_target=15.0):
    """返回 (结果, 说明)。strength 统一缩放所有效果，0 不处理。"""
    if strength <= 0:
        return image, "关闭"
    lab = cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    h = np.degrees(np.arctan2(b, a)) % 360

    # 1. 色相归拢（彩度很低的像素色相不可靠，按彩度渐入）
    warm = _member(h, 50.0, 55.0)          # 约 -5°~105°：红到黄绿
    warm = warm * np.clip((h - 15.0) / 25.0, 0, 1) * (h < 180) + warm * (h >= 180)  # 正红（<15°）不动，15~40° 渐入：红桥、红灯笼保持红
    cool = _member(h, 205.0, 75.0)         # 约 130°~280°：绿到蓝
    rel = np.clip(C / 8.0, 0, 1)
    dh = (warm * _hdiff(ORANGE, h) + cool * _hdiff(TEAL, h)) * pull * strength * rel
    h2 = np.radians(h + dh)

    # 3. 饱和度（暗部不加）
    dark_w = np.clip((L - 8) / 12, 0, 1)
    C2 = C * (1 + sat * strength * (warm + cool) * dark_w)
    C2 = np.minimum(C2, np.maximum(C, cap))
    a2, b2 = C2 * np.cos(h2), C2 * np.sin(h2)

    # 2. 分离色调：暗部→青，亮部→橙；中性色才叠
    neutral = np.clip(1 - C / 25.0, 0, 1)
    # 明暗分界按本张照片的亮度分布：取第 60 百分位（限制在 L 12~35），夜景整体很暗时分界也跟着下移
    pivot = float(np.clip(np.percentile(L[::4, ::4], 60), 12, 35))
    tl = np.clip((L - pivot) / 40, -1, 1)
    tl = np.where(tl < 0, np.clip((L - pivot) / pivot, -1, 0), tl)   # 暗部这一侧按分界缩放，满幅在 L→0
    # 暗部染青；亮部只按 warm_split 轻染（参考作品的中性亮部几乎不偏色，橙只在本来就暖的灯光上）
    tone = split * strength * neutral * np.abs(tl) * np.where(tl < 0, 1.0, warm_split)
    ang = np.where(tl < 0, np.radians(TEAL), np.radians(ORANGE))
    shadow_floor = np.clip(L / 2.5, 0, 1)           # 只有真正的黑（L<2.5）不染色
    a2 = a2 + tone * np.cos(ang) * shadow_floor
    b2 = b2 + tone * np.sin(ang) * shadow_floor

    # 4. 影调：参考作品黑位 L1≈2、中位 L≈15。整体很暗（中位 < mid_target）时用幂曲线把中位提到目标，
    #    黑（L→0）和高光（L→100）不动；不再抬黑（会发灰）
    med = float(np.median(L[::4, ::4]))
    if 0.5 < med < mid_target:
        g = np.log(mid_target / 100.0) / np.log(med / 100.0)          # 让中位落到目标
        g = 1 - (1 - max(g, 0.6)) * strength
        L2 = 100.0 * (np.clip(L, 0, 100) / 100.0) ** g
    else:
        L2 = L
    # 黑点：第 1 百分位压回 L≈2（参考作品黑位），线性重映射，高光端不动
    p1 = float(np.percentile(L2[::4, ::4], 1))
    if p1 > 2.0:
        L2 = np.maximum((L2 - (p1 - 2.0)) * 100.0 / (100.0 - (p1 - 2.0)), 0)

    out = cv2.cvtColor(np.stack([L2, a2, b2], -1).astype(np.float32), cv2.COLOR_Lab2BGR)
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8), f"橙青 强度 {strength:.2f}"
