"""去眩光（veiling glare）：画面里或画面边上的强光源让镜头内部散射，在整片区域叠一层亮度，
暗处被抬灰、反差变低。5246、5247 脸旁有路灯，脸周围最暗处 L 15、8；干净的照片约 2。

只在人脸周围估计：那里一定有头发、眉毛、鼻孔这类本该接近黑的东西，“附近最暗处 ≈ 眩光层”才成立。
（在整张图上逐区域估计会失败：白墙、灰白天空附近没有暗的东西，它们自身的亮度会被当成眩光——
暗通道去雾的经典失效，5 张样片全被误判过。）
模型：I = J + V，V 取脸周围逐通道最暗值（带灯光颜色），在人物区域内当作常数。
去除：out = (I − s·V·w) · B / (B − s·V·w)，B 为局部平均亮度、w 为人物区域的平滑权重：
最暗处回到接近 0，局部平均亮度不变——恢复反差而不把脸压暗；人物区域外平滑过渡到不处理。
"""
import cv2
import numpy as np

from scripts.analyze import to_linear, to_srgb


def _L(y):
    y = max(float(y), 0.0)
    return 116 * y ** (1 / 3) - 16 if y > 0.008856 else 903.3 * y


def face_floor(lin, face):
    """脸周围逐通道的最暗值（15 像素最小值滤波后的 2 分位，线性）。"""
    x, y, w, h = face.box
    H, W = lin.shape[:2]
    roi = lin[max(y - w // 2, 0):min(y + 2 * w, H), max(x - w // 2, 0):min(x + w + w // 2, W)]
    se = np.ones((15, 15), np.uint8)
    return np.array([np.percentile(cv2.erode(roi[..., c], se), 2) for c in range(3)], np.float32)


def deveil(image, faces, strength=0.8, min_veil_L=4.0):
    """返回 (结果, 说明)。脸周围最暗处低于 L min_veil_L 时视为没有眩光，不处理。"""
    if strength <= 0 or not faces:
        return image, "跳过"
    lin = to_linear(image)
    H, W = lin.shape[:2]
    veil = np.zeros_like(lin)
    weight = np.zeros((H, W), np.float32)
    levels = []
    for f in faces:
        V = face_floor(lin, f)
        v_L = _L(V.min())
        levels.append(v_L)
        if v_L < min_veil_L:
            continue
        x, y, w, _ = f.box
        q = 1 / 16  # 人物区域的平滑权重在缩小图上算（大半径模糊在原图上要几十秒）
        m = np.zeros((int(H * q) + 1, int(W * q) + 1), np.float32)
        cx = x + w / 2
        cv2.rectangle(m, (int((cx - 3 * w) * q), int((y - w) * q)), (int((cx + 3 * w) * q), int((y + 7 * w) * q)), 1.0, -1)
        m = cv2.resize(cv2.GaussianBlur(m, (0, 0), w * q), (W, H), interpolation=cv2.INTER_LINEAR)
        veil = np.maximum(veil, V[None, None, :] * m[..., None])
        weight = np.maximum(weight, m)
    if not weight.any():
        return image, "眩光弱（脸周最暗 L " + "、".join(f"{v:.1f}" for v in levels) + "），跳过"
    sV = strength * veil
    q = 0.25
    B = cv2.resize(cv2.GaussianBlur(cv2.resize(lin, None, fx=q, fy=q, interpolation=cv2.INTER_AREA), (0, 0),
                                    max(1.0, faces[0].box[2] * q / 6)), (W, H), interpolation=cv2.INTER_LINEAR)
    # 局部平均接近眩光层的极暗处，放大倍数会失控、放大噪点：限制在 3 倍以内
    out = (lin - sV) * (B / np.maximum(B - sV, B / 3 + 1e-6))
    return to_srgb(np.clip(out, 0, 1)), "眩光（脸周最暗 L " + "、".join(f"{v:.1f}" for v in levels) + f"），去除 {strength:.0%}"
