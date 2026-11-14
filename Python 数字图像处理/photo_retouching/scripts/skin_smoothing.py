"""多尺度、纹理保护型磨皮。

把皮肤拆成三层（只在皮肤像素内做高斯平均，头发、五官、背景不会混进来）：
  高频 high = I - B1   毛孔、细纹理、噪点/颗粒
  中频 mid  = B1 - B2  痘印、小范围明暗不均、细小褶皱 —— 主要削弱这一层
  低频 low  = B2       肤色和大的光影 —— 不动（由匀肤和 Dodge & Burn 处理）
高频不按固定比例丢弃：先从整张图最平坦的区域（墙、天空、背景）估计相机噪声，再做 Wiener 式收缩，
幅度接近噪声的（颗粒）压掉，明显高于噪声的（毛孔、纹理）保留。鼻子、身体皮肤的中频削弱更少。
"""
import cv2
import numpy as np

from scripts.filters import masked_blur


def estimate_noise(image):
    """相机噪声标准差：细节层在 16x16 块里的标准差，取最平坦的 5% 块。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    detail = gray - cv2.GaussianBlur(gray, (0, 0), 1.0)
    h, w = detail.shape
    h, w = h - h % 16, w - w % 16
    blocks = detail[:h, :w].reshape(h // 16, 16, w // 16, 16).std(axis=(1, 3))
    return max(0.3, float(np.percentile(blocks, 5)))


def skin_smoothing(image, masks, scale, strength=0.7, grain=0.6):
    if strength <= 0 and grain <= 0:
        return image
    img = image.astype(np.float32)
    region = (masks.skin_hard > 0).astype(np.float32)
    b1 = cv2.GaussianBlur(img, (0, 0), max(1.0, scale / 250))
    b2 = masked_blur(b1, region, max(2.0, scale / 40))
    high, mid = img - b1, b1 - b2

    # 分区权重：鼻子毛孔多、保留更多；脸以外的身体皮肤略弱
    region_w = 1.0 - 0.4 * masks.nose - 0.2 * np.clip(masks.skin - masks.face, 0, 1)
    mid_keep = 1.0 - strength * region_w * masks.skin

    # 高频：按相机噪声水平收缩
    sigma_n = estimate_noise(image)
    lum = high.mean(axis=2)
    g = lum ** 2 / (lum ** 2 + (2.0 * sigma_n) ** 2 + 1e-6)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    high_keep = 1.0 - grain * masks.skin * (1.0 - g)

    out = img + mid * (mid_keep - 1)[..., None] + high * (high_keep - 1)[..., None]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
