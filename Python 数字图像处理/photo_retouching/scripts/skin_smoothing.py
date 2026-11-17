"""多尺度、纹理保护型磨皮。

把皮肤拆成三层（只在皮肤像素内做高斯平均，头发、五官、背景不会混进来）：
  高频 high = I - B1   毛孔、细纹理、噪点/颗粒
  中频 mid  = B1 - B2  痘印、小范围明暗不均、细小褶皱 —— 主要削弱这一层（开阔皮肤全额；五官、轮廓交界带只削小起伏）
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


def boundary_zone(skin_hard, width):
    """皮肤内、离非皮肤（保护区或皮肤外）不足 width 像素的交界带：边上为 1，向内平滑降到 0。"""
    dist = cv2.distanceTransform((skin_hard > 0).astype(np.uint8), cv2.DIST_L2, 5)
    t = np.clip(1 - dist / max(width, 1.0), 0, 1)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def skin_smoothing(image, masks, scale, strength=0.7, grain=0.6):
    if strength <= 0 and grain <= 0:
        return image
    img = image.astype(np.float32)
    region = (masks.skin_hard > 0).astype(np.float32)
    b1 = cv2.GaussianBlur(img, (0, 0), max(1.0, scale / 250))
    b2 = masked_blur(b1, region, max(2.0, scale / 40))
    high, mid = img - b1, b1 - b2

    # 分区权重：脸以外的身体皮肤略弱。鼻子不再减弱——鼻梁鼻翼的毛孔、油光颗粒正是最显眼的“坑洼”
    region_w = 1.0 - 0.2 * np.clip(masks.skin - masks.face, 0, 1)
    mid_keep = 1.0 - strength * region_w * masks.skin

    # 高频：按相机噪声水平收缩
    sigma_n = estimate_noise(image)
    lum = high.mean(axis=2)
    g = lum ** 2 / (lum ** 2 + (2.0 * sigma_n) ** 2 + 1e-6)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    high_keep = 1.0 - grain * masks.skin * (1.0 - g)
    # 鼻子：毛孔在这里最密最深，高频在噪声收缩之外再削一半（仍保留一半质感）
    high_keep *= 1.0 - 0.5 * masks.nose * masks.skin

    # 分区处理：开阔皮肤全额去中频（那里的大起伏就是痘印、斑驳，应该去掉）；
    # 只在皮肤与非皮肤（鼻孔、嘴、眼、发丝、脸部轮廓）的交界带里按幅度保留大起伏——那里的大起伏是
    # 结构阴影，全额去掉会把鼻孔周围、脸边抹成一圈发白的硬边。交界带宽度 2 倍中频半径（硬边只出现在这个范围内），权重平滑过渡。
    zone = boundary_zone(masks.skin_hard, scale / 20)
    m = np.abs(mid.mean(axis=2))
    small = np.exp(-(m / 25.0) ** 2)  # 起伏 10 灰度时磨皮效果保留 85%，25 时 37%，50 以上基本不动
    factor = (1 - zone) + zone * small
    out = img + mid * ((mid_keep - 1) * factor)[..., None] + high * (high_keep - 1)[..., None]
    out = _soften_glints(out, masks, scale)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _soften_glints(img, masks, scale, keep=0.35):
    """油光颗粒：比周围亮、只有几个像素大的高光点（鼻梁、额头最多）。用小尺度白顶帽找出，压到 keep；
    大片的柔和高光（脸颊受光面）尺度大，不受影响。只在皮肤里做。"""
    L = img.mean(axis=2)
    k = max(3, int(scale / 60)) | 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = L - cv2.morphologyEx(L, cv2.MORPH_OPEN, se)
    tophat = np.clip(tophat - 3.0, 0, None)            # 3 灰度以内是纹理和噪声，不算油光
    w = masks.skin * (0.5 + 0.5 * masks.nose)           # 鼻子全额，其他皮肤减半
    return img - (tophat * (1 - keep) * w)[..., None]
