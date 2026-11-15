"""自适应降噪：强度由实测噪声和随后的提亮增益决定，噪声本来就低（相机 JPEG 机内已降噪）时不动。

- 色度：Cr/Cb 用亮度作导向的导向滤波，大半径去彩色斑点，边缘跟着亮度走不串色。
- 亮度：自导向的导向滤波，局部方差低于噪声方差的地方被抹平，纹理、边缘（方差远大于噪声）保留；
  再按 keep 留一部分原始颗粒，避免塑料感。
噪声在提亮后才显眼，所以“有效噪声” = 实测噪声 × 该处增益；低于可见阈值（亮度 visible、色度 visible_c，
色度的估计尺度更大、数值偏小）时该处不处理。
"""
import cv2
import numpy as np

from scripts.exposure import guided_coeffs


def _guided(I, p, r, eps):
    a, b = guided_coeffs(I, p, r, eps)
    return a * I + b


def denoise(image, noise_luma, noise_chroma, gain=None, strength=1.0, keep=0.3, visible=1.5, visible_c=0.8):
    """返回 (结果, 说明)。noise_* 为 8-bit 单位的标准差。"""
    if strength <= 0:
        return image, "关闭"
    g = np.ones(image.shape[:2], np.float32) if gain is None else gain
    g_med = float(np.median(g))
    eff_l, eff_c = noise_luma * g_med, noise_chroma * g_med
    if eff_l < visible and eff_c < visible_c:
        return image, f"噪声低（亮度 {eff_l:.1f} 色度 {eff_c:.1f}，提亮后），跳过"

    ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y = ycc[..., 0]
    # 每个像素的处理比例：有效噪声越过可见阈值越多，处理越足
    w_l = np.clip((noise_luma * g - visible) / visible, 0, 1) * strength
    w_c = np.clip((noise_chroma * g - visible_c) / visible_c, 0, 1) * strength
    if eff_c >= visible_c:
        eps_c = (3 * noise_chroma) ** 2
        for c in (1, 2):
            sm = _guided(Y / 255, ycc[..., c] / 255, 8, eps_c / 255 ** 2) * 255
            ycc[..., c] += (sm - ycc[..., c]) * w_c
    if eff_l >= visible:
        sm = _guided(Y / 255, Y / 255, 2, (2 * noise_luma) ** 2 / 255 ** 2) * 255
        ycc[..., 0] = Y + (sm - Y) * w_l * (1 - keep)
    out = cv2.cvtColor(np.clip(ycc + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    return out, f"亮度 {eff_l:.1f} 色度 {eff_c:.1f}（提亮后）"
