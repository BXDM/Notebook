"""匀肤（色彩）：只在 Lab 的 a/b 色度通道上做低频均衡，去掉泛红、色斑、眼下发青发紫，不改明暗。

明暗由 dodge_burn 负责，纹理由 skin_smoothing 负责，三者分工互不干扰。
"""
import cv2
import numpy as np

from scripts.filters import from_lab, masked_blur, taper, to_lab


def frequency_separation(image, masks, scale, strength=0.5, ref_div=8.0):
    """ref_div：参考色的平均半径 = 脸宽 / ref_div。8（R10 配置）会连腮红这种大片渐变一起拉平；
    人像配置用 20，只拉平比脸宽 1/20 小的斑驳、泛红块，腮红、修容的大面积色彩层次保留。"""
    if strength <= 0:
        return image
    lab = to_lab(image)
    ab = lab[..., 1:]
    ab_s = cv2.GaussianBlur(ab, (0, 0), max(1.5, scale / 80))
    region = (masks.skin_hard > 0).astype(np.float32)
    ref = masked_blur(ab_s, region, scale / ref_div)
    corr = -taper(ab_s - ref, 8.0)
    lab[..., 1:] = ab + corr * (strength * masks.skin)[..., None]
    return from_lab(lab)
