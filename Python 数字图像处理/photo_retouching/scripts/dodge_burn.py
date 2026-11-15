"""Dodge & Burn：局部明暗均衡。

只改 Lab 的亮度 L。先去掉纹理得到皮肤明暗 Ls，再用只取皮肤像素的大尺度平均得到“应有的”光影 ref，
偏差 d = Ls - ref 中的小幅度部分（斑驳、局部发暗发亮）被拉平，幅度大的部分（鼻梁、颧骨、下颌的
立体阴影）按 taper 曲线逐渐不动，从而保留脸的立体感。眼下区域单独用更宽的容差来提亮黑眼圈。
"""
import cv2
import numpy as np

from scripts.filters import from_lab, masked_blur, taper, to_lab


def dodge_burn(image, masks, faces, scale, strength=0.5, eye=0.5):
    if strength <= 0 and eye <= 0:
        return image
    lab = to_lab(image)
    L = lab[..., 0]
    Ls = cv2.GaussianBlur(L, (0, 0), max(1.5, scale / 80))
    region = (masks.face_hard > 0).astype(np.float32)
    ref = masked_blur(Ls, region, scale / 10)
    d = Ls - ref

    # 普通皮肤：容差 8 个 L 单位；眼下：容差 14，只提亮
    eye_mask = np.zeros(L.shape, np.float32)
    for f in faces:
        if f.points is not None:
            eye_mask = np.maximum(eye_mask, f.under_eye_mask(image.shape))
    eye_mask *= masks.face > 0.05
    tau = 8.0 + 6.0 * eye_mask
    corr = -taper(d, tau)
    gain = np.where(corr > 0, strength, strength * 0.6)  # 压暗比提亮保守，保留自然高光
    gain = gain * (1 - eye_mask) + np.where(corr > 0, eye, 0.0) * eye_mask

    delta = corr * gain
    # 五官（嘴、鼻孔，以及保护区比五官大出的一圈皮肤）不参与估计，但要跟着周围皮肤一起变，
    # 否则提亮后保护区边缘会留一圈没变亮的硬边；用周围皮肤的校正量插值填进去
    fill = np.clip(masks.tone - masks.skin, 0, 1) if masks.tone is not None else 0.0
    delta_fill = masked_blur(delta, region, scale / 30)
    lab[..., 0] = L + delta * masks.face + delta_fill * fill
    return from_lab(lab)
