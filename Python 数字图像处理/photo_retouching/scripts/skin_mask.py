"""皮肤与保护区域蒙版。

优先：人脸语义分割（脸、鼻、耳、脖子、五官、头发像素级区分）+ 关键点（鼻孔）；
分割覆盖不到的身体皮肤（肩、手臂）用从脸上采样的自适应肤色模型补充，且必须与脸/脖子相连。
分割不可用时退回纯肤色模型 + 关键点挖五官。
"""
from dataclasses import dataclass

import cv2
import numpy as np

from scripts.face import nostril_parts_mask
from scripts.face_parsing import parse_faces


@dataclass
class Masks:
    skin: np.ndarray        # 0~1 float32，所有要处理的皮肤（羽化），供磨皮、美白、锐化保护
    skin_hard: np.ndarray   # 0/255 uint8
    face: np.ndarray        # 0~1 float32，脸部皮肤（含鼻，去五官），供匀肤、光影、祛痘
    face_hard: np.ndarray   # 0/255 uint8，脸部（不含脖子）
    protect: np.ndarray     # 0/255 uint8，五官、鼻孔、头发：任何修复都不碰
    nose: np.ndarray        # 0~1 float32，鼻子区域（毛孔明显，磨皮减弱）
    method: str             # "parsing" / "color"


def feature_scale(image, faces):
    """各步骤半径的参考尺寸：有人脸用脸宽，否则用短边的 1/3。"""
    if faces:
        return float(np.median([f.box[2] for f in faces]))
    return min(image.shape[:2]) / 3.0


def _ellipse(k):
    k = max(1, int(k)) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _color_model(image, faces, sample_mask=None):
    """自适应肤色模型：返回 (马氏距离², 亮度下限)；无人脸返回 None。"""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    if sample_mask is not None and np.count_nonzero(sample_mask) > 200:
        samples = ycrcb[sample_mask > 0]
    else:
        samples = []
        for f in faces:
            x, y, w, h = f.box
            if f.points is not None:
                region = cv2.erode(f.part_mask(image.shape, ["oval"]), np.ones((9, 9), np.uint8))
                region[f.feature_mask(image.shape, grow=w // 30) > 0] = 0
                samples.append(ycrcb[region > 0])
            else:
                samples.append(ycrcb[y + h * 2 // 5:y + h * 7 // 10, x + w // 4:x + w * 3 // 4].reshape(-1, 3))
        if not samples:
            return None
        samples = np.concatenate(samples)
    samples = samples.astype(np.float32)
    # 去掉亮度两端（阴影、高光）后再估计肤色分布
    lo, hi = np.percentile(samples[:, 0], [10, 90])
    core = samples[(samples[:, 0] >= lo) & (samples[:, 0] <= hi), 1:]
    mean = np.median(core, axis=0)
    cov = np.cov(core, rowvar=False) + np.eye(2, dtype=np.float32) * 2.0
    diff = ycrcb[..., 1:].reshape(-1, 2).astype(np.float32) - mean
    dist = np.einsum("ij,jk,ik->i", diff, np.linalg.inv(cov), diff).reshape(image.shape[:2])
    return dist, lo


def _clean(hard, scale):
    k = max(3, int(scale / 40) | 1)
    hard = cv2.morphologyEx(hard, cv2.MORPH_OPEN, _ellipse(k))
    return cv2.morphologyEx(hard, cv2.MORPH_CLOSE, _ellipse(k), iterations=2)


def _keep_connected(hard, seed, min_area):
    """只保留与 seed 相交且面积足够的连通域。"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hard)
    keep = np.zeros(n, bool)
    keep[np.unique(labels[seed > 0])] = True
    keep[0] = False
    keep &= stats[:, cv2.CC_STAT_AREA] >= min_area
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def _fixed_ycrcb(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2]
    return (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127) & (y > 40) & (y < 250)


def skin_mask(image, faces, use_parsing=True):
    h, w = image.shape[:2]
    scale = feature_scale(image, faces)
    y_chan = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[..., 0]
    feather = scale / 30

    # 关键点给出的保护区（五官 + 鼻孔）
    protect = np.zeros((h, w), np.uint8)
    for f in faces:
        if f.points is not None:
            protect |= f.feature_mask(image.shape, grow=f.box[2] // 60)
            protect |= nostril_parts_mask(f, image.shape, grow=f.box[2] // 80)

    parsed = parse_faces(image, faces) if use_parsing else None
    if parsed is not None:
        method = "parsing"
        face_hard = (parsed["face"] > 0.5).astype(np.uint8) * 255
        head_skin = face_hard | (((parsed["ears"] > 0.5) | (parsed["neck"] > 0.5)).astype(np.uint8) * 255)
        protect |= ((parsed["features"] > 0.4) | (parsed["hair"] > 0.5)).astype(np.uint8) * 255

        # 身体皮肤（胸口、肩、手臂）：用脸部皮肤采样的肤色模型，排除分割认定的头发、衣物、饰品，
        # 且必须与脸/脖子相连（种子区放大一圈，跨过项链、颈带这类细的遮挡）
        sample = cv2.erode((parsed["face"] > 0.8).astype(np.uint8) * 255, _ellipse(scale / 40))
        sample[protect > 0] = 0
        dist, lo = _color_model(image, faces, sample)
        not_skin = (parsed["hair"] > 0.5) | (parsed["features"] > 0.4) | (parsed["cloth"] > 0.5)
        body = ((dist < 9.0) & (y_chan > lo * 0.6) & ~not_skin).astype(np.uint8) * 255
        body = _clean(body, scale)
        body = _keep_connected(body, cv2.dilate(head_skin, _ellipse(scale / 5)), (scale / 6) ** 2)
        skin_hard = head_skin | body
        nose = cv2.GaussianBlur(parsed["nose"], (0, 0), feather)
    else:
        method = "color"
        model = _color_model(image, faces)
        if model is not None:
            dist, lo = model
            hard = (dist < 9.0) & (y_chan > lo * 0.6)
            for f in faces:
                if f.points is not None:  # 脸部轮廓内放宽条件，补进阴影、高光处的皮肤
                    oval = f.part_mask(image.shape, ["oval"]) > 0
                    hard |= oval & (dist < 25.0) & (y_chan > lo * 0.45)
        else:
            hard = _fixed_ycrcb(image)
        hard &= (y_chan > 40) & (y_chan < 250)
        hard = _clean(hard.astype(np.uint8) * 255, scale)
        seed = np.zeros((h, w), np.uint8)
        for x, y, bw, bh in (f.box for f in faces):
            seed[y + bh // 4:min(y + bh * 3 // 2, h), x + bw // 4:x + bw * 3 // 4] = 255
        # 没有人脸时不猜皮肤：整图肤色模型会把路面、树干、衣服都算进去，皮肤类步骤全部跳过
        skin_hard = _keep_connected(hard, seed, (scale / 6) ** 2)
        # 脸部 = 脸部轮廓（或人脸框内切椭圆）∩ 皮肤
        face_hard = np.zeros((h, w), np.uint8)
        for f in faces:
            if f.points is not None:
                face_hard |= f.part_mask(image.shape, ["oval"])
            else:
                x, y, bw, bh = f.box
                cv2.ellipse(face_hard, (x + bw // 2, y + bh // 2), (bw * 2 // 5, bh // 2), 0, 0, 360, 255, -1)
        face_hard &= skin_hard
        nose = np.zeros((h, w), np.float32)
        for f in faces:
            if f.points is not None:
                nose = np.maximum(nose, f.part_mask(image.shape, ["nose"]).astype(np.float32) / 255)
        nose = cv2.GaussianBlur(nose, (0, 0), feather)

    skin_hard[protect > 0] = 0
    face_hard[protect > 0] = 0
    skin = cv2.GaussianBlur(skin_hard.astype(np.float32) / 255.0, (0, 0), feather)
    face = cv2.GaussianBlur(face_hard.astype(np.float32) / 255.0, (0, 0), feather)
    # 羽化不能把效果晕进五官和头发
    hard_protect = protect > 0
    skin[hard_protect] *= 0.0
    face[hard_protect] *= 0.0
    return Masks(skin, skin_hard, face, face_hard, protect, nose, method)
