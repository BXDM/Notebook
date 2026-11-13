import cv2
import numpy as np

def feature_scale(image, faces):
    """各步骤半径的参考尺寸：有人脸用脸宽，否则用短边的 1/3。"""
    if faces:
        return float(np.median([f.box[2] for f in faces]))
    return min(image.shape[:2]) / 3.0


def skin_mask(image, faces):
    """返回 (hard, soft) 两个皮肤掩码：hard 为 0/255 uint8，soft 为 0~1 float32（已羽化）。

    有人脸时在脸上采样 Cr/Cb 做自适应肤色模型（有关键点时取轮廓内去掉五官的区域，
    否则取人脸框中心），没有人脸时退回 YCrCb 固定阈值。有关键点时还会：
    把脸部轮廓内的阴影补进来，并挖掉眉、眼、嘴，避免磨糊五官。
    """
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    crcb = ycrcb[..., 1:].reshape(-1, 2).astype(np.float32)

    y_chan = ycrcb[..., 0]
    samples = []
    for f in faces:
        x, y, w, h = f.box
        if f.points is not None:
            region = cv2.erode(f.part_mask(image.shape, ["oval"]), np.ones((9, 9), np.uint8))
            region[f.feature_mask(image.shape, grow=w // 30) > 0] = 0
            samples.append(ycrcb[region > 0])
        else:
            # 取脸颊到鼻子的中心区域采样，避开眼睛、眉毛和嘴
            samples.append(ycrcb[y + h * 2 // 5:y + h * 7 // 10, x + w // 4:x + w * 3 // 4].reshape(-1, 3))
    if samples:
        samples = np.concatenate(samples).astype(np.float32)
        # 去掉亮度两端（阴影、高光）后再估计肤色分布
        lo, hi = np.percentile(samples[:, 0], [10, 90])
        core = samples[(samples[:, 0] >= lo) & (samples[:, 0] <= hi), 1:]
        mean = np.median(core, axis=0)
        cov = np.cov(core, rowvar=False) + np.eye(2, dtype=np.float32) * 2.0
        diff = crcb - mean
        dist = np.einsum("ij,jk,ik->i", diff, np.linalg.inv(cov), diff)
        dist = dist.reshape(image.shape[:2])
        hard = dist < 3.0 ** 2
        # 头发和肤色色相接近但更暗
        hard &= y_chan > lo * 0.6
        # 脸部轮廓内放宽条件，把阴影、高光处的皮肤也补进来（遮住脸的深色头发仍排除）
        for f in faces:
            if f.points is not None:
                oval = f.part_mask(image.shape, ["oval"]) > 0
                hard |= oval & (dist < 5.0 ** 2) & (y_chan > lo * 0.45)
    else:
        cr, cb = ycrcb[..., 1], ycrcb[..., 2]
        hard = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    # 太暗或过曝的像素颜色不可靠
    hard &= (y_chan > 40) & (y_chan < 250)
    hard = hard.astype(np.uint8) * 255

    scale = feature_scale(image, faces)
    k = max(3, int(scale / 40) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    hard = cv2.morphologyEx(hard, cv2.MORPH_OPEN, kernel)
    hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 去掉零散小块，只保留面积足够大的连通域
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hard)
    min_area = (scale / 6) ** 2
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    if faces:
        # 有人脸时只保留与人脸（含脖子）区域相连的皮肤，排除背景中的近肤色物体
        near_face = np.zeros(n, bool)
        for x, y, w, h in (f.box for f in faces):
            region = labels[y + h // 4:min(y + h * 3 // 2, labels.shape[0]), x + w // 4:x + w * 3 // 4]
            near_face[np.unique(region)] = True
        keep &= near_face
    hard = np.where(keep[labels], 255, 0).astype(np.uint8)

    # 挖掉眉、眼、嘴
    for f in faces:
        if f.points is not None:
            hard[f.feature_mask(image.shape, grow=f.box[2] // 60) > 0] = 0

    soft = cv2.GaussianBlur(hard.astype(np.float32) / 255.0, (0, 0), scale / 30)
    return hard, soft
