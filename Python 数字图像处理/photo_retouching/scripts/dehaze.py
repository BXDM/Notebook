"""分层去雾（保留空气透视）：中景、近景去雾加反差，远景的雾留着做纵深。

参照用户 Lightroom 修的 IMG_5703（雾浓度 = 暗通道 L）：远景平原 81→81 不动，远山 73→70，
中景山脊 49→40，近景小镇 36→26；反差中景、近景 +27~29%。

模型（暗通道去雾）：I = J·t + A·(1−t)，A 为大气光（雾的颜色），雾浓度 h = 暗通道 / A。
J = (I − A)/t + A，t = 1 − ω(h)·h。ω 基础 0.5，雾越浓越小
（h 从 0.30 到 0.65 平滑降到 0），所以远景几乎不动、中近景去掉约一半的雾——和上面的参照基本吻合。
暗通道先做最小值滤波再用导向滤波（以原图亮度为导向）细化，边缘不出光晕。
只在户外（有天空）做：室内白墙、阴天白楼的暗通道也高，会被误当成雾；天空本身雾浓度最高，自然不动。
只对大尺度基底去雾，细节层最多放大 1.15 倍（整图除以透射率会把纹理和 JPEG 压缩块放大到发脆）。
最后在中近景加一点清晰度（只加强中间尺度），同样随雾浓度减弱，远景和天空不加。
"""
import cv2
import numpy as np

from scripts.analyze import to_linear, to_srgb
from scripts.exposure import guided_coeffs


def _guided(I, p, r, eps):
    a, b = guided_coeffs(I, p, r, eps)
    return a * I + b


def _fast_guided(I, p, work_side, r, eps):
    """快速导向滤波：系数在缩小图上求，放大后用原分辨率的引导图合成，边缘仍按原图对齐（不出光晕）。"""
    H, W = I.shape
    q = min(1.0, work_side / max(H, W))
    Is = cv2.resize(I, None, fx=q, fy=q, interpolation=cv2.INTER_AREA)
    ps = cv2.resize(p, None, fx=q, fy=q, interpolation=cv2.INTER_AREA)
    a, b = guided_coeffs(Is, ps, r, eps)
    a = cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
    b = cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR)
    return a * I + b


def atmosphere(lin_small, dark):
    """大气光 A：暗通道最高的 1% 像素（去掉已溢出的）的逐通道中位数。"""
    ok = lin_small.max(axis=2) < 0.98
    d = np.where(ok, dark, -1)
    thr = np.percentile(d[ok], 99) if ok.any() else d.max()
    sel = (d >= thr) & ok
    A = np.median(lin_small[sel], axis=0) if sel.any() else np.ones(3, np.float32)
    # 天空整片溢出时，排除溢出后剩下的是楼、树，A 会被估成深灰（5193 估出 0.42）；白天户外的雾不会这么暗
    return A if A.mean() >= 0.6 else np.ones(3, np.float32)


def haze_map(lin, work_side=1500):
    """返回 (雾浓度 h，原图尺寸 0~1, 大气光 A)。"""
    H, W = lin.shape[:2]
    s = work_side / max(H, W)
    small = cv2.resize(lin, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    k = max(3, int(work_side / 100)) | 1
    dark = cv2.erode(small.min(axis=2), np.ones((k, k), np.uint8))
    A = atmosphere(small, dark)
    h = cv2.erode((small / np.maximum(A, 1e-3)).min(axis=2), np.ones((k, k), np.uint8))
    gray = small.mean(axis=2)
    h = np.clip(_guided(gray, h, k * 2, 1e-3), 0, 1)
    return cv2.resize(h, (W, H), interpolation=cv2.INTER_LINEAR), A


def dehaze(image, strength=1.0, omega=0.5, fade=(0.30, 0.65), clarity=0.25, detail_gain=1.15):
    """返回 (结果, 说明)。"""
    if strength <= 0:
        return image, "关闭"
    lin = to_linear(image)
    h, A = haze_map(lin)
    t = np.clip((h - fade[0]) / (fade[1] - fade[0]), 0, 1)
    w = omega * strength * (1 - t * t * (3 - 2 * t))      # 雾越浓去得越少，远景保留
    trans = np.clip(1 - w * h, 0.3, 1)[..., None]
    # 只在大尺度基底上去雾：整图除以透射率会把最细的纹理和 JPEG 压缩块一起放大 1.3~2 倍，显得发脆。
    # 细节层的放大封顶 detail_gain；基底用全分辨率引导的导向滤波（保边），树冠对天空不出光晕
    y = lin.mean(axis=2)
    base_y = _fast_guided(y, y, 1500, 2, 1e-3)
    ratio = (base_y / np.maximum(y, 1e-4))[..., None]
    base = lin * np.clip(ratio, 0, 4)
    detail = lin - base
    J = (base - A) / trans + A + detail * np.minimum(1 / trans, detail_gain)
    J = np.clip(J, 0, 1)

    if clarity > 0:
        # 清晰度：只加强中间尺度（细基底 − 粗基底），最细的纹理不动；按去雾权重分布（远景、天空 w≈0 不加）
        y = J.mean(axis=2)
        fine = _fast_guided(y, y, 1500, 2, 1e-3)
        coarse = _fast_guided(y, y, 1500, 15, 1e-3)
        amt = clarity * strength * (w / omega)
        y_new = np.clip(y + (fine - coarse) * amt, 0, 1)
        J = J * (y_new / np.maximum(y, 1e-4))[..., None]

    hh = float(np.median(h))
    return to_srgb(np.clip(J, 0, 1)), f"雾浓度中位 {hh:.2f}，大气光 {A.round(2).tolist()}"
