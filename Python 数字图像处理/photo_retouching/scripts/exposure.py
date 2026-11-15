"""亮度校正：主体提亮到目标亮度，高光按区域软压缩，纹理不变。

在线性亮度的对数域里用导向滤波取“照明层”base（保边，避免灯旁光晕），只对 base 做色调映射，
得到增益图 gain = curve(k·base) / base，再乘回线性 RGB，细节层原样保留。
- k：曝光倍数。人像让脸部皮肤到目标 L（默认 58）；夜景、风景按中位亮度温和调整；最多 +2 档。
- curve：膝点以下恒等（不改变中间调对比），膝点以上用有理函数软肩把最亮的 base 收回到 1，
  所以提亮后路灯、招牌不会新增溢出。
- 极暗处（夜空、阴影深处）增益逐渐回到 1，保留夜景的黑。
导向滤波在长边约 1500 的缩小图上算系数再放大（fast guided filter），24MP 只需约 1 秒。
"""
import cv2
import numpy as np

from scripts.analyze import luminance, to_linear, to_srgb


def L_to_lin(L):
    f = (L + 16) / 116
    return f ** 3 if f ** 3 > 0.008856 else L / 903.3


def _box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided_coeffs(I, p, r, eps):
    """导向滤波系数 (a, b)，输出 = a * I + b。"""
    mI, mp = _box(I, r), _box(p, r)
    cov = _box(I * p, r) - mI * mp
    var = _box(I * I, r) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return _box(a, r), _box(b, r)


def illumination(y, r_frac=1 / 40, eps=0.05, work_side=1500):
    """线性亮度 y 的照明层（对数域导向滤波，按 fast guided filter 在缩小图上求系数）。"""
    logy = np.log(y + 1e-4)
    h, w = y.shape
    s = min(1.0, work_side / max(h, w))
    small = cv2.resize(logy, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    r = max(2, int(max(small.shape) * r_frac))
    a, b = guided_coeffs(small, small, r, eps)
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
    b = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.exp(a * logy + b) - 1e-4


def shoulder(x, knee, white):
    """膝点以下 y = x；以上斜率从 1 平滑减小，x = white 时 y = 1。"""
    if white <= 1.0:
        return np.minimum(x, 1.0)
    c = ((white - knee) / (1 - knee) - 1) / (white - knee)
    d = np.maximum(x - knee, 0)
    return np.where(x <= knee, x, knee + d / (1 + c * d))


def exposure_factor(an, target_L=58.0, strength=1.0):
    """返回曝光倍数 k 和说明。"""
    if an.subject is not None:
        # 偏暗才提亮；在目标到 72 之间不动；更亮（脸过曝）才往回压到 70
        face_L = _L(an.subject)
        goal = target_L if face_L < target_L else (70.0 if face_L > 72 else face_L)
        k = float(np.clip(L_to_lin(goal) / max(an.subject, 1e-4), 0.7, 4.0))
        why = f"脸部 L {face_L:.0f} 到 {goal:.0f}"
        # 侧光时脸的受光面已经很亮：提亮后它的最亮通道不超过 0.8（sRGB 约 231）
        if an.subject_hi and k > 1 and k * an.subject_hi > 0.8:
            k = max(1.0, 0.8 / an.subject_hi)
            why += f"，受光面限制为 L {_L(k * an.subject):.0f}"
    elif an.scene == "night":
        k = float(np.clip(L_to_lin(28) / max(an.p50, 1e-4), 1.0, 2.0))
        why = f"夜景中位 L {_L(an.p50):.0f}"
    else:
        k = float(np.clip(L_to_lin(48) / max(an.p50, 1e-4), 0.8, 2.0))
        why = f"中位 L {_L(an.p50):.0f}"
    return k ** strength, why


def _L(y):
    y = max(float(y), 0.0)
    return 116 * y ** (1 / 3) - 16 if y > 0.008856 else 903.3 * y


def gain_map(image, an, target_L=58.0, strength=1.0, knee=0.5):
    """返回 (增益图 float32, 曝光倍数 k, 说明)。"""
    lin = to_linear(image)
    y = luminance(lin)
    base = illumination(y)
    k, why = exposure_factor(an, target_L, strength)
    white = k * float(np.percentile(base[::4, ::4], 99.9))
    mapped = shoulder(k * base, knee, white)
    gain = mapped / np.maximum(base, 1e-4)
    # 极暗处增益回到 1：base 低于 L≈3 不动，L≈12 以上全额
    t = np.clip((base - 0.003) / (0.014 - 0.003), 0, 1)
    t = t * t * (3 - 2 * t)
    gain = 1 + (gain - 1) * t
    return gain.astype(np.float32), k, why


def apply_gain(image, gain, knee=0.6):
    """乘增益后按每个像素的最亮通道做软肩（保持 RGB 比例即色相、饱和度），只压提亮后新超出的部分。
    照明层的软肩管不到纹理里更亮的像素（白衣服、受光的脸颊），皮肤的红通道也比亮度先到顶，所以按最亮通道算。"""
    lin0 = to_linear(image)
    lin = lin0 * gain[..., None]
    m0, m = lin0.max(axis=2), lin.max(axis=2)
    white = max(float(np.percentile(m[::4, ::4], 99.95)), 1.0)
    m_new = shoulder(m, knee, white)
    m_new = np.maximum(np.minimum(m_new, m), np.minimum(m0, m))  # 不比原图更暗
    lin *= (np.minimum(m_new, 1.0) / np.maximum(m, 1e-6))[..., None]
    return to_srgb(lin)
