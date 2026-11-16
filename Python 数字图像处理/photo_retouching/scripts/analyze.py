"""照片分析：EXIF、噪声、曝光统计、场景判断。各功能模块共用，不改图。"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from scripts.face import detect_faces

# sRGB <-> 线性，查表加速（8-bit 输入）
_LIN = np.where(np.arange(256) / 255.0 <= 0.04045, np.arange(256) / 255.0 / 12.92,
                ((np.arange(256) / 255.0 + 0.055) / 1.055) ** 2.4).astype(np.float32)


def to_linear(image):
    """BGR uint8 -> 线性 float32 (0~1)。"""
    return _LIN[image]


def to_srgb(lin):
    """线性 float32 -> BGR uint8。"""
    lin = np.clip(lin, 0, 1)
    s = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return np.clip(s * 255 + 0.5, 0, 255).astype(np.uint8)


def luminance(lin):
    """线性 BGR -> 线性亮度 Y。"""
    return 0.0722 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.2126 * lin[..., 2]


def read_exif(path):
    """返回 {iso, exposure, fnumber, focal, lens, camera}，读不到的键缺省。"""
    try:
        from PIL import Image
    except ImportError:
        return {}
    try:
        with Image.open(path) as im:
            ex = im.getexif()
            sub = ex.get_ifd(0x8769)
    except Exception:
        return {}
    out = {}
    for key, tag, src in [("iso", 34855, sub), ("exposure", 33434, sub), ("fnumber", 33437, sub),
                          ("focal", 37386, sub), ("lens", 42036, sub), ("camera", 272, ex)]:
        v = src.get(tag)
        if v is None:
            continue
        if isinstance(v, tuple):
            v = v[0]
        out[key] = v.strip() if isinstance(v, str) else float(v)
    # 焦距换算成像素：焦平面分辨率（像素/英寸或/厘米）× 焦距；没有就用 35mm 等效焦距
    fpx, unit, f35 = sub.get(0xA20E), sub.get(0xA210), sub.get(0xA405)
    if "focal" in out and fpx:
        per_mm = float(fpx) / {2: 25.4, 3: 10.0}.get(int(unit or 2), 25.4)
        out["focal_px"] = out["focal"] * per_mm
    elif f35:
        out["focal35"] = float(f35)
    return out


def noise_level(image):
    """亮度与色度噪声标准差（8-bit 单位）：细节层在 16x16 块里的标准差，取中间亮度块里最平坦的 10%。
    夜景里最平坦的块是死黑的天空和过曝的灯，那里没有噪声，不能参与估计。"""
    ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    h, w = ycc.shape[:2]
    h, w = h - h % 16, w - w % 16
    mean_y = ycc[:h, :w, 0].reshape(h // 16, 16, w // 16, 16).mean(axis=(1, 3))
    usable = (mean_y > 40) & (mean_y < 230)

    def flat_std(ch, sigma):
        detail = ch - cv2.GaussianBlur(ch, (0, 0), sigma)
        blocks = detail[:h, :w].reshape(h // 16, 16, w // 16, 16).std(axis=(1, 3))
        sel = blocks[usable] if usable.sum() > 100 else blocks.ravel()
        return float(np.percentile(sel, 10))
    # 色度噪声多是几个像素大小的彩色斑块（去马赛克 + 机内降噪后），用更大的尺度才量得到
    return flat_std(ycc[..., 0], 1.0), (flat_std(ycc[..., 1], 4.0) + flat_std(ycc[..., 2], 4.0)) / 2


@dataclass
class Analysis:
    exif: dict
    faces: list
    noise_luma: float
    noise_chroma: float
    p50: float          # 线性亮度中位数
    p99: float          # 线性亮度 99.9 分位
    clip: float         # 高光溢出（任一通道 >= 254）比例
    subject: float      # 主体线性亮度：有人脸取脸部皮肤中位数，否则 None
    subject_hi: float   # 脸部皮肤最亮通道的 97 分位（线性）：侧光时受光面，提亮不能把它推到顶
    scene: str          # "portrait" / "night" / "landscape"
    notes: list = field(default_factory=list)

    def summary(self):
        e = self.exif
        shot = []
        if "iso" in e:
            shot.append(f"ISO {e['iso']:.0f}")
        if "exposure" in e:
            shot.append(f"1/{1 / e['exposure']:.0f}s" if e["exposure"] < 1 else f"{e['exposure']:.1f}s")
        if "fnumber" in e:
            shot.append(f"f/{e['fnumber']:.1f}")
        subj = f"，主体 L {lin_to_L(self.subject):.0f}" if self.subject is not None else ""
        return (f"{self.scene}，{' '.join(shot) or '无 EXIF'}，人脸 {len(self.faces)}，"
                f"中位 L {lin_to_L(self.p50):.0f}{subj}，溢出 {self.clip * 100:.2f}%，"
                f"噪声 亮度 {self.noise_luma:.1f} 色度 {self.noise_chroma:.1f}")


def lin_to_L(y):
    """线性亮度 -> CIE L*（0~100），便于阅读。"""
    y = max(float(y), 0.0)
    return 116 * y ** (1 / 3) - 16 if y > 0.008856 else 903.3 * y


def face_skin_stats(lin, lin_y, faces):
    """脸部皮肤（关键点凸包去五官；无关键点取人脸框中部）：(亮度中位数, 最亮通道 97 分位)。"""
    ys, ms = [], []
    lin_max = lin.max(axis=2)
    for f in faces:
        x, y, w, h = f.box
        if f.points is not None:
            mask = np.zeros(lin_y.shape, np.uint8)
            cv2.fillConvexPoly(mask, cv2.convexHull(f.points.astype(np.int32)), 255)
            mask[f.feature_mask(lin_y.shape, grow=w // 30) > 0] = 0
            sel = mask > 0
            ys.append(lin_y[sel])
            ms.append(lin_max[sel])
        else:
            sl = np.s_[max(y + h * 2 // 5, 0):y + h * 7 // 10, max(x + w // 4, 0):x + w * 3 // 4]
            ys.append(lin_y[sl].ravel())
            ms.append(lin_max[sl].ravel())
    ys = np.concatenate(ys) if ys else np.array([])
    if not ys.size:
        return None, None
    return float(np.median(ys)), float(np.percentile(np.concatenate(ms), 97))


def analyze(image, path=None, faces=None, use_landmarks=True):
    exif = read_exif(path) if path else {}
    if faces is None:
        faces = detect_faces(image, use_landmarks=use_landmarks)
    nl, nc = noise_level(image)
    lin = to_linear(image)
    y = luminance(lin)
    small = cv2.resize(y, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    p50, p99 = np.percentile(small, [50, 99.9])
    clip = float((image.max(axis=2) >= 254).mean())
    subject, subject_hi = face_skin_stats(lin, y, faces) if faces else (None, None)
    if faces:
        scene = "portrait"
    elif lin_to_L(p50) < 25:
        scene = "night"
    else:
        scene = "landscape"
    return Analysis(exif, faces, nl, nc, float(p50), float(p99), clip, subject, subject_hi, scene)
