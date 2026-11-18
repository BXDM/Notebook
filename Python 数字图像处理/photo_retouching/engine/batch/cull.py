"""选片（批次层，先于单张精修）：主体模糊、闭眼、连拍重复分组（每组选一张），只标记不删除，召回优先。

每张照片：
- 主体清晰度：最大的那张脸，在**原图分辨率**上两只眼睛分开量，取清楚的那只（浅景深时远处那只眼虚是正常的）：
  每只眼缩到眼宽 160 像素（顺带平均掉噪点颗粒），最强 1% 边缘的梯度 / 眼部亮度范围。
  X-H2 176 张实测（原图核对标注）：模糊 0.31~0.70，清晰 0.78~1.36；拉普拉斯能量、细/粗尺度比都分不开
  （噪点被当成边缘，缩图量又丢了细节）。拖影（沿一个方向仍有陡边）靠同组对比抓。
  没有人脸时取全图最清晰的一块，只做组内对比（背景虚化不算模糊）。
- 闭眼：MediaPipe 表情系数 eyeBlink，取闭得更紧的那只眼（单眼闭也算）；同时在往下看（eyeLookDown 高）记为“垂眼”，
  多半是故意的，不算废片。
- 画面特征：DINOv2-small 的 CLS 向量（Apache-2.0，Hugging Face facebook/dinov2-small），用于连拍分组。
  缩略图保持原比例（长边 224，边长取 14 的倍数）：压成正方形时换场景的两张也能有 0.88 的相似度，分不开。
分组：按文件名（相机编号连续）排序，与上一张拍摄间隔 ≤ GAP_S 秒、与上一张相似度 ≥ SIM、
且与本组平均特征的相似度 ≥ SIM_GROUP 才归入同一组。
组内：分数 = 清晰度 / 组内最清晰（闭眼 ×0.3），最高的为“最佳”，其余为“重复”。
模糊：有人脸时清晰度低于 BLUR_ABS，或低于同组最清晰的 BLUR_REL 倍。组内最佳本身也模糊或闭眼时同样算废片（召回优先）。
"""
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

GAP_S = 10.0
SIM = 0.83           # 与上一张（保持原比例的缩略图）：同组连拍 ≥ 0.835（含模糊帧），换场景/机位 ≤ 0.82（X-H2 实测）
SIM_GROUP = 0.80     # 与整组平均特征：防止一张张都像上一张、顺着链条串到别的场景
BLUR_ABS = 0.72
BLUR_REL = 0.55
BLINK = 0.55
LOOK_DOWN = 0.45


@dataclass
class Shot:
    path: str
    time: float = None
    sharp: float = None
    has_face: bool = False
    blink: float = 0.0
    look_down: float = 0.0
    smile: float = 0.0
    emb: np.ndarray = None
    tags: list = field(default_factory=list)
    group: int = -1
    best: bool = False

    @property
    def name(self):
        return os.path.basename(self.path)


def _exif_time(path):
    from datetime import datetime
    from PIL import Image
    try:
        with Image.open(path) as im:
            sub = im.getexif().get_ifd(0x8769)
        t = sub.get(36867)
        sec = sub.get(37521)
        ts = datetime.strptime(t, "%Y:%m:%d %H:%M:%S").timestamp()
        return ts + (float("0." + str(sec).strip()) if sec else 0.0)
    except Exception:
        return None


def _edge_steepness(gray, width=160):
    """缩到固定宽度后，最强 1% 边缘的梯度 / 亮度范围（p99 − p1）：对亮度、对比度、噪点颗粒不敏感。"""
    g = gray.astype(np.float32)
    s = width / max(g.shape[1], 1)
    g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return float(np.percentile(mag, 99) / (np.percentile(g, 99) - np.percentile(g, 1) + 10))


def _eye_sharpness(full, points, fw):
    """原图上两只眼睛分开量，取清楚的那只（只取眼睛不含眉毛：眉毛常被刘海挡住）。"""
    from scripts.face import _parts
    vals = []
    for part in ("left_eye", "right_eye"):
        q = points[_parts[part]]
        x0, y0 = q.min(axis=0) - 0.04 * fw
        x1, y1 = q.max(axis=0) + 0.04 * fw
        crop = full[max(int(y0), 0):int(y1), max(int(x0), 0):int(x1)]
        if crop.size:
            vals.append(_edge_steepness(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)))
    return max(vals) if vals else None


def _best_tile(gray, n=8):
    h, w = gray.shape
    vals = []
    for i in range(n):
        for j in range(n):
            t = gray[i * h // n:(i + 1) * h // n, j * w // n:(j + 1) * w // n]
            if t.std() > 6:
                vals.append(_edge_steepness(t))
    return float(np.percentile(vals, 95)) if vals else 0.0


def analyze_one(path):
    """单张照片的选片特征（不含 DINOv2，DINOv2 在主进程里批量算）。"""
    from scripts.face import detect_faces
    s = Shot(path, _exif_time(path))
    full = cv2.imread(path)
    if full is None:
        s.tags.append("读不出")
        return s
    k = min(1.0, 3000 / max(full.shape[:2]))
    im = cv2.resize(full, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else full
    faces = [f for f in detect_faces(im) if f.points is not None]
    if faces:
        f = max(faces, key=lambda q: float(np.linalg.norm(q.points[454] - q.points[234])))
        pts = f.points / k
        s.sharp = _eye_sharpness(full, pts, float(np.linalg.norm(pts[454] - pts[234])))
        s.has_face = True
        b = f.blend or {}
        s.blink = max(b.get("eyeBlinkLeft", 0), b.get("eyeBlinkRight", 0))
        s.look_down = (b.get("eyeLookDownLeft", 0) + b.get("eyeLookDownRight", 0)) / 2
        s.smile = (b.get("mouthSmileLeft", 0) + b.get("mouthSmileRight", 0)) / 2
    else:
        gray = cv2.cvtColor(cv2.resize(im, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        s.sharp = _best_tile(gray)
    h, w = im.shape[:2]
    r = 224 / max(h, w)
    s.thumb = cv2.resize(im, (max(14, int(w * r) // 14 * 14), max(14, int(h * r) // 14 * 14)), interpolation=cv2.INTER_AREA)
    return s


def embed(shots):
    """DINOv2-small CLS 向量（单位化）。缩略图尺寸各不相同（保持原比例），逐张算。"""
    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained("facebook/dinov2-small").eval()
    mean = np.float32([0.485, 0.456, 0.406])
    std = np.float32([0.229, 0.224, 0.225])
    for s in shots:
        if getattr(s, "thumb", None) is None:
            continue
        x = ((cv2.cvtColor(s.thumb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1)[None]
        with torch.no_grad():
            e = model(pixel_values=torch.from_numpy(x)).last_hidden_state[0, 0].numpy()
        s.emb = e / np.linalg.norm(e)


def group(shots):
    shots.sort(key=lambda s: s.name)
    g = -1
    prev, acc = None, None
    for s in shots:
        new = prev is None or s.emb is None or prev.emb is None
        if not new:
            dt = abs(s.time - prev.time) if (s.time is not None and prev.time is not None) else 0.0
            centroid = acc / np.linalg.norm(acc)
            new = dt > GAP_S or float(s.emb @ prev.emb) < SIM or float(s.emb @ centroid) < SIM_GROUP
        if new:
            g += 1
            acc = np.zeros_like(s.emb) if s.emb is not None else None
        if s.emb is not None and acc is not None:
            acc = acc + s.emb
        s.group = g
        prev = s
    return shots


def decide(shots):
    by = {}
    for s in shots:
        by.setdefault(s.group, []).append(s)
    for members in by.values():
        top = max((m.sharp or 0) for m in members) or 1e-6
        for m in members:
            if m.sharp is None:
                continue
            if (m.has_face and m.sharp < BLUR_ABS) or (len(members) > 1 and m.sharp < BLUR_REL * top):
                m.tags.append("模糊")
            if m.has_face and m.blink > BLINK:
                m.tags.append("垂眼" if m.look_down > LOOK_DOWN else "闭眼")
        score = lambda m: (m.sharp or 0) / top * (0.3 if "闭眼" in m.tags else 1.0) + 0.05 * m.smile
        best = max(members, key=score)
        best.best = True
        if len(members) > 1:
            for m in members:
                if m is not best:
                    m.tags.append("重复")
    return shots


def verdict(s):
    if "读不出" in s.tags or {"模糊", "闭眼"} & set(s.tags):
        return "废片"
    return "保留" if s.best else "重复"
