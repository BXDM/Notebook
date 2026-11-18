"""选片（批次层，先于单张精修）：主体模糊、闭眼、连拍重复分组（每组选一张），只标记不删除，召回优先。

每张照片：
- 主体清晰度：最大的那张脸，在**原图分辨率**上两只眼睛分开量，取清楚的那只（浅景深时远处那只眼虚是正常的）：
  每只眼缩到眼宽 160 像素（顺带平均掉噪点颗粒），最强 1% 边缘的梯度 / 眼部亮度范围。
  X-H2 176 张实测（原图核对标注）：模糊 0.31~0.70，清晰 0.78~1.36；拉普拉斯能量、细/粗尺度比都分不开
  （噪点被当成边缘，缩图量又丢了细节）。拖影（沿一个方向仍有陡边）靠同组对比抓。
  没有人脸时取全图最清晰的一块，只做组内对比（背景虚化不算模糊）。
  眼睛被挡住（墨镜、手、头发：眼部明显比脸颊暗且内部没层次）时改量鼻子到嘴一带（含镜框边缘），记“眼睛被遮挡”，
  不参与设备校正统计——R10 实测墨镜脸量眼睛只有 0.7~0.85，全被误判成模糊。鼻嘴一带的边缘天生比睫毛柔和，
  量级低：阈值取 HIDDEN_K × 正常阈值，组内也只和同样被遮挡的照片比（R10 核对：糊 0.28~0.38，软 0.39~0.49，清楚 ≥ 0.59）。
- 闭眼：MediaPipe 表情系数 eyeBlink，取闭得更紧的那只眼（单眼闭也算）。闭眼本身不当废片：
  X-H2 被标闭眼/垂眼的 11 张核对下来都是主动的姿势（垂眼、低头看东西、仰头半闭眼），而且 eyeBlink 对垂眼也给 0.5~0.7。
  区分靠连拍上下文：同组里别的帧睁着眼、只有这张闭着 → “眨眼”（拍摄问题，组内选睁眼的那张）；
  单张或整组都闭着 → “闭眼”提示，保留，留给人判断。
- 画面特征：DINOv2-small 的 CLS 向量（Apache-2.0，Hugging Face facebook/dinov2-small），用于连拍分组。
  缩略图保持原比例（长边 224，边长取 14 的倍数）：压成正方形时换场景的两张也能有 0.88 的相似度，分不开。
- 拖影：脸部（双眼、鼻、嘴角）按 8 个方向量边缘陡度，最弱 / 最强 < MOTION_ANISO 且清晰度不高时记“疑似拖影”
  （人脸本身横向结构多，方向比天然不均，只做保守一档；拖影主要靠同组对比抓）。
- 对焦失误：脸模糊、而画面别处最清楚的一块比眼部清楚 FOCUS_ELSEWHERE 倍以上（同一尺度：原图缩到 1/4 再量），
  且那一块本身 ≥ FOCUS_SHARP，记“焦点在别处”（只作原因说明，不影响是否废片）；否则是整张都糊（多半是手抖或失焦到很远）。拿整张脸比不公平：脸上大部分是平滑皮肤，本来就没边缘。
- 设备校正：同一机型的有脸照片 ≥ CALIB_MIN 张时，模糊阈值 = 清晰那一半的中位数 × CALIB_K（限制在 0.45~1.0；
  CALIB_K 由 X-H2 人工标定反推：阈值 0.72 / 清晰那一半中位 1.22 ≈ 0.6），
  不同相机、镜头、JPEG 设置的清晰度量级不同，用这批照片自己的分布定；不够就用 BLUR_ABS。
- 美学：LAION 美学评分（engine/batch/aesthetic.py），只在组内挑选时用。
分组：按文件名（相机编号连续）排序，与上一张拍摄间隔 ≤ GAP_S 秒、与上一张相似度 ≥ SIM、
且与本组平均特征的相似度 ≥ SIM_GROUP 才归入同一组。
组内：先留下清晰度 ≥ 组内最清晰 × PICK_SHARP、且不是眨眼的候选，再选美学分最高的（没有美学分时选最清晰的）；
其余为“重复”。
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
MOTION_ANISO = 0.55
FOCUS_ELSEWHERE = 1.5
FOCUS_SHARP = 2.0    # 别处那块本身也要真清楚：白底前的细叶对比强，糊了也有 1.7（0933），对焦到头饰上是 2.25（0776）
CALIB_MIN = 20
CALIB_K = 0.6
HIDDEN_K = 0.6
PICK_SHARP = 0.85


@dataclass
class Shot:
    path: str
    time: float = None
    sharp: float = None
    has_face: bool = False
    blink: float = 0.0
    look_down: float = 0.0
    smile: float = 0.0
    camera: str = ""
    aniso: float = None
    face_tile: float = None
    best_tile: float = None
    aes: float = None
    eyes_hidden: bool = False
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


def _box(full, pts, fw, pad):
    x0, y0 = pts.min(axis=0) - pad * fw
    x1, y1 = pts.max(axis=0) + pad * fw
    return full[max(int(y0), 0):int(y1), max(int(x0), 0):int(x1)]


def _eye_sharpness(full, points, fw):
    """原图上两只眼睛分开量，取清楚的那只（只取眼睛不含眉毛：眉毛常被刘海挡住）。
    返回 (清晰度, 眼睛是否被挡住)。眼睛被挡住时改量鼻子到嘴一带（按同一比例缩放：眼宽约 0.25 脸宽 → 160 像素）。"""
    from scripts.face import _parts
    cheek = _box(full, points[[50, 280, 205, 425]], fw, 0.03)
    cheek_l = float(cv2.cvtColor(cheek, cv2.COLOR_BGR2GRAY).mean()) if cheek.size else 128.0
    vals, hidden = [], 0
    for part in ("left_eye", "right_eye"):
        crop = _box(full, points[_parts[part]], fw, 0.04)
        if not crop.size:
            continue
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # 墨镜、手、头发挡住：眼部平均亮度不到脸颊的 55%，且内部亮度范围很小
        if g.mean() < 0.55 * cheek_l and np.percentile(g, 95) - np.percentile(g, 5) < 0.45 * cheek_l:
            hidden += 1
            continue
        vals.append(_edge_steepness(g))
    if vals:
        return max(vals), False
    crop = _box(full, points[[1, 4, 98, 327, 61, 291, 0, 17, 168]], fw, 0.04)
    if not crop.size:
        return None, True
    return _edge_steepness(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), width=max(40, int(crop.shape[1] * 640 / fw))), True


def _best_tile(gray, n=8):
    h, w = gray.shape
    vals = []
    for i in range(n):
        for j in range(n):
            t = gray[i * h // n:(i + 1) * h // n, j * w // n:(j + 1) * w // n]
            if t.std() > 6:
                vals.append(_edge_steepness(t))
    return float(np.percentile(vals, 95)) if vals else 0.0


def _anisotropy(full, points, fw):
    """脸部（双眼、鼻、嘴角）8 个方向的边缘陡度，最弱 / 最强。"""
    from scripts.face import _parts
    q = points[_parts["left_eye"] + _parts["right_eye"] + [1, 4, 61, 291]]
    x0, y0 = q.min(axis=0) - 0.05 * fw
    x1, y1 = q.max(axis=0) + 0.05 * fw
    crop = full[max(int(y0), 0):int(y1), max(int(x0), 0):int(x1)]
    if crop.size == 0:
        return None
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.resize(g, None, fx=300 / g.shape[1], fy=300 / g.shape[1], interpolation=cv2.INTER_AREA)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    v = [np.percentile(np.abs(gx * np.cos(a) + gy * np.sin(a)), 99) for a in np.linspace(0, np.pi, 8, endpoint=False)]
    return float(min(v) / max(max(v), 1e-6))


def analyze_one(path):
    """单张照片的选片特征（不含 DINOv2，DINOv2 在主进程里批量算）。"""
    from scripts.analyze import read_exif
    from scripts.face import detect_faces
    s = Shot(path, _exif_time(path))
    s.camera = str(read_exif(path).get("camera", ""))
    full = cv2.imread(path)
    if full is None:
        s.tags.append("读不出")
        return s
    k = min(1.0, 3000 / max(full.shape[:2]))
    im = cv2.resize(full, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else full
    faces = [f for f in detect_faces(im, extras=True) if f.points is not None]
    if faces:
        f = max(faces, key=lambda q: float(np.linalg.norm(q.points[454] - q.points[234])))
        pts = f.points / k
        fw = float(np.linalg.norm(pts[454] - pts[234]))
        s.sharp, s.eyes_hidden = _eye_sharpness(full, pts, fw)
        s.aniso = _anisotropy(full, pts, fw)
        s.has_face = True
        # 对焦失误：同一尺度下脸部那块与全图最清楚的一块比较（缩到 1/4 再分块）
        q4 = cv2.cvtColor(cv2.resize(full, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        from scripts.face import _parts
        e = pts[_parts["left_eye"] + _parts["right_eye"]] * 0.25
        m4 = 0.05 * fw * 0.25
        fx0, fy0 = (e.min(axis=0) - m4).astype(int)
        fx1, fy1 = (e.max(axis=0) + m4).astype(int)
        fc = q4[max(fy0, 0):fy1, max(fx0, 0):fx1]
        s.face_tile = _edge_steepness(fc) if fc.size else None
        s.best_tile = _best_tile(q4, 10)
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
    r2 = 448 / max(h, w)
    s.aes_img = cv2.resize(im, (int(w * r2), int(h * r2)), interpolation=cv2.INTER_AREA)
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


def calibrate(shots):
    """按机型定模糊阈值。返回 {机型: (阈值, 说明)}。"""
    out = {}
    for cam in {s.camera for s in shots}:
        v = np.array([s.sharp for s in shots if s.camera == cam and s.has_face and s.sharp is not None and not s.eyes_hidden])
        if len(v) >= CALIB_MIN:
            upper = v[v >= np.median(v)]
            thr = float(np.clip(CALIB_K * np.median(upper), 0.45, 1.0))
            out[cam] = (thr, f"{cam or '未知机型'}：{len(v)} 张有脸，清晰那一半中位 {np.median(upper):.2f} → 阈值 {thr:.2f}")
        else:
            out[cam] = (BLUR_ABS, f"{cam or '未知机型'}：有脸照片 {len(v)} 张 < {CALIB_MIN}，用默认阈值 {BLUR_ABS}")
    return out


def decide(shots, thresholds=None):
    thresholds = thresholds or {}
    by = {}
    for s in shots:
        by.setdefault(s.group, []).append(s)
    for members in by.values():
        tops = {h: max([(m.sharp or 0) for m in members if m.eyes_hidden == h] or [0]) or 1e-6 for h in (False, True)}
        for m in members:
            if m.sharp is None:
                continue
            thr = thresholds.get(m.camera, (BLUR_ABS, ""))[0] * (HIDDEN_K if m.eyes_hidden else 1.0)
            same = [o for o in members if o.eyes_hidden == m.eyes_hidden]
            if m.eyes_hidden:
                m.tags.append("眼睛被遮挡")
            if (m.has_face and m.sharp < thr) or (len(same) > 1 and m.sharp < BLUR_REL * tops[m.eyes_hidden]):
                m.tags.append("模糊")
                if (m.face_tile and m.best_tile and m.best_tile > FOCUS_ELSEWHERE * m.face_tile
                        and m.best_tile >= FOCUS_SHARP):
                    m.tags.append("焦点在别处")
            elif (m.has_face and not m.eyes_hidden and m.aniso is not None and m.aniso < MOTION_ANISO
                  and m.sharp < 1.2 * thr):      # 墨镜框的横线很强，方向比失真，被遮挡时不判拖影
                m.tags.append("疑似拖影")
            if m.has_face and m.blink > BLINK:
                opened = [o for o in members if o is not m and o.has_face and o.blink < 0.3]
                m.tags.append("眨眼" if opened else ("垂眼" if m.look_down > LOOK_DOWN else "闭眼"))
        cands = [m for m in members if (m.sharp or 0) >= PICK_SHARP * tops[m.eyes_hidden] and "眨眼" not in m.tags
                 and "模糊" not in m.tags] or members
        if all(m.aes is not None for m in cands):
            best = max(cands, key=lambda m: m.aes)
        else:
            best = max(cands, key=lambda m: (m.sharp or 0) + 0.05 * m.smile)
        best.best = True
        if len(members) > 1:
            for m in members:
                if m is not best:
                    m.tags.append("重复")
    return shots


def verdict(s):
    if "读不出" in s.tags or "模糊" in s.tags or "疑似拖影" in s.tags:
        return "废片"
    return "保留" if s.best else "重复"
