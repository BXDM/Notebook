"""人脸检测：优先用 mediapipe FaceLandmarker（478 个关键点），不可用时退回 OpenCV Haar 人脸框。"""
import contextlib
import os
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

from scripts.filters import grow as grow_mask



@contextlib.contextmanager
def _quiet_stderr():
    """mediapipe / TFLite 的 C++ 层直接往 fd 2 写启动日志，临时重定向掉。"""
    fd = os.dup(2)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(fd, 2)
            os.close(fd)

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/latest/face_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "face_landmarker.task")

_HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_options = None         # FaceLandmarkerOptions，懒加载；False 表示不可用
_parts = {}             # 部位名 -> 关键点下标
# 两只眼的下眼睑关键点（从外眼角到内眼角）
_LOWER_LIDS = [[33, 7, 163, 144, 145, 153, 154, 155, 133], [263, 249, 390, 373, 374, 380, 381, 382, 362]]


@dataclass
class Face:
    box: tuple                  # (x, y, w, h)，有关键点时按 Haar 框的比例换算，便于各步共用参数
    points: np.ndarray = None   # (478, 2) 像素坐标；Haar 模式下为 None

    def hull(self, part):
        return cv2.convexHull(self.points[_parts[part]].astype(np.int32))

    def part_mask(self, shape, parts, grow=0):
        """把若干部位的凸包画成 0/255 掩码，grow 为膨胀像素。"""
        mask = np.zeros(shape[:2], np.uint8)
        for p in parts:
            cv2.fillConvexPoly(mask, self.hull(p), 255)
        if grow > 0:
            mask = grow_mask(mask, int(grow) * 2 + 1)
        return mask

    def under_eye_mask(self, shape):
        """眼袋/黑眼圈区域：下眼睑往下约 0.1 倍脸宽的带状区，0~1 float32（已羽化）。"""
        pts = self.points
        face_w = np.linalg.norm(pts[454] - pts[234])
        eye_line = pts[263] - pts[33]
        down = np.array([-eye_line[1], eye_line[0]], np.float32)
        down /= np.linalg.norm(down) + 1e-6
        if np.dot(down, pts[152] - pts[33]) < 0:  # 让方向指向下巴
            down = -down
        mask = np.zeros(shape[:2], np.uint8)
        for lid in _LOWER_LIDS:
            p = pts[lid]
            band = np.vstack([p + down * face_w * 0.015, p[::-1] + down * face_w * 0.11])
            cv2.fillPoly(mask, [band.astype(np.int32)], 255)
        return cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), face_w / 40)

    def feature_mask(self, shape, grow=0):
        """眉、眼、嘴（不含鼻）。"""
        return self.part_mask(shape, ["left_eye", "right_eye", "left_brow", "right_brow", "lips"], grow)


def _load_options():
    global _options
    if _options is not None:
        return _options
    try:
        with _quiet_stderr():
            import mediapipe as mp  # noqa: F401
            from mediapipe.tasks.python import BaseOptions, vision
    except ImportError:
        print("[face] 未安装 mediapipe，使用 Haar 人脸框（瘦脸、五官排除精度较低）")
        _options = False
        return _options
    if not os.path.exists(MODEL_PATH):
        try:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            print(f"[face] 下载关键点模型 -> {MODEL_PATH}")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH + ".part")
            os.replace(MODEL_PATH + ".part", MODEL_PATH)
        except Exception as e:  # 网络不通时退回 Haar
            print(f"[face] 模型下载失败（{e}），使用 Haar 人脸框")
            _options = False
            return _options

    conn = vision.FaceLandmarksConnections
    for name, edges in {
        "oval": conn.FACE_LANDMARKS_FACE_OVAL,
        "left_eye": conn.FACE_LANDMARKS_LEFT_EYE, "right_eye": conn.FACE_LANDMARKS_RIGHT_EYE,
        "left_brow": conn.FACE_LANDMARKS_LEFT_EYEBROW, "right_brow": conn.FACE_LANDMARKS_RIGHT_EYEBROW,
        "lips": conn.FACE_LANDMARKS_LIPS, "nose": conn.FACE_LANDMARKS_NOSE,
    }.items():
        _parts[name] = sorted({i for e in edges for i in (e.start, e.end)})

    _options = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL_PATH), num_faces=10)
    return _options


def _haar_faces(image):
    gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    min_size = max(24, min(image.shape[:2]) // 12)
    faces = _HAAR.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_size, min_size))
    return [Face(tuple(int(v) for v in f)) for f in faces]


def _tiles(h, w):
    """整图 + 边长为短边 3/4 的方块（步长半块）。mediapipe 先把输入缩到很小再找脸，
    广角环境人像里脸只占画面 6%~10%，整图检测不到，裁成小块后脸相对变大才能检出。"""
    tiles = [(0, 0, w, h)]
    if max(h, w) <= 2000:
        return tiles
    t = int(min(h, w) * 0.75)
    step = t // 2
    xs = sorted({*range(0, w - t, step), w - t})
    ys = sorted({*range(0, h - t, step), h - t})
    return tiles + [(x, y, t, t) for y in ys for x in xs]


def _iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-6)


def detect_faces(image, use_landmarks=True):
    """返回 [Face, ...]。大图分块检测再合并，块边缘被切到的脸丢弃（相邻块里有完整的）。"""
    options = _load_options() if use_landmarks else False
    if not options:
        return _haar_faces(image)

    import mediapipe as mp
    from mediapipe.tasks.python import vision
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = []
    # 每次用 with 创建并关闭；常驻实例在解释器退出时 close() 会报错
    with _quiet_stderr(), vision.FaceLandmarker.create_from_options(options) as landmarker:
        for tx, ty, tw, th in _tiles(h, w):
            tile = np.ascontiguousarray(rgb[ty:ty + th, tx:tx + tw])
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=tile))
            for lms in result.face_landmarks:
                pts = np.array([(tx + p.x * tw, ty + p.y * th) for p in lms], np.float32)
                oval = pts[_parts["oval"]]
                x0, y0 = oval.min(axis=0)
                x1, y1 = oval.max(axis=0)
                cut = ((x0 < tx + 2 and tx > 0) or (y0 < ty + 2 and ty > 0)
                       or (x1 > tx + tw - 2 and tx + tw < w) or (y1 > ty + th - 2 and ty + th < h))
                if cut:
                    continue
                # 换算成与 Haar 框大致相同的框（Haar 框比脸部轮廓略宽、上沿在额头中部）
                bw = (x1 - x0) * 1.2
                cx = (x0 + x1) / 2
                by = y0 + (y1 - y0) * 0.08
                box = (int(cx - bw / 2), int(by), int(bw), int(bw))
                if all(_iou(box, f.box) < 0.3 for f in faces):
                    faces.append(Face(box, pts))
    return faces


def nostril_parts_mask(face, shape, grow=0, image=None):
    """鼻孔保护区。鼻尖以下的鼻子关键点圈出下半个鼻子；给了 image 时只保留其中明显比鼻子皮肤暗的像素
    （鼻孔开口、鼻翼沟），鼻翼和鼻小柱的皮肤照常磨皮、祛痘。"""
    nose = face.points[_parts["nose"]]
    tip_y = face.points[4][1]
    low = nose[nose[:, 1] >= tip_y - (face.box[3] * 0.03)]
    mask = np.zeros(shape[:2], np.uint8)
    if len(low) >= 3:
        cv2.fillConvexPoly(mask, cv2.convexHull(low.astype(np.int32)), 255)
    if image is not None and mask.any():
        L = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[..., 0]
        whole = np.zeros(shape[:2], np.uint8)
        cv2.fillConvexPoly(whole, cv2.convexHull(nose.astype(np.int32)), 255)
        ref = float(np.median(L[whole > 0]))
        dark = ((L < ref * 0.7) & (mask > 0)).astype(np.uint8) * 255
        k = max(3, face.box[2] // 100) | 1
        mask = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    if grow > 0:
        mask = grow_mask(mask, int(grow) * 2 + 1)
    return mask
