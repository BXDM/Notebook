"""人脸检测：优先用 mediapipe FaceLandmarker（478 个关键点），不可用时退回 OpenCV Haar 人脸框。"""
import contextlib
import os
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np



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
            k = int(grow) * 2 + 1
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        return mask

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


def detect_faces(image, use_landmarks=True):
    """返回 [Face, ...]。"""
    options = _load_options() if use_landmarks else False
    if not options:
        return _haar_faces(image)

    import mediapipe as mp
    from mediapipe.tasks.python import vision
    h, w = image.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # 每次用 with 创建并关闭；常驻实例在解释器退出时 close() 会报错
    with _quiet_stderr(), vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    faces = []
    for lms in result.face_landmarks:
        pts = np.array([(p.x * w, p.y * h) for p in lms], np.float32)
        oval = pts[_parts["oval"]]
        x0, y0 = oval.min(axis=0)
        x1, y1 = oval.max(axis=0)
        # 换算成与 Haar 框大致相同的框（Haar 框比脸部轮廓略宽、上沿在额头中部）
        bw = (x1 - x0) * 1.2
        cx = (x0 + x1) / 2
        by = y0 + (y1 - y0) * 0.08
        faces.append(Face((int(cx - bw / 2), int(by), int(bw), int(bw)), pts))
    return faces


def nostril_parts_mask(face, shape, grow=0):
    """鼻翼和鼻孔区域：鼻子关键点中位于鼻尖以下的部分。"""
    nose = face.points[_parts["nose"]]
    tip_y = face.points[4][1]
    low = nose[nose[:, 1] >= tip_y - (face.box[3] * 0.03)]
    mask = np.zeros(shape[:2], np.uint8)
    if len(low) >= 3:
        cv2.fillConvexPoly(mask, cv2.convexHull(low.astype(np.int32)), 255)
    if grow > 0:
        k = int(grow) * 2 + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return mask
