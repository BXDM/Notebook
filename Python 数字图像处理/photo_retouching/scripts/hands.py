"""手部分区：mediapipe HandLandmarker（每只手 21 个关键点）→ 手的蒙版。

贴脸、托腮的手会通过脖子连进“身体皮肤”被当成脸一样磨皮，或被人脸分割当成头发/五官保护起来完全不处理，
两种都让手和脸的质感、肤色不一致。这里单独画出手：掌心用多边形（0,1,5,9,13,17），每根手指按关节连线画成
有粗细的线段——不用整只手的凸包，否则手指之间露出的脸也会被圈进来。粗细按手掌大小（手腕到中指根）定。
"""
import os
import urllib.request

import cv2
import numpy as np

from scripts.face import _quiet_stderr

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "hand_landmarker.task")
PALM = [0, 1, 5, 9, 13, 17]
FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]


def _options():
    try:
        with _quiet_stderr():
            from mediapipe.tasks.python import BaseOptions, vision
    except ImportError:
        return None
    if not os.path.exists(MODEL_PATH):
        try:
            print(f"[hands] 下载手部关键点模型 -> {MODEL_PATH}")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH + ".part")
            os.replace(MODEL_PATH + ".part", MODEL_PATH)
        except Exception as e:
            print(f"[hands] 模型下载失败（{e}），不做手部分区")
            return None
    return vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL_PATH), num_hands=4,
                                        min_hand_detection_confidence=0.4)


def hand_mask(image, work_side=1600):
    """返回 (0/255 uint8 手部蒙版, 手的数量)；没有模型或没检测到返回 (None, 0)。"""
    opts = _options()
    if opts is None:
        return None, 0
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    h, w = image.shape[:2]
    s = min(1.0, work_side / max(h, w))
    small = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    with _quiet_stderr(), vision.HandLandmarker.create_from_options(opts) as det:
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.hand_landmarks:
        return None, 0
    mask = np.zeros((h, w), np.uint8)
    for lms in res.hand_landmarks:
        p = np.array([(q.x * w, q.y * h) for q in lms], np.float32)
        palm = float(np.linalg.norm(p[9] - p[0]))           # 手腕到中指根
        cv2.fillPoly(mask, [cv2.convexHull(p[PALM].astype(np.int32))], 255)
        t = max(3, int(palm * 0.32))                          # 手指粗细
        for f in FINGERS:
            for a, b in zip(f[:-1], f[1:]):
                cv2.line(mask, tuple(p[a].astype(int)), tuple(p[b].astype(int)), 255, t, cv2.LINE_AA)
            cv2.circle(mask, tuple(p[f[-1]].astype(int)), t // 2, 255, -1)   # 指尖圆头
        cv2.line(mask, tuple(p[0].astype(int)), tuple(p[1].astype(int)), 255, t)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return mask, len(res.hand_landmarks)
