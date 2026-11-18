"""分析层：一张照片的共享分析结果。每项只在第一次用到时计算，之后复用；
模块写回图像时声明让哪些分析失效（曝光改亮度 → 皮肤蒙版重算，但场景类型不变；拉直、裁剪 → 全部重算）。"""
from dataclasses import dataclass

import numpy as np

from scripts.analyze import ev100, read_exif


@dataclass
class Person:
    """人物区域：皮肤类步骤只在这块裁剪上做再贴回。蒙版在曝光、去眩光之后的图上算。"""
    roi: tuple            # (x0, y0, x1, y1)
    faces: list           # 坐标相对于 roi
    scale: float          # 各步骤半径的参考尺寸（脸宽）
    hands: object         # 手部蒙版（人像配置才有）
    masks: object         # skin_mask 的结果（roi 尺寸）

    def crop(self, image):
        x0, y0, x1, y1 = self.roi
        return image[y0:y1, x0:x1]

    def paste(self, image, crop):
        x0, y0, x1, y1 = self.roi
        out = image.copy()
        out[y0:y1, x0:x1] = crop
        return out

    def full(self, arr, shape):
        """roi 尺寸的蒙版放回整图尺寸。"""
        x0, y0, x1, y1 = self.roi
        big = np.zeros(shape[:2], arr.dtype)
        big[y0:y1, x0:x1] = arr
        return big


class PhotoContext:
    def __init__(self, image, path=None, opts=None):
        self.path = path
        self.original = image
        self.image = image
        self.opts = opts
        self.exif = read_exif(path) if path else {}
        self.ev100 = ev100(self.exif) if path else None
        self.profile = "portrait"
        self.stages = {"original": image}
        self.reports = []
        self._cache = {}

    @property
    def iso(self):
        return self.exif.get("iso")

    # ---- 分析项（惰性） ----
    @property
    def faces(self):
        if "faces" not in self._cache:
            from scripts.face import detect_faces
            self._cache["faces"] = detect_faces(self.image, use_landmarks=not getattr(self.opts, "no_landmarks", False))
        return self._cache["faces"]

    @property
    def scale(self):
        if "scale" not in self._cache:
            from scripts.skin_mask import feature_scale
            self._cache["scale"] = feature_scale(self.image, self.faces)
        return self._cache["scale"]

    @property
    def scene(self):
        """场景类型（夜景、晚霞……）。亮度调整不改变场景类型，只有几何改动才重算。"""
        if "scene" not in self._cache:
            from scripts.scene import classify
            self._cache["scene"] = classify(self.image, self.ev100)
        return self._cache["scene"]

    @property
    def person(self):
        """人物区域与皮肤蒙版；在当前图像上算（所以要在曝光、去眩光之后第一次用）。"""
        if "person" not in self._cache:
            from main import _shift_face, person_roi
            from scripts.hands import hand_mask
            from scripts.skin_mask import skin_mask
            faces, image = self.faces, self.image
            h, w = image.shape[:2]
            roi = person_roi(image.shape, faces) if faces else (0, 0, w, h)
            x0, y0, x1, y1 = roi
            sub = [_shift_face(f, x0, y0) for f in faces]
            crop = image[y0:y1, x0:x1]
            hands = hand_mask(crop)[0] if (self.profile != "r10" and faces) else None
            masks = skin_mask(crop, sub, use_parsing=not self.opts.portrait.no_parsing, hand=hands)
            self._cache["person"] = Person(roi, sub, self.scale, hands, masks)
        return self._cache["person"]

    def put(self, key, value):
        self._cache[key] = value

    def get(self, key, default=None):
        return self._cache.get(key, default)

    # ---- 图像更新 ----
    def update(self, image, stage, invalidates=()):
        """模块写回结果。invalidates：让哪些分析失效（"all" = 全部，如拉直、裁剪）。"""
        self.image = image
        if stage:
            self.stages[stage] = image
        keys = list(self._cache) if invalidates == "all" else invalidates
        for k in keys:
            self._cache.pop(k, None)

