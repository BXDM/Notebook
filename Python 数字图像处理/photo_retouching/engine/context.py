"""分析层：一张照片的共享分析结果。每项只在第一次用到时计算，之后复用；
模块改了图像后按改动类型让相应的分析失效（改像素值：皮肤蒙版、曝光统计重算；改几何：全部重算）。"""
from scripts.analyze import ev100, read_exif

# 分析项 → 依赖的图像属性。"pixels" 类在像素值改动后失效，"geometry" 类只在几何改动（拉直、裁剪）后失效
_KIND = {"faces": "geometry", "scale": "geometry", "roi": "geometry",
         "hands": "pixels", "masks": "pixels", "analysis": "pixels", "scene": "pixels"}


class PhotoContext:
    def __init__(self, image, path=None, opts=None):
        self.path = path
        self.original = image
        self.image = image
        self.opts = opts
        self.exif = read_exif(path) if path else {}
        self.ev100 = ev100(self.exif) if path else None
        self.stages = {"original": image}
        self.reports = []
        self._cache = {}

    # ---- 分析项 ----
    @property
    def faces(self):
        if "faces" not in self._cache:
            from scripts.face import detect_faces
            self._cache["faces"] = detect_faces(self.image, use_landmarks=not getattr(self.opts, "no_landmarks", False))
        return self._cache["faces"]

    @property
    def iso(self):
        return self.exif.get("iso")

    @property
    def camera(self):
        return str(self.exif.get("camera", ""))

    def put(self, key, value):
        """模块把自己算出的分析结果（如人像模块的皮肤蒙版）放进来给后面的模块用。"""
        self._cache[key] = value

    def get(self, key, default=None):
        return self._cache.get(key, default)

    # ---- 图像更新 ----
    def update(self, image, stage, geometry=False):
        """模块写回结果。geometry=True（拉直、裁剪、变形）时所有分析失效，否则只让像素类分析失效。"""
        self.image = image
        self.stages[stage] = image
        for k in list(self._cache):
            if geometry or _KIND.get(k, "pixels") == "pixels":
                del self._cache[k]
