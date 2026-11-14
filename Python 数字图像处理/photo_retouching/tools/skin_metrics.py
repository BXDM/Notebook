"""皮肤质量指标：在脸部皮肤内统计三个尺度的亮度起伏（L*，越小越平滑）。

    python tools/skin_metrics.py 原图.jpg 结果1.jpg [结果2.jpg ...]

细节 = 毛孔/颗粒，中频 = 斑驳/痘印/细褶皱，大尺度 = 明暗不均。理想的精修：中频、大尺度明显下降，
细节只在有颗粒时下降、干净照片上基本不变（毛孔保留）。蒙版取自第一张图。
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.face import detect_faces  # noqa: E402
from scripts.preprocess import preprocess_image  # noqa: E402
from scripts.skin_mask import feature_scale, skin_mask  # noqa: E402


def bands(img, mask, scale):
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) / 2.55
    g1 = cv2.GaussianBlur(L, (0, 0), max(1.0, scale / 250))
    g2 = cv2.GaussianBlur(L, (0, 0), scale / 40)
    g3 = cv2.GaussianBlur(L, (0, 0), scale / 10)
    m = mask > 0
    return [float(x[m].std()) for x in (L - g1, g1 - g2, g2 - g3)]


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ref = preprocess_image(sys.argv[1])
    faces = detect_faces(ref)
    if not faces:
        sys.exit("第一张图没检测到人脸")
    scale = feature_scale(ref, faces)
    mask = cv2.erode(skin_mask(ref, faces).face_hard, np.ones((15, 15), np.uint8))
    print(f"{'':32s}{'细节':>8s}{'中频':>8s}{'大尺度':>8s}")
    for p in sys.argv[1:]:
        img = preprocess_image(p)
        if img.shape != ref.shape:
            print(f"{os.path.basename(p):32s}  尺寸与原图不同，跳过")
            continue
        print(f"{os.path.basename(p)[:30]:32s}" + "".join(f"{v:8.2f}" for v in bands(img, mask, scale)))


if __name__ == "__main__":
    main()
