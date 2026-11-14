"""祛痘回归测试：在真实人像的干净皮肤上随机合成痘、色斑、雀斑、痣，统计检出与分类。

    python tools/blemish_bench.py 人像1.jpg [人像2.jpg ...] [--rounds 6]

只在检测器会检查的区域内撒点（眼周、眉间、嘴角、鼻侧、轮廓边缘不检测），点间距 > 脸宽/7。
输出“真值 -> 判定”混淆表，以及原图本身的检出（真实照片上的误判要人工看 --debug 的 blemishes.jpg）。
"""
import argparse
import os
import sys
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.blemish import _ellipse, remove_blemishes  # noqa: E402
from scripts.face import detect_faces  # noqa: E402
from scripts.preprocess import preprocess_image  # noqa: E402
from scripts.skin_mask import feature_scale, skin_mask  # noqa: E402

KINDS = ["acne"] * 3 + ["spot"] * 2 + ["freckle"] * 2 + ["mole"]


def add_spot(img, x, y, r, dL, da, db):
    """高斯形状的斑点：Lab(uint8) 中亮度降 dL、a 升 da、b 升 db。"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    pad = int(r * 3)
    y0, y1, x0, x1 = max(0, y - pad), y + pad, max(0, x - pad), x + pad
    yy, xx = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * (r / 1.6) ** 2))
    roi = lab[y0:y1, x0:x1]
    roi[..., 0] -= dL * g
    roi[..., 1] += da * g
    roi[..., 2] += db * g
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def detectable_region(img, faces, masks, scale):
    region = masks.face_hard.copy()
    for f in faces:
        if f.points is not None:
            region[f.part_mask(img.shape, ["left_eye", "right_eye", "lips"], grow=f.box[2] * 0.1) > 0] = 0
            region[f.under_eye_mask(img.shape) > 0.1] = 0
            region[f.part_mask(img.shape, ["nose"], grow=f.box[2] * 0.08) > 0] = 0
    return cv2.erode(region, _ellipse(scale / 7))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--rounds", type=int, default=6)
    args = ap.parse_args()
    conf, extra, total = Counter(), 0, 0
    for path in args.images:
        img = preprocess_image(path)
        faces = detect_faces(img)
        if not faces:
            print(f"{path}: 没检测到人脸，跳过")
            continue
        scale = feature_scale(img, faces)
        masks = skin_mask(img, faces)
        _, base = remove_blemishes(img, masks, faces, scale)
        print(f"{os.path.basename(path)} 原图检出: " + (", ".join(f"{b.kind}@({b.center[0]:.0f},{b.center[1]:.0f})"
                                                           for b in base) or "无"))
        ys, xs = np.nonzero(detectable_region(img, faces, masks, scale))
        if len(xs) == 0:
            continue
        params = {"acne": (scale / 60, 12, 18, 4), "spot": (scale / 50, 22, 3, 6),
                  "freckle": (scale / 110, 10, 2, 5), "mole": (scale / 45, 70, 2, 6)}
        for seed in range(args.rounds):
            rng = np.random.RandomState(seed)
            pts, tries = [], 0
            while len(pts) < len(KINDS) and tries < 5000:
                tries += 1
                i = rng.randint(len(xs))
                p = (int(xs[i]), int(ys[i]))
                if all(np.hypot(p[0] - q[0], p[1] - q[1]) > scale / 7 for q in pts):
                    pts.append(p)
            syn = img.copy()
            for k, p in zip(KINDS, pts):
                syn = add_spot(syn, p[0], p[1], *params[k])
            _, found = remove_blemishes(syn, skin_mask(syn, faces), faces, scale)
            used = set()
            for k, p in zip(KINDS, pts):
                near = [j for j, b in enumerate(found) if np.hypot(b.center[0] - p[0], b.center[1] - p[1]) < scale / 25]
                conf[(k, found[near[0]].kind if near else "miss")] += 1
                used.update(near)
                total += 1
            extra += sum(1 for j, b in enumerate(found) if j not in used and b.kind != "freckle")
    print("\n真值 -> 判定")
    for k in ["acne", "spot", "freckle", "mole"]:
        row = {g: conf[(k, g)] for g in ["acne", "spot", "freckle", "mole", "miss"] if conf[(k, g)]}
        n = sum(row.values())
        print(f"  {k:8s} 正确 {row.get(k, 0)}/{n}  {row}")
    print(f"非雀斑的额外检出（可能误修）: {extra}，合成斑点共 {total}")


if __name__ == "__main__":
    main()
