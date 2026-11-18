"""在已有成片上只做自适应瘦脸（开发用：瘦脸是几何操作，与降噪、磨皮无关，不必整条流程重跑）。

    python tools/slim_pass.py 成片目录 输出目录 [--strength 1] [--compare 对比图目录]
没触发瘦脸的照片不输出（只在日志里写原因）。
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.face import detect_faces  # noqa: E402
from scripts.slim import need, slim  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--compare", help="另存左右对比图（脸部放大）的目录")
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(a.src, "*.[jJ][pP][gG]"))):
        im = cv2.imread(f)
        faces = detect_faces(im)
        out, notes = slim(im, faces, a.strength)
        n = os.path.basename(f)
        print(f"{n}: " + ("；".join(notes) or "没有人脸"), flush=True)
        if out is im:
            continue
        cv2.imwrite(os.path.join(a.dst, n), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if a.compare:
            os.makedirs(a.compare, exist_ok=True)
            f0 = max(faces, key=lambda q: need(q, im, a.strength)[0])     # 裁被瘦得最多的那张脸
            x, y, w, _ = f0.box
            x0, y0 = max(int(x - 0.3 * w), 0), max(int(y - 0.2 * w), 0)
            x1, y1 = min(int(x + 1.3 * w), im.shape[1]), min(int(y + 1.3 * w), im.shape[0])
            pair = [cv2.resize(c[y0:y1, x0:x1], None, fx=900 / (x1 - x0), fy=900 / (x1 - x0)) for c in (im, out)]
            cv2.imwrite(os.path.join(a.compare, n), np.hstack([pair[0], np.full((pair[0].shape[0], 12, 3), 255, np.uint8), pair[1]]),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
