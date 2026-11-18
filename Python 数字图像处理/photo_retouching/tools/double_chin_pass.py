"""在一个目录（或若干文件）的照片上只跑去双下巴，输出下巴到脖子的局部前后对比图（开发用）。

    python tools/double_chin_pass.py 目录或文件... --compare 对比图目录 [--dst 结果目录] [--strength 1]
大图先缩到长边 3000 做人脸检测，关键点换算回原图，处理在原图分辨率上做。
没触发的照片不输出（只在日志里写原因）。对比图：原图 | 处理后，下巴到脖子局部，宽 1200。
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.double_chin import reduce_double_chin  # noqa: E402
from scripts.face import Face, detect_faces  # noqa: E402

DETECT_LONG = 3000


def faces_full_res(image):
    """缩到长边 DETECT_LONG 检测，把关键点换算回原图坐标。"""
    s = min(1.0, DETECT_LONG / max(image.shape[:2]))
    small = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else image
    out = []
    for f in detect_faces(small, extras=True):
        if f.points is None:
            continue
        box = tuple(int(v / s) for v in f.box)
        out.append(Face(box, f.points / s, None if f.depth is None else f.depth / s, f.pose, f.blend))
    return out


def chin_crop(face, shape):
    """下巴到脖子的局部：嘴角高度到下巴下 0.6 脸宽，左右各 0.65 脸宽。"""
    p = face.points
    fw = float(np.linalg.norm(p[454] - p[234]))
    c = p[152]
    x0, x1 = int(max(c[0] - 0.65 * fw, 0)), int(min(c[0] + 0.65 * fw, shape[1]))
    y0, y1 = int(max(p[13][1], 0)), int(min(c[1] + 0.6 * fw, shape[0]))
    return x0, y0, x1, y1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="+")
    ap.add_argument("--compare", required=True, help="对比图目录")
    ap.add_argument("--dst", help="另存处理后的整张图")
    ap.add_argument("--strength", type=float, default=1.0)
    a = ap.parse_args()
    files = []
    for s in a.src:
        files += sorted(glob.glob(os.path.join(s, "*.[jJ][pP][gG]"))) if os.path.isdir(s) else [s]
    os.makedirs(a.compare, exist_ok=True)
    done = 0
    for f in files:
        im = cv2.imread(f)
        if im is None:
            continue
        faces = faces_full_res(im)
        out, notes = reduce_double_chin(im, faces, a.strength)
        n = os.path.splitext(os.path.basename(f))[0]
        print(f"{n}: " + ("；".join(notes) or "没有人脸"), flush=True)
        if out is im:
            continue
        done += 1
        if a.dst:
            os.makedirs(a.dst, exist_ok=True)
            cv2.imwrite(os.path.join(a.dst, n + ".jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        for k, fc in enumerate(faces):
            x0, y0, x1, y1 = chin_crop(fc, im.shape)
            if not np.any(out[y0:y1, x0:x1] != im[y0:y1, x0:x1]):
                continue
            sc = 600 / (x1 - x0)
            pair = [cv2.resize(c[y0:y1, x0:x1], None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) for c in (im, out)]
            sep = np.full((pair[0].shape[0], 8, 3), 255, np.uint8)
            cv2.imwrite(os.path.join(a.compare, f"{n}_{k}.jpg"), np.hstack([pair[0], sep, pair[1]]),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"共 {len(files)} 张，处理 {done} 张")


if __name__ == "__main__":
    main()
