"""瘦脸阈值标定：扫一批照片里的人脸，输出丰满度分布、各过滤条件命中数，拼出几组脸供肉眼核对。

    python tools/slim_calib.py 照片目录 [更多目录...] --out 结果目录

结果目录：faces.tsv（每张脸的度量）、grid_top.jpg（丰满度最高且通过过滤）、grid_edge.jpg（阈值附近）、
grid_filtered.jpg（丰满度高但被过滤掉的，检查过滤是否太严）、grid_median.jpg（中位数附近，作参照）。
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def one(f):
    from scripts.face import detect_faces
    from scripts.slim import MAX_PITCH, MAX_YAW, MIN_FACE_PX, WE_RANGE, _face_like, face_shape_metrics, is_child
    im = cv2.imread(f)
    if im is None:
        return []
    rows = []
    for fc in detect_faces(im):
        s = face_shape_metrics(fc)
        if s is None:
            continue
        ok, why = _face_like(im, fc)
        b = fc.blend or {}
        expr = (b.get("cheekPuff", 0) > 0.3 or b.get("mouthPucker", 0) > 0.6 or b.get("jawOpen", 0) > 0.4
                or abs(b.get("mouthSmileLeft", 0) - b.get("mouthSmileRight", 0)) > 0.3)
        status = ("小" if s["W2d"] < MIN_FACE_PX else "侧" if abs(s["yaw"]) > MAX_YAW or abs(s["pitch"]) > MAX_PITCH else
                  "非脸" if not ok else "表情" if expr else "儿童" if is_child(im, fc) else "通过")
        x0, y0 = fc.points.min(axis=0)
        x1, y1 = fc.points.max(axis=0)
        rows.append((f, int(x0), int(y0), int(x1), int(y1), s["width_ratio"], s["yaw"], s["W2d"], status,
                     float(np.clip((s["width_ratio"] - WE_RANGE[0]) / (WE_RANGE[1] - WE_RANGE[0]), 0, 1))))
    return rows


def grid(rows, path, title_fn, n=18, cols=6, size=260):
    tiles = []
    for r in rows[:n]:
        im = cv2.imread(r[0])
        x0, y0, x1, y1 = r[1:5]
        w, h = x1 - x0, y1 - y0
        m = int(max(w, h) * 0.35)
        c = im[max(y0 - m, 0):y1 + m, max(x0 - m, 0):x1 + m]
        c = cv2.resize(c, (size, int(size * 1.15)))
        cv2.putText(c, title_fn(r), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        tiles.append(c)
    if not tiles:
        return
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(path, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]),
                [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = [f for d in a.dirs for f in sorted(glob.glob(os.path.join(d, "*.[jJ][pP][gG]")))]
    with Pool(6) as p:
        rows = [r for rs in p.imap_unordered(one, files, chunksize=4) for r in rs]
    with open(os.path.join(a.out, "faces.tsv"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    from scripts.slim import WE_RANGE
    big = [r for r in rows if r[8] in ("通过", "非脸", "表情", "儿童")]
    we = np.array([r[5] for r in big])
    print(f"照片 {len(files)}，人脸 {len(rows)}；状态：" +
          "、".join(f"{k} {sum(r[8] == k for r in rows)}" for k in ("通过", "儿童", "表情", "非脸", "侧", "小")))
    if len(we):
        print("丰满度（够大、不太侧）分位 10/50/90/95/97：", np.percentile(we, [10, 50, 90, 95, 97]).round(3))
    ok = sorted([r for r in rows if r[8] == "通过"], key=lambda r: -r[5])
    trig = [r for r in ok if r[5] >= WE_RANGE[0]]
    print(f"通过过滤 {len(ok)} 张，会瘦 {len(trig)} 张（{len(trig) / max(len(ok), 1):.0%}）")
    t = lambda r: f"{r[5]:.2f} {r[6]:+.0f}d {r[8]}"
    grid(ok, os.path.join(a.out, "grid_top.jpg"), t)
    edge = sorted(ok, key=lambda r: abs(r[5] - WE_RANGE[0]))
    grid(edge, os.path.join(a.out, "grid_edge.jpg"), t)
    grid(sorted([r for r in rows if r[8] in ("非脸", "表情", "儿童")], key=lambda r: -r[5]),
         os.path.join(a.out, "grid_filtered.jpg"), t)
    med = float(np.median(we)) if len(we) else 1.56
    grid(sorted(ok, key=lambda r: abs(r[5] - med)), os.path.join(a.out, "grid_median.jpg"), t)


if __name__ == "__main__":
    main()
