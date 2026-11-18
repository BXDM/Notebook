"""在一组照片上跑发缝填充（开发用，不改原图），输出头顶局部的前后对比图。

    python tools/hair_part_pass.py 目录或图片... --compare 对比图目录 [--strength 1] [--scan 统计.tsv] [--debug]
--scan 只做检测、把每张脸的判定写进 tsv（用来从大批素材里挑候选），不输出图。
对比图：左原图头顶局部 | 右处理后，100% 裁剪（区域过大时才缩小）；--debug 再加一栏头皮掩码。
隐私：自动跳过 2026-09-17 ~ 2026-10-10 日期目录。
"""
import argparse
import dataclasses
import glob
import os
import re
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.face import detect_faces  # noqa: E402
from scripts.hair_part import fill_hair_part  # noqa: E402

DETECT_LONG = 3000
EXTS = (".jpg", ".jpeg", ".png")


def private(path):
    m = re.search(r"(20\d\d)-(\d\d)-(\d\d)", path)
    return bool(m) and "2026-09-17" <= m.group(0) <= "2026-10-10"


def collect(srcs, every):
    files = []
    for s in srcs:
        if os.path.isdir(s):
            fs = sorted(f for f in glob.glob(os.path.join(s, "*")) if f.lower().endswith(EXTS))
            files += fs[::every]
        else:
            files.append(s)
    return [f for f in files if not private(f)]


def detect(im):
    """大图先缩到长边 3000 检测，再把关键点/框换算回原图坐标。"""
    s = min(1.0, DETECT_LONG / max(im.shape[:2]))
    small = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else im
    faces = detect_faces(small)
    if s == 1:
        return faces
    return [dataclasses.replace(f, box=tuple(int(v / s) for v in f.box), points=f.points / s,
                                depth=None if f.depth is None else f.depth / s) for f in faces]


def compare(im, out, info, debug):
    x0, y0, x1, y1 = info["box"]
    sc = info["score"]
    ys, xs = np.where(sc > 0.1)
    s = info["scale"]
    cx, cy = x0 + xs.mean() / s, y0 + ys.mean() / s
    ext = max(np.ptp(xs), np.ptp(ys)) / s * 1.4 + info["fw"] * 0.25
    half = int(max(250, min(ext / 2, 1200)))
    a0, b0 = int(max(cx - half, 0)), int(max(cy - half, 0))
    a1, b1 = int(min(cx + half, im.shape[1])), int(min(cy + half, im.shape[0]))
    panels = [im[b0:b1, a0:a1], out[b0:b1, a0:a1]]
    if debug:
        m = cv2.resize(sc, (x1 - x0, y1 - y0))
        full = np.zeros(im.shape[:2], np.float32)
        full[y0:y1, x0:x1] = m
        ov = im[b0:b1, a0:a1].copy()
        ov[..., 2] = np.clip(ov[..., 2] + full[b0:b1, a0:a1] * 200, 0, 255)
        panels.append(ov)
    side = max(p.shape[1] for p in panels)
    f = min(1.0, 800 / side)        # 每栏最多 800px；100% 优先
    panels = [cv2.resize(p, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else p for p in panels]
    gap = np.full((panels[0].shape[0], 10, 3), 255, np.uint8)
    row = [panels[0]]
    for p in panels[1:]:
        row += [gap, p]
    return np.hstack(row), f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="+")
    ap.add_argument("--compare", help="对比图输出目录")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--scan", help="只检测，写 tsv")
    ap.add_argument("--every", type=int, default=1, help="目录里每 N 张取一张")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    files = collect(a.src, a.every)
    tsv = open(a.scan, "w") if a.scan else None
    if tsv:
        tsv.write("file\tface\tstatus\treason\tarea\tcontrast\tkinds\tfw\n")
    n_do = n_skip = 0
    for path in files:
        im = cv2.imread(path)
        if im is None:
            continue
        t = time.time()
        faces = detect(im)
        infos = []
        if a.scan:
            from scripts.hair_part import analyze
            infos = [analyze(im, f) for f in faces]
            for i, info in enumerate(infos):
                tsv.write(f"{path}\t{i}\t{info['status']}\t{info.get('reason', '')}\t{info.get('area', 0):.4f}\t"
                          f"{info.get('contrast', 0):.1f}\t{','.join(c['kind'] for c in info.get('comps', []))}\t"
                          f"{info.get('fw', 0):.0f}\n")
            tsv.flush()
            print(f"{os.path.basename(path)}: " + "；".join(
                f"{x['status']} {x.get('reason', '')}" for x in infos), flush=True)
            continue
        out, notes = fill_hair_part(im, faces, a.strength, infos=infos)
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"{name}: " + "；".join(notes) + f"  ({time.time() - t:.1f}s)", flush=True)
        if out is im:
            n_skip += 1
            continue
        n_do += 1
        if a.compare:
            os.makedirs(a.compare, exist_ok=True)
            for i, info in enumerate(infos):
                if info["status"] != "do":
                    continue
                img, f = compare(im, out, info, a.debug)
                tag = "100%" if f == 1 else f"{f * 100:.0f}%"
                cv2.imwrite(os.path.join(a.compare, f"{name}_脸{i + 1}_{tag}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not a.scan:
        print(f"处理 {n_do} 张，跳过 {n_skip} 张")


if __name__ == "__main__":
    main()
