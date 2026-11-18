"""回归对比：新版本输出 vs 基准输出，逐张给出差异，列出变化明显的照片。

    python tools/regress.py 基准目录 新输出目录 [--thr 1.0]

按文件名主干配对（忽略 _auto 等后缀和扩展名大小写）。尺寸不同（拉直、裁剪不同）时把新图缩放到基准尺寸再比。
指标：平均绝对差（8-bit）、差 > 8 的像素占比、Lab 平均 ΔL / Δa / Δb（看整体是亮了还是偏色了）。
GPU 降噪每次运行有细微随机差异，平均差 < 0.5 视为一致。
"""
import argparse
import glob
import os
import re

import cv2
import numpy as np


def stem(f):
    return re.sub(r"_(auto|retouched)$", "", os.path.splitext(os.path.basename(f))[0]).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--thr", type=float, default=1.0, help="平均差超过多少算“变化明显”（默认 1.0）")
    a = ap.parse_args()
    exts = ("*.jpg", "*.JPG", "*.jpeg", "*.png")
    base = {stem(f): f for e in exts for f in glob.glob(os.path.join(a.base, e))}
    new = {stem(f): f for e in exts for f in glob.glob(os.path.join(a.new, e))}
    same, changed, missing = 0, [], sorted(set(base) - set(new))
    for k in sorted(set(base) & set(new)):
        x, y = cv2.imread(base[k]), cv2.imread(new[k])
        resized = x.shape != y.shape
        if resized:
            y = cv2.resize(y, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_AREA)
        d = np.abs(x.astype(np.int16) - y).astype(np.float32)
        mad, big = float(d.mean()), float((d.max(axis=2) > 8).mean())
        lx = cv2.cvtColor(x[::4, ::4].astype(np.float32) / 255, cv2.COLOR_BGR2Lab).reshape(-1, 3).mean(0)
        ly = cv2.cvtColor(y[::4, ::4].astype(np.float32) / 255, cv2.COLOR_BGR2Lab).reshape(-1, 3).mean(0)
        dl = ly - lx
        if mad < 0.5:
            same += 1
        if mad >= a.thr or resized:
            changed.append((k, mad, big, dl, resized))
    print(f"共 {len(set(base) & set(new))} 张：一致（平均差<0.5）{same} 张，变化明显 {len(changed)} 张"
          + (f"，缺输出 {len(missing)} 张：{'、'.join(missing)}" if missing else ""))
    for k, mad, big, dl, resized in sorted(changed, key=lambda t: -t[1]):
        print(f"  {k:32s} 平均差 {mad:5.2f}  差>8 {big:6.1%}  ΔL {dl[0]:+5.1f} Δa {dl[1]:+5.1f} Δb {dl[2]:+5.1f}"
              + ("  （尺寸不同）" if resized else ""))


if __name__ == "__main__":
    main()
