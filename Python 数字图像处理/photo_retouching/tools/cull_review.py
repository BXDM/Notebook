"""把选片结果的所有组拼成若干审阅页（每页若干组，组与组之间有分隔条和组号），便于逐组核对。

    python tools/cull_review.py 选片报告目录 照片目录 [--per-page 8]

输出 报告目录/审阅/页NN.jpg。每张缩略图：绿框保留、红框废片、灰框重复，* 组内最佳，下方是标记和美学分。
"""
import argparse
import csv
import os

import cv2
import numpy as np

COL = {"保留": (60, 200, 60), "废片": (40, 40, 230), "重复": (150, 150, 150)}
EN = {"模糊": "blur", "闭眼": "eyes?", "眨眼": "blink", "垂眼": "down", "重复": "dup", "焦点在别处": "misfocus", "疑似拖影": "motion"}


def thumb(path, size=200):
    im = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_8)
    t = np.zeros((size, size, 3), np.uint8)
    if im is not None:
        h, w = im.shape[:2]
        r = size / max(h, w)
        im = cv2.resize(im, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)
        y, x = (size - im.shape[0]) // 2, (size - im.shape[1]) // 2
        t[y:y + im.shape[0], x:x + im.shape[1]] = im
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("photos")
    ap.add_argument("--per-page", type=int, default=8)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(a.report, "选片.csv"), encoding="utf-8-sig")))
    groups = {}
    for r in rows:
        groups.setdefault(int(r["组"]), []).append(r)
    out = os.path.join(a.report, "审阅")
    os.makedirs(out, exist_ok=True)
    strips = []
    for g in sorted(groups):
        tiles = []
        for r in groups[g]:
            t = thumb(os.path.join(a.photos, r["文件"]))
            v = r["判定"]
            t = cv2.copyMakeBorder(t, 5, 30, 5, 5, cv2.BORDER_CONSTANT, value=COL["保留" if v.startswith("保留") else v])
            name = os.path.splitext(r["文件"])[0][-4:]
            cv2.putText(t, ("*" if r["组内最佳"] else "") + name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            tags = ",".join(EN.get(x, x) for x in r["原因"].split("、") if x and x != "重复")
            aes = f" a{float(r['美学分']):.1f}" if r.get("美学分") else ""
            cv2.putText(t, (tags + aes).strip(), (6, t.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            tiles.append(t)
        strip = np.hstack(tiles[:10])
        head = np.full((26, strip.shape[1], 3), 40, np.uint8)
        cv2.putText(head, f"G{g}  ({len(groups[g])})", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        strips.append(np.vstack([head, strip]))
    width = max(s.shape[1] for s in strips)
    strips = [np.pad(s, ((0, 6), (0, width - s.shape[1]), (0, 0)), constant_values=255) for s in strips]
    for i in range(0, len(strips), a.per_page):
        cv2.imwrite(os.path.join(out, f"页{i // a.per_page + 1:02d}.jpg"), np.vstack(strips[i:i + a.per_page]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{len(groups)} 组，{(len(strips) + a.per_page - 1) // a.per_page} 页 -> {out}")


if __name__ == "__main__":
    main()
