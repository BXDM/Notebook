"""瘦脸盲评：每张照片的主脸生成 不瘦 / 轻度（上限一半）/ 中度（完整上限）三版，打乱成 A/B/C 并排，答案另存。

    python tools/slim_eval.py 输出目录 --dir 照片目录 [--dir ...] [--list 照片清单.txt] [--n 40] [--seed 1]

三版都强制生成（不管自动判断怎么说，形变后照常做防穿帮检查，检查不通过的那版等于原图），
这样评审结果既能回答“瘦有没有用、瘦多少合适”，也能对照自动判断（key.json 里的 decision）。
输出：NN.jpg（A|B|C，脸部及周围）、key.json（答案与自动判断）、答卷.txt（每行填最好的一张，可加“穿帮”等备注）。
"""
import argparse
import glob
import json
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.face import detect_faces  # noqa: E402
from scripts.slim import MIN_FACE_PX, _check, _face_like, is_child, need, warp_face  # noqa: E402


def main_face(im):
    best = None
    for f in detect_faces(im):
        if f.points is None or f.depth is None:
            continue
        w = float(np.linalg.norm(f.points[454] - f.points[234]))
        if w < MIN_FACE_PX or not _face_like(im, f)[0] or is_child(im, f):
            continue
        if best is None or w > best[1]:
            best = (f, w)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dir", action="append", default=[])
    ap.add_argument("--list", help="照片路径清单（每行一个）")
    ap.add_argument("--n", type=int, default=40, help="最多几张")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    files = [f for d in a.dir for f in sorted(glob.glob(os.path.join(d, "*.[jJ][pP][gG]")))]
    if a.list:
        files += [l.strip() for l in open(a.list, encoding="utf-8") if l.strip()]
    rnd = random.Random(a.seed)
    rnd.shuffle(files)
    os.makedirs(a.out, exist_ok=True)
    key, n = {}, 0
    for f in files:
        if n >= a.n:
            break
        im = cv2.imread(f)
        if im is None:
            continue
        mf = main_face(im)
        if mf is None:
            continue
        face, fw = mf
        decision = need(face, im)[1]
        versions = {"不瘦": im}
        for name, amt in (("轻度", 0.5), ("中度", 1.0)):
            warped, stats = warp_face(im, face, amt)
            ok, why = _check(stats)
            versions[name] = warped if ok else im
            if not ok:
                decision += f"；{name}检查不通过（{why}）"
        p = face.points
        x0, y0 = (p.min(axis=0) - fw * 0.6).astype(int)
        x1, y1 = (p.max(axis=0) + fw * 0.6).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        names = list(versions)
        rnd.shuffle(names)
        tiles = []
        for letter, name in zip("ABC", names):
            c = versions[name][y0:y1, x0:x1]
            c = cv2.resize(c, (600, int(600 * c.shape[0] / c.shape[1])), interpolation=cv2.INTER_AREA)
            cv2.putText(c, letter, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 5)
            cv2.putText(c, letter, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 2)
            tiles.append(c)
        gap = np.full((tiles[0].shape[0], 10, 3), 255, np.uint8)
        n += 1
        tag = f"{n:02d}"
        cv2.imwrite(os.path.join(a.out, f"{tag}.jpg"), np.hstack([tiles[0], gap, tiles[1], gap, tiles[2]]),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        key[tag] = {"file": f, "order": dict(zip("ABC", names)), "decision": decision}
        print(tag, os.path.basename(f), decision, flush=True)
    json.dump(key, open(os.path.join(a.out, "key.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(os.path.join(a.out, "答卷.txt"), "w", encoding="utf-8") as fh:
        fh.write("每行填最好看的那一张（A/B/C），三张看不出差别填“同”；可加备注，如“B 穿帮”“A 脸歪了”。\n")
        for tag in key:
            fh.write(f"{tag}: \n")


if __name__ == "__main__":
    main()
