"""选片（废片分拣）：主体模糊、闭眼、连拍重复（每组留一张）。只标记，不移动、不删除原图。

    python cull.py 照片目录 -o 报告目录

报告目录：选片.csv（每张的判定、原因、组号、清晰度、闭眼值）、总览.jpg（全部缩略图，绿框保留、红框废片、
灰框重复，★ 为组内最佳）、组/NN.jpg（每个连拍组一张，便于核对组内挑得对不对）。
算法见 engine/batch/cull.py。
"""
import argparse
import csv
import glob
import os
from multiprocessing import Pool

import cv2
import numpy as np

from engine.batch import aesthetic
from engine.batch.cull import analyze_one, calibrate, decide, embed, group, verdict

COLORS = {"保留": (60, 200, 60), "废片": (40, 40, 230), "重复": (150, 150, 150)}


def tile(s, size=220):
    t = np.zeros((size, size, 3), np.uint8)
    if getattr(s, "thumb", None) is not None:
        h, w = s.thumb.shape[:2]
        r = size / max(h, w)
        im = cv2.resize(s.thumb, (int(w * r), int(h * r)))
        y, x = (size - im.shape[0]) // 2, (size - im.shape[1]) // 2
        t[y:y + im.shape[0], x:x + im.shape[1]] = im
    v = verdict(s)
    c = COLORS["保留" if v.startswith("保留") else v]
    t = cv2.copyMakeBorder(t, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=c)
    label = s.name[4:8] if s.name.upper().startswith(("DSCF", "IMG_")) else s.name[:8]
    cv2.putText(t, ("*" if s.best else "") + label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    tags = ",".join({"模糊": "blur", "闭眼": "eyes?", "眨眼": "blink", "垂眼": "down", "重复": "dup", "焦点在别处": "misfocus", "疑似拖影": "motion", "眼睛被遮挡": "eyes-hidden"}[x] for x in s.tags if x != "读不出")
    cv2.putText(t, tags, (8, size + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return t


def sheet(shots, cols):
    ts = [tile(s) for s in shots]
    while len(ts) % cols:
        ts.append(np.zeros_like(ts[0]))
    return np.vstack([np.hstack(ts[i:i + cols]) for i in range(0, len(ts), cols)])


def main():
    ap = argparse.ArgumentParser(description="选片：主体模糊、闭眼、连拍重复")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default="output/选片")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    files = sorted(f for e in ("*.jpg", "*.JPG", "*.jpeg", "*.png") for f in glob.glob(os.path.join(a.input, e)))
    with Pool(a.workers) as p:
        shots = p.map(analyze_one, files, chunksize=2)
    embed(shots)
    imgs = [s for s in shots if getattr(s, "aes_img", None) is not None]
    scores = aesthetic.score([s.aes_img for s in imgs])
    if scores is not None:
        for s, v in zip(imgs, scores):
            s.aes = v
    th = calibrate(shots)
    for _, note in th.values():
        print("设备校正", note)
    decide(group(shots), th)
    os.makedirs(os.path.join(a.output, "组"), exist_ok=True)
    with open(os.path.join(a.output, "选片.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["文件", "判定", "原因", "组", "组内最佳", "清晰度", "美学分", "方向比", "有人脸", "闭眼值", "下视值", "机型"])
        for s in shots:
            w.writerow([s.name, verdict(s), "、".join(s.tags), s.group, "是" if s.best else "", f"{s.sharp:.3f}" if s.sharp else "",
                        f"{s.aes:.2f}" if s.aes is not None else "", f"{s.aniso:.2f}" if s.aniso is not None else "",
                        "是" if s.has_face else "", f"{s.blink:.2f}", f"{s.look_down:.2f}", s.camera])
    cv2.imwrite(os.path.join(a.output, "总览.jpg"), sheet(shots, 12), [cv2.IMWRITE_JPEG_QUALITY, 85])
    groups = {}
    for s in shots:
        groups.setdefault(s.group, []).append(s)
    for g, members in groups.items():
        if len(members) > 1:
            cv2.imwrite(os.path.join(a.output, "组", f"{g:03d}.jpg"), sheet(members, min(len(members), 8)),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
    v = [verdict(s) for s in shots]
    print(f"共 {len(shots)} 张，{len(groups)} 组（多张的 {sum(len(m) > 1 for m in groups.values())} 组）："
          f"保留 {sum(x.startswith('保留') for x in v)}、重复 {v.count('重复')}、废片 {v.count('废片')}；"
          f"模糊 {sum('模糊' in s.tags for s in shots)}、眨眼 {sum('眨眼' in s.tags for s in shots)}、"
          f"闭眼（可能是故意的，保留）{sum('闭眼' in s.tags for s in shots)}、垂眼 {sum('垂眼' in s.tags for s in shots)}")


if __name__ == "__main__":
    main()
