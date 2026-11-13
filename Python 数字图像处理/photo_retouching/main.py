"""人像精修流水线（纯 OpenCV，无需模型）。

用法:
    python main.py                       # 处理 data/raw/ 下所有图片，输出到 output/
    python main.py photo.jpg -o out.jpg  # 单张
    python main.py data/raw --compare    # 额外输出前后对比图
"""
import argparse
import os
import time

import cv2
import numpy as np

from scripts.blemish import remove_blemishes
from scripts.color_correction import color_correct
from scripts.frequency_separation import frequency_separation
from scripts.preprocess import preprocess_image
from scripts.save_image import save_image
from scripts.sharpening import sharpen_image
from scripts.skin_mask import detect_faces, feature_scale, skin_mask
from scripts.skin_smoothing import skin_smoothing
from scripts.warping import slim_face
from scripts.whitening import whiten_skin

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def process_image(image, opts):
    faces = detect_faces(image)
    scale = feature_scale(image, faces)
    hard, soft = skin_mask(image, faces)

    # 祛痘祛痣
    image, spots = remove_blemishes(image, hard, faces, scale, opts.blemish)
    # 色彩校正（放在磨皮前，避免 CLAHE 把磨掉的颗粒又拉出来）
    image = color_correct(image, opts.color)
    # 高低频分离匀肤
    image = frequency_separation(image, soft, scale, opts.even)
    # 磨皮
    image = skin_smoothing(image, soft, scale, opts.smooth)
    # 美白
    image = whiten_skin(image, soft, opts.whiten)
    # 锐化
    image = sharpen_image(image, soft, scale, opts.sharpen)
    # 瘦脸（最后做，前面各步用的掩码都对应变形前的位置）
    image = slim_face(image, faces, opts.slim)

    info = {"faces": len(faces), "spots": cv2.connectedComponents(spots)[0] - 1}
    return image, (soft * 255).astype(np.uint8), info


def collect_jobs(inp, out):
    if os.path.isdir(inp):
        out = out or "output"
        os.makedirs(out, exist_ok=True)
        return [(os.path.join(inp, f), os.path.join(out, f))
                for f in sorted(os.listdir(inp)) if f.lower().endswith(EXTS)]
    if out is None:
        os.makedirs("output", exist_ok=True)
        stem, ext = os.path.splitext(os.path.basename(inp))
        out = os.path.join("output", f"{stem}_retouched{ext}")
    return [(inp, out)]


def main():
    ap = argparse.ArgumentParser(description="人像精修：祛痘、匀肤、磨皮、美白、瘦脸、调色、锐化")
    ap.add_argument("input", nargs="?", default="data/raw", help="图片文件或目录（默认 data/raw）")
    ap.add_argument("-o", "--output", help="输出文件（单张）或目录（批量），默认 output/")
    ap.add_argument("--blemish", type=float, default=1.0, help="祛痘灵敏度，0 关闭（默认 1.0）")
    ap.add_argument("--even", type=float, default=0.5, help="匀肤强度 0~1（默认 0.5）")
    ap.add_argument("--smooth", type=float, default=0.7, help="磨皮强度 0~1（默认 0.7）")
    ap.add_argument("--whiten", type=float, default=0.2, help="美白强度 0~1（默认 0.2）")
    ap.add_argument("--slim", type=float, default=0.0, help="瘦脸强度 0~1（默认 0 关闭，位置是估计的）")
    ap.add_argument("--color", type=float, default=0.3, help="白平衡/对比度校正强度 0~1（默认 0.3）")
    ap.add_argument("--sharpen", type=float, default=0.5, help="锐化强度（默认 0.5）")
    ap.add_argument("--max-side", type=int, help="长边超过该值时先等比缩小")
    ap.add_argument("--compare", action="store_true", help="额外保存左右对比图 *_compare.jpg")
    ap.add_argument("--save-mask", action="store_true", help="额外保存皮肤掩码 *_mask.png")
    opts = ap.parse_args()

    if not os.path.exists(opts.input):
        ap.error(f"找不到输入: {opts.input}")
    jobs = collect_jobs(opts.input, opts.output)
    if not jobs:
        print(f"{opts.input} 下没有图片")
        return

    for src, dst in jobs:
        t0 = time.time()
        original = preprocess_image(src, opts.max_side)
        result, mask, info = process_image(original.copy(), opts)
        save_image(result, dst)
        stem = os.path.splitext(dst)[0]
        if opts.compare:
            save_image(np.hstack([original, result]), f"{stem}_compare.jpg")
        if opts.save_mask:
            save_image(mask, f"{stem}_mask.png")
        print(f"{os.path.basename(src)}: 人脸 {info['faces']}，修复斑点 {info['spots']}，"
              f"{time.time() - t0:.1f}s -> {dst}")


if __name__ == "__main__":
    main()
