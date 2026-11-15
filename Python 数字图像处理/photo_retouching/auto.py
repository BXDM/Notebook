"""自动校正：亮度 + 降噪（人像、夜景、风景通用，不做皮肤处理）。

用法:
    python auto.py photo.jpg                 # 输出 output/photo_auto.jpg
    python auto.py ~/PycharmProjects/sample  # 批量，输出到 output/
    python auto.py photo.jpg --compare       # 额外输出左右对比图
    python auto.py photo.jpg --debug         # 增益图、分析报告 -> debug/<图名>/
    python auto.py photo.jpg --target 62 --denoise 0
顺序：先按分析结果算增益图，再降噪（强度取决于提亮后的噪声），最后乘增益。
"""
import argparse
import os
import time

import cv2
import numpy as np

from scripts.analyze import analyze
from scripts.denoise import denoise
from scripts.exposure import apply_gain, gain_map
from scripts.preprocess import preprocess_image
from scripts.save_image import save_image

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def process(image, path, opts):
    an = analyze(image, path, use_landmarks=not opts.no_landmarks)
    notes = [an.summary()]
    if opts.exposure > 0:
        gain, k, why = gain_map(image, an, opts.target, opts.exposure)
        notes.append(f"曝光 ×{k:.2f}（{why}），增益范围 {np.percentile(gain, 1):.2f}~{np.percentile(gain, 99):.2f}")
    else:
        gain = None
    result, how = denoise(image, an.noise_luma, an.noise_chroma, gain, opts.denoise)
    notes.append(f"降噪：{how}")
    if gain is not None:
        result = apply_gain(result, gain)
    return result, gain, an, notes


def gain_view(gain):
    """增益图可视化：灰 = 不变，越亮 = 提亮越多，越暗 = 压暗（高光）。"""
    v = np.clip(128 + np.log2(np.maximum(gain, 1e-3)) * 64, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(v, cv2.COLORMAP_TWILIGHT_SHIFTED)


def collect_jobs(inp, out):
    if os.path.isdir(inp):
        out = out or "output"
        os.makedirs(out, exist_ok=True)
        return [(os.path.join(inp, f), os.path.join(out, os.path.splitext(f)[0] + "_auto.jpg"))
                for f in sorted(os.listdir(inp)) if f.lower().endswith(EXTS)]
    if out is None:
        os.makedirs("output", exist_ok=True)
        out = os.path.join("output", os.path.splitext(os.path.basename(inp))[0] + "_auto.jpg")
    return [(inp, out)]


def main():
    ap = argparse.ArgumentParser(description="自动校正：亮度 + 降噪")
    ap.add_argument("input", help="图片文件或目录")
    ap.add_argument("-o", "--output", help="输出文件（单张）或目录（批量），默认 output/")
    ap.add_argument("--exposure", type=float, default=1.0, help="亮度校正强度 0~1，0 关闭（默认 1）")
    ap.add_argument("--target", type=float, default=58.0, help="人像脸部目标亮度 L*（默认 58）")
    ap.add_argument("--denoise", type=float, default=1.0, help="降噪强度 0~1，0 关闭（默认 1，噪声低时自动跳过）")
    ap.add_argument("--no-landmarks", action="store_true", help="不用 mediapipe 关键点，只用 Haar 人脸框")
    ap.add_argument("--compare", action="store_true", help="额外保存左右对比图 *_compare.jpg")
    ap.add_argument("--debug", nargs="?", const="debug", metavar="DIR", help="保存增益图和分析报告到 DIR/<图名>/")
    opts = ap.parse_args()

    if not os.path.exists(opts.input):
        ap.error(f"找不到输入: {opts.input}")
    jobs = collect_jobs(opts.input, opts.output)
    for src, dst in jobs:
        t0 = time.time()
        image = preprocess_image(src)
        result, gain, an, notes = process(image, src, opts)
        save_image(result, dst)
        stem = os.path.splitext(dst)[0]
        if opts.compare:
            save_image(np.hstack([image, result]), f"{stem}_compare.jpg")
        if opts.debug:
            d = os.path.join(opts.debug, os.path.splitext(os.path.basename(src))[0])
            os.makedirs(d, exist_ok=True)
            if gain is not None:
                save_image(gain_view(gain), os.path.join(d, "gain.jpg"))
            with open(os.path.join(d, "report.txt"), "w", encoding="utf-8") as fh:
                fh.write("\n".join(notes) + "\n")
        print(f"{os.path.basename(src)}: " + "；".join(notes) + f"，{time.time() - t0:.1f}s -> {dst}")


if __name__ == "__main__":
    main()
