"""自动校正：建筑拉直 + 亮度 + 降噪 + 分层去雾 + 天空高光恢复（人像、夜景、风景通用，不做皮肤处理）。

用法:
    python auto.py photo.jpg                 # 输出 output/photo_auto.jpg
    python auto.py ~/PycharmProjects/sample  # 批量，输出到 output/
    python auto.py photo.jpg --compare       # 额外输出左右对比图
    python auto.py photo.jpg --debug         # 增益图、分析报告 -> debug/<图名>/
    python auto.py photo.jpg --target 62 --denoise 0
    python auto.py photo.jpg --keystone 1    # 透视完全拉直（默认 0.8）；--straighten 0 关闭拉直
    python auto.py photo.jpg --sky 0         # 关闭天空高光恢复；--sky-darken 调天空减光档数（默认 0.6）
    python auto.py photo.jpg --dehaze 0      # 关闭分层去雾（只在户外、有天空时做）
    python auto.py photo.jpg --grade none    # 默认 auto：自动识别夜景并上橙青；--night / --glow 调夜景光线、晚霞增强
顺序：先拉直（建筑照片才做），再按分析结果算增益图、降噪（强度取决于提亮后的噪声）、乘增益，再分层去雾（户外才做），最后恢复天空高光。
"""
import argparse
import os
import time

import cv2
import numpy as np

from scripts import geometry
from scripts.analyze import analyze, read_exif
from scripts.dehaze import dehaze
from scripts.learned_denoise import smart_denoise
from scripts.grade import orange_teal
from scripts.scene import classify
from scripts.scene_adjust import enhance_glow, night_light
from scripts.exposure import apply_gain, gain_map
from scripts.preprocess import preprocess_image
from scripts.sky import analyze_sky, recover_sky
from scripts.save_image import save_image

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def process(image, path, opts):
    notes = []
    geo = None
    if opts.straighten:
        geo = geometry.estimate(image, read_exif(path), keystone=opts.keystone)
        if geo.ok:
            image = geometry.apply(image, geo)
            notes.append(f"拉直：歪 {geo.roll:+.2f}°、仰俯 {geo.pitch:+.2f}°（透视校正 {opts.keystone:.0%}），"
                         f"关键边 {len(geo.edges)} 条，裁剪后 {geo.size[0]}x{geo.size[1]}")
        else:
            notes.append(f"拉直：跳过，{geo.reason}")
    an = analyze(image, path, use_landmarks=not opts.no_landmarks)
    scene = classify(image, an.ev100)
    notes.insert(0, an.summary())
    notes.insert(1, f"场景：{scene.label}")
    if opts.exposure > 0:
        gain, k, why = gain_map(image, an, opts.target, opts.exposure, shadows=opts.shadows)
        notes.append(f"曝光 ×{k:.2f}（{why}），增益范围 {np.percentile(gain, 1):.2f}~{np.percentile(gain, 99):.2f}")
    else:
        gain = None
    result, how = smart_denoise(image, None, opts.gpu, an.exif.get("iso")) if opts.denoise > 0 else (image, "关闭")
    notes.append(f"降噪：{how}")
    if gain is not None:
        result = apply_gain(result, gain)
    sky_w = None
    sky = analyze_sky(result)
    if opts.dehaze > 0:
        if an.ev100 is not None and an.ev100 < 10:
            notes.append(f"去雾：现场亮度 EV {an.ev100:.1f}，不是白天户外，跳过——雾的模型在夜景、黄昏不成立")
        elif sky.found:
            result, how = dehaze(result, opts.dehaze)
            notes.append(f"去雾：{how}")
        else:
            notes.append("去雾：没有天空（室内或近景），跳过——白墙会被误当成雾")
    if opts.sky > 0:
        result, sky_w, how = recover_sky(result, analyze_sky(result), opts.sky, opts.sky_darken)
        notes.append(f"天空：{how}")
    if scene.kind == "night" and opts.night > 0:
        result, how = night_light(result, opts.night)
        notes.append(how)
    if scene.glow > 0 and opts.glow > 0:
        result, how = enhance_glow(result, scene, opts.glow)
        notes.append(how)
    grade = opts.grade if opts.grade != "auto" else ("orange-teal" if scene.kind == "night" else "none")
    if grade == "orange-teal":
        result, how = orange_teal(result, opts.grade_strength)
        notes.append(f"调色：{how}")
    return result, gain, an, notes, geo, sky_w


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
    ap.add_argument("--shadows", type=float, default=1.0, help="大光比（白天室内、逆光）时的阴影提亮强度，0 关闭（默认 1，约 1.5 明显更亮）")
    ap.add_argument("--target", type=float, default=58.0, help="人像脸部目标亮度 L*（默认 58）")
    ap.add_argument("--denoise", type=float, default=1.0, help="降噪强度 0~1，0 关闭（默认 1，噪声低时自动跳过）")
    ap.add_argument("--straighten", type=int, default=1, help="建筑拉直，0 关闭（默认 1，没有建筑时自动跳过）")
    ap.add_argument("--keystone", type=float, default=0.8, help="仰拍透视校正比例 0~1（默认 0.8，1 为完全拉直）")
    ap.add_argument("--dehaze", type=float, default=1.0, help="分层去雾强度 0~1，0 关闭（默认 1，只在有天空时做，远景的雾保留）")
    ap.add_argument("--sky", type=float, default=1.0, help="天空高光恢复强度 0~1，0 关闭（默认 1，没有天空时跳过）")
    ap.add_argument("--sky-darken", type=float, default=0.6, help="天空顶部减光档数（默认 0.6，向下渐变到 0）")
    ap.add_argument("--grade", choices=["auto", "none", "orange-teal"], default="auto",
                    help="风格调色：auto 只给夜景上橙青（默认），none 不调，orange-teal 全部上")
    ap.add_argument("--night", type=float, default=1.0, help="夜景光线调整强度，0 关闭（默认 1，只对夜景）")
    ap.add_argument("--glow", type=float, default=1.0, help="晚霞增强强度，0 关闭（默认 1，按识别出的晚霞强度作用）")
    ap.add_argument("--grade-strength", type=float, default=1.0, help="风格调色强度（默认 1）")
    ap.add_argument("--gpu", choices=["auto", "never", "always"], default="always",
                    help="学习型降噪放哪：always 一律上 Modal GPU（开发阶段默认），auto 只在预计 CPU 超过 5 分钟时上传，never 全在本机")
    ap.add_argument("--no-landmarks", action="store_true", help="不用 mediapipe 关键点，只用 Haar 人脸框")
    ap.add_argument("--compare", action="store_true", help="额外保存左右对比图 *_compare.jpg")
    ap.add_argument("--debug", nargs="?", const="debug", metavar="DIR", help="保存增益图和分析报告到 DIR/<图名>/")
    opts = ap.parse_args()

    if not os.path.exists(opts.input):
        ap.error(f"找不到输入: {opts.input}")
    jobs = collect_jobs(opts.input, opts.output)
    failed = []
    for src, dst in jobs:
        try:
            run_one(src, dst, opts)
        except Exception as e:  # 批处理里一张出错不能拖垮整批：记下来，继续下一张
            failed.append(os.path.basename(src))
            print(f"{os.path.basename(src)}: 失败，跳过（{type(e).__name__}: {e}）")
    if failed:
        print(f"共 {len(jobs)} 张，失败 {len(failed)} 张：{'、'.join(failed)}")


def run_one(src, dst, opts):
    t0 = time.time()
    image = preprocess_image(src)
    result, gain, an, notes, geo, sky_w = process(image, src, opts)
    save_image(result, dst)
    stem = os.path.splitext(dst)[0]
    if opts.compare:
        left = cv2.resize(image, (result.shape[1] * image.shape[0] // result.shape[0], result.shape[0])) \
            if image.shape[:2] != result.shape[:2] else image
        save_image(np.hstack([left, result]), f"{stem}_compare.jpg")
    if opts.debug:
        d = os.path.join(opts.debug, os.path.splitext(os.path.basename(src))[0])
        os.makedirs(d, exist_ok=True)
        if gain is not None:
            save_image(gain_view(gain), os.path.join(d, "gain.jpg"))
        if sky_w is not None:
            save_image((sky_w * 255).astype(np.uint8), os.path.join(d, "sky_weight.jpg"))
        if geo is not None and geo.box is not None:
            save_image(geometry.overlay(image, geo), os.path.join(d, "key_edges.jpg"))
        with open(os.path.join(d, "report.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(notes) + "\n")
    print(f"{os.path.basename(src)}: " + "；".join(notes) + f"，{time.time() - t0:.1f}s -> {dst}")


if __name__ == "__main__":
    main()
