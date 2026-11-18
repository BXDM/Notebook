"""2.0 统一入口：一张照片进来，分析 → 编排器选模块 → 唯一成片。

    python photo.py photo.jpg                 # 输出 output/photo.jpg
    python photo.py 照片目录 -o out/            # 批量，文件名不变
    python photo.py 照片目录 -o out/ --mode main   # 兼容模式：与 main.py 输出一致（回归用）
    python photo.py 照片目录 -o out/ --mode auto   # 兼容模式：与 auto.py 输出一致

engine 模式（默认）：由 engine/modules/core.py 里拆出的模块组成，编排器按分析结果决定每个模块做不做。
目前沿用 1.x 的分工：有人脸走人像链路，没有人脸走场景链路（拉直、降噪、曝光、去雾、天空、夜景、晚霞）。
与 1.x 的区别：风格调色（橙青）默认不做（--grade auto 恢复 1.x 的夜景自动橙青）。
main.py / auto.py 的参数都可以直接传（如 --smooth、--sky 0），各自取用。
"""
import argparse
import os
import sys
import time

import auto as A
import main as M
from engine import Orchestrator, PhotoContext
from engine.modules.core import default_modules
from engine.modules.legacy import AutoPipeline, PortraitPipeline
from scripts.preprocess import preprocess_image
from scripts.save_image import save_image

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def parse(argv):
    ap = argparse.ArgumentParser(description="自动摄影后期引擎 2.0")
    ap.add_argument("input", help="图片文件或目录")
    ap.add_argument("-o", "--output", help="输出文件（单张）或目录（批量），默认 output/")
    ap.add_argument("--mode", choices=["engine", "main", "auto"], default="engine",
                    help="engine：按分析结果自动选模块（默认）；main/auto：兼容模式，与 1.x 对应脚本输出一致")
    opts, rest = ap.parse_known_args(argv)
    p_opts, p_rest = M.build_parser(io=False).parse_known_args(rest)
    a_opts, a_rest = A.build_parser(io=False).parse_known_args(rest)
    unknown = [x for x in p_rest if x in a_rest and x.startswith("-")]
    if unknown:
        ap.error(f"不认识的参数：{' '.join(unknown)}")
    if opts.mode == "engine" and not any(x == "--grade" or x.startswith("--grade=") for x in rest):
        a_opts.grade = "none"          # 2.0：不默认改氛围
    opts.portrait, opts.auto = p_opts, a_opts
    opts.max_side = p_opts.max_side
    opts.no_landmarks = p_opts.no_landmarks
    return ap, opts


def modules_for(mode):
    if mode == "main":
        return [PortraitPipeline(always=True)]
    if mode == "auto":
        return [AutoPipeline(always=True)]
    return default_modules()


def collect_jobs(inp, out):
    if os.path.isdir(inp):
        out = out or "output"
        os.makedirs(out, exist_ok=True)
        return [(os.path.join(inp, f), os.path.join(out, f)) for f in sorted(os.listdir(inp)) if f.lower().endswith(EXTS)]
    if out is None:
        os.makedirs("output", exist_ok=True)
        out = os.path.join("output", os.path.basename(inp))
    return [(inp, out)]


def run_one(src, dst, opts):
    t0 = time.time()
    image = preprocess_image(src, opts.max_side)
    ctx = PhotoContext(image, src, opts)
    ctx.profile = M.pick_profile(ctx.exif, opts.portrait.profile)
    ctx.popts = M.profile_opts(opts.portrait, ctx.profile, M.PARSER_DEFAULTS)
    result = Orchestrator(modules_for(opts.mode)).run(ctx)
    save_image(result, dst)
    if opts.mode == "engine" and ctx.popts.debug and ctx.faces:
        from scripts.debug_report import write_debug
        p = ctx.person if "person" in ctx._cache else None
        if p is not None:
            masks = M._embed_masks(p.masks, result.shape, p.roi)
            write_debug(ctx.popts.debug, os.path.splitext(os.path.basename(src))[0], ctx.stages, masks,
                        ctx.get("blemishes", []), ctx.faces, p.scale, vars(ctx.popts))
    print(f"{os.path.basename(src)}: {Orchestrator.summary(ctx)}，{time.time() - t0:.1f}s -> {dst}", flush=True)


def main(argv=None):
    ap, opts = parse(sys.argv[1:] if argv is None else argv)
    if not os.path.exists(opts.input):
        ap.error(f"找不到输入: {opts.input}")
    jobs = collect_jobs(opts.input, opts.output)
    failed = []
    for src, dst in jobs:
        try:
            run_one(src, dst, opts)
        except Exception as e:  # 批处理里一张出错不能拖垮整批
            failed.append(os.path.basename(src))
            print(f"{os.path.basename(src)}: 失败，跳过（{type(e).__name__}: {e}）")
    if failed:
        print(f"共 {len(jobs)} 张，失败 {len(failed)} 张：{'、'.join(failed)}")


if __name__ == "__main__":
    main()
