"""人像精修流水线（OpenCV + 可选的 mediapipe 关键点 / BiSeNet 人脸分割）。

用法:
    python main.py                       # 处理 data/raw/ 下所有图片，输出到 output/
    python main.py photo.jpg -o out.jpg  # 单张
    python main.py data/raw --compare    # 额外输出前后对比图
    python main.py photo.jpg --debug     # 输出各阶段结果、蒙版、瑕疵分类和 100% 局部对比，用于质量检查
"""
import argparse
import os
import time
from collections import Counter
from dataclasses import fields, replace

import cv2
import numpy as np

from scripts.blemish import remove_blemishes
from scripts.color_correction import color_correct
from scripts.debug_report import write_debug
from scripts.dodge_burn import dodge_burn
from scripts.face import detect_faces
from scripts.frequency_separation import frequency_separation
from scripts.preprocess import preprocess_image
from scripts.save_image import save_image
from scripts.sharpening import sharpen_image
from scripts.skin_mask import feature_scale, skin_mask
from scripts.skin_smoothing import skin_smoothing
from scripts.warping import slim_face
from scripts.whitening import whiten_skin

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def person_roi(shape, faces):
    """皮肤处理只需要人物周围：每张脸左右 3.5 倍脸宽、上 1.5 倍到下 7.5 倍（比人像分割区域多留半个脸宽给羽化）。"""
    h, w = shape[:2]
    x0, y0, x1, y1 = w, h, 0, 0
    for f in faces:
        x, y, bw, _ = f.box
        cx = x + bw / 2
        x0, x1 = min(x0, cx - 3.5 * bw), max(x1, cx + 3.5 * bw)
        y0, y1 = min(y0, y - 1.5 * bw), max(y1, y + 7.5 * bw)
    return int(max(x0, 0)), int(max(y0, 0)), int(min(x1, w)), int(min(y1, h))


def _shift_face(f, dx, dy):
    x, y, bw, bh = f.box
    pts = None if f.points is None else f.points - np.float32([dx, dy])
    return replace(f, box=(x - dx, y - dy, bw, bh), points=pts)


def _embed_masks(masks, shape, roi):
    """把人物区域的蒙版放回整图尺寸（调试图、--save-mask 用）。"""
    x0, y0, x1, y1 = roi
    full = {}
    for k in fields(masks):
        v = getattr(masks, k.name)
        if isinstance(v, np.ndarray):
            big = np.zeros(shape[:2], v.dtype)
            big[y0:y1, x0:x1] = v
            v = big
        full[k.name] = v
    return type(masks)(**full)


def whiten_amount(crop, masks, strength):
    """美白按脸原本的亮度递减：脸部 L* 58 以下全额，70 以上不美白（本来就亮的脸再提会发白）。"""
    if strength <= 0 or not (masks.face_hard > 0).any():
        return strength
    L = float(np.median(cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[..., 0][masks.face_hard > 0])) / 2.55
    return strength * float(np.clip((70 - L) / 12, 0, 1))


def process_image(image, opts, full_masks=False):
    """返回 (结果, 蒙版, 瑕疵列表, 各阶段图像, 附加信息)。

    皮肤类步骤只在人物区域（裁剪）上做再贴回，调色、锐化作用于整图；24MP 原图上人物通常只占一小块。
    没有人脸时跳过所有皮肤步骤。
    """
    faces = detect_faces(image, use_landmarks=not opts.no_landmarks)
    scale = feature_scale(image, faces)
    stages = {"original": image}
    h, w = image.shape[:2]
    roi = person_roi(image.shape, faces) if faces else (0, 0, w, h)
    x0, y0, x1, y1 = roi
    sub_faces = [_shift_face(f, x0, y0) for f in faces]
    masks = skin_mask(image[y0:y1, x0:x1], sub_faces, use_parsing=not opts.no_parsing)

    def on_person(step):
        out = image.copy()
        if faces:
            out[y0:y1, x0:x1] = step(image[y0:y1, x0:x1])
        return out

    # 瑕疵分类修复
    blemishes = []

    def fix(crop):
        crop, found = remove_blemishes(crop, masks, sub_faces, scale, opts.blemish, opts.freckles, opts.moles)
        blemishes.extend(replace(b, center=(b.center[0] + x0, b.center[1] + y0)) for b in found)
        return crop
    image = on_person(fix)
    stages["blemish"] = image
    # 色彩校正（放在磨皮前，避免放大被磨掉的颗粒）
    face_px = image[y0:y1, x0:x1][masks.face_hard > 0] if faces else None
    image = color_correct(image, opts.color, face_px)
    stages["color"] = image
    # 匀肤：色度低频均衡
    image = on_person(lambda c: frequency_separation(c, masks, scale, opts.even))
    stages["even"] = image
    # Dodge & Burn：亮度局部均衡、黑眼圈
    image = on_person(lambda c: dodge_burn(c, masks, sub_faces, scale, opts.dodge_burn, opts.eye))
    stages["dodge_burn"] = image
    # 多尺度磨皮
    image = on_person(lambda c: skin_smoothing(c, masks, scale, opts.smooth, opts.grain))
    stages["smooth"] = image
    # 美白
    image = on_person(lambda c: whiten_skin(c, masks.tone, whiten_amount(c, masks, opts.whiten)))
    stages["whiten"] = image
    # 锐化（只锐化非皮肤）
    skin_full = np.zeros((h, w), np.float32)
    skin_full[y0:y1, x0:x1] = masks.skin
    image = sharpen_image(image, skin_full, scale, opts.sharpen)
    stages["sharpen"] = image
    # 瘦脸（最后做，前面各步用的蒙版都对应变形前的位置）
    image = on_person(lambda c: slim_face(c, sub_faces, opts.slim))
    stages["slim"] = image

    if full_masks:
        masks = _embed_masks(masks, image.shape, roi)
    info = {"faces": faces, "scale": scale,
            "landmarks": any(f.points is not None for f in faces)}
    return image, masks, blemishes, stages, info


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
    ap.add_argument("--blemish", type=float, default=1.0, help="瑕疵检测灵敏度，0 关闭（默认 1.0）")
    ap.add_argument("--freckles", type=float, default=0.0, help="雀斑减淡强度 0~1（默认 0 保留）")
    ap.add_argument("--moles", action="store_true", help="去除痣（默认保留）")
    ap.add_argument("--even", type=float, default=0.5, help="匀肤（色度）强度 0~1（默认 0.5）")
    ap.add_argument("--dodge-burn", type=float, default=0.6, help="局部明暗均衡强度 0~1（默认 0.6）")
    ap.add_argument("--eye", type=float, default=0.5, help="黑眼圈提亮强度 0~1（默认 0.5）")
    ap.add_argument("--smooth", type=float, default=0.7, help="磨皮（中频）强度 0~1（默认 0.7）")
    ap.add_argument("--grain", type=float, default=0.6, help="颗粒/噪点抑制 0~1（默认 0.6，毛孔保留）")
    ap.add_argument("--whiten", type=float, default=0.2, help="美白强度 0~1（默认 0.2）")
    ap.add_argument("--slim", type=float, default=0.0, help="瘦脸强度 0~1（默认 0 关闭）")
    ap.add_argument("--color", type=float, default=0.3, help="白平衡/对比度校正强度 0~1（默认 0.3）")
    ap.add_argument("--sharpen", type=float, default=0.5, help="锐化强度（默认 0.5）")
    ap.add_argument("--no-parsing", action="store_true", help="不用人脸/人像语义分割，只用肤色模型")
    ap.add_argument("--no-landmarks", action="store_true", help="不用 mediapipe 关键点，只用 Haar 人脸框")
    ap.add_argument("--max-side", type=int, help="长边超过该值时先等比缩小")
    ap.add_argument("--compare", action="store_true", help="额外保存左右对比图 *_compare.jpg")
    ap.add_argument("--save-mask", action="store_true", help="额外保存皮肤掩码 *_mask.png")
    ap.add_argument("--debug", nargs="?", const="debug", metavar="DIR",
                    help="保存各阶段输出、蒙版、瑕疵分类和 100%% 局部对比到 DIR/<图名>/（默认 debug/）")
    opts = ap.parse_args()

    if not os.path.exists(opts.input):
        ap.error(f"找不到输入: {opts.input}")
    jobs = collect_jobs(opts.input, opts.output)
    if not jobs:
        print(f"{opts.input} 下没有图片")
        return

    params = {k: v for k, v in vars(opts).items() if k not in ("input", "output")}
    for src, dst in jobs:
        t0 = time.time()
        original = preprocess_image(src, opts.max_side)
        result, masks, blemishes, stages, info = process_image(original.copy(), opts,
                                                                  full_masks=bool(opts.debug or opts.save_mask))
        save_image(result, dst)
        stem = os.path.splitext(dst)[0]
        if opts.compare:
            save_image(np.hstack([original, result]), f"{stem}_compare.jpg")
        if opts.save_mask:
            save_image((masks.skin * 255).astype(np.uint8), f"{stem}_mask.png")
        if opts.debug:
            name = os.path.splitext(os.path.basename(src))[0]
            write_debug(opts.debug, name, stages, masks, blemishes, info["faces"], info["scale"], params)
        if not info["faces"]:
            mode = "未检出，跳过皮肤处理"
        else:
            mode = ("关键点" if info["landmarks"] else "Haar") + ("+分割" if masks.method == "parsing" else "")
        kinds = Counter(b.kind for b in blemishes if b.weight > 0)
        fixed = "、".join(f"{k} {v}" for k, v in kinds.items()) or "无"
        print(f"{os.path.basename(src)}: 人脸 {len(info['faces'])}（{mode}），处理瑕疵 {fixed}，"
              f"{time.time() - t0:.1f}s -> {dst}")


if __name__ == "__main__":
    main()
