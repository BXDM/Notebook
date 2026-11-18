"""1.x 流程原样包装成 2.0 模块（第一步：先接入、输出与 1.x 一致，之后再逐步拆成独立模块）。

- PortraitPipeline = main.py 的 process_image（人像配置 / R10 配置都在里面），人脸用 PhotoContext 里算好的。
- AutoPipeline = auto.py 的 process（拉直、亮度、降噪、去雾、天空、夜景/晚霞、橙青）。
"""
import os
from collections import Counter

from engine.module import Module, Report, Request


class PortraitPipeline(Module):
    name = "人像精修(1.x)"
    region = "person"
    order = 40

    def __init__(self, always=False):
        self.always = always           # 兼容模式：无论有没有人脸都走（与 main.py 一致）

    def plan(self, ctx):
        if self.always or ctx.faces:
            return Request(goal="portrait_retouch", preserve=["makeup", "texture", "face_structure"]), ""
        return None, "没有人脸"

    def run(self, ctx, req):
        import main as M
        o = ctx.opts.portrait
        profile = M.pick_profile(ctx.exif, o.profile)
        po = M.profile_opts(o, profile, M.PARSER_DEFAULTS)
        result, masks, blemishes, stages, info = M.process_image(
            ctx.image.copy(), po, full_masks=bool(po.debug or po.save_mask), iso=ctx.iso, path=ctx.path,
            profile=profile, faces=ctx.faces)
        ctx.update(result, "portrait")
        ctx.stages.update(stages)
        ctx.put("masks", masks)
        ctx.put("blemishes", blemishes)
        if po.debug and ctx.path:
            from scripts.debug_report import write_debug
            name = os.path.splitext(os.path.basename(ctx.path))[0]
            write_debug(po.debug, name, stages, masks, blemishes, info["faces"], info["scale"], vars(po))
        if not info["faces"]:
            mode = "未检出，跳过皮肤处理"
        else:
            mode = ("关键点" if info["landmarks"] else "Haar") + ("+分割" if masks.method == "parsing" else "")
        kinds = Counter(b.kind for b in blemishes if b.weight > 0)
        fixed = "、".join(f"{k} {v}" for k, v in kinds.items()) or "无"
        notes = (f"{profile}：人脸 {len(info['faces'])}（{mode}），{info['exposure']}，{info['veil']}，"
                 f"{info['noise']}，处理瑕疵 {fixed}")
        return Report(self.name, ran=True, notes=notes, changed=None, channels=("L", "ab"),
                      bands=("low", "mid", "high"))


class AutoPipeline(Module):
    name = "自动校正(1.x)"
    region = "global"
    geometry = True
    order = 40

    def __init__(self, always=False):
        self.always = always

    def plan(self, ctx):
        if self.always or not ctx.faces:
            return Request(goal="scene_correction"), ""
        return None, "有人脸，由人像精修处理"

    def run(self, ctx, req):
        import auto as A
        o = ctx.opts.auto
        result, gain, an, notes, geo, sky_w = A.process(ctx.image, ctx.path, o)
        ctx.update(result, "auto", geometry=bool(geo is not None and geo.ok))
        if o.debug and ctx.path:
            import numpy as np
            from scripts import geometry
            from scripts.save_image import save_image
            d = os.path.join(o.debug, os.path.splitext(os.path.basename(ctx.path))[0])
            os.makedirs(d, exist_ok=True)
            if gain is not None:
                save_image(A.gain_view(gain), os.path.join(d, "gain.jpg"))
            if sky_w is not None:
                save_image((sky_w * 255).astype(np.uint8), os.path.join(d, "sky_weight.jpg"))
            if geo is not None and geo.box is not None:
                save_image(geometry.overlay(ctx.original, geo), os.path.join(d, "key_edges.jpg"))
            with open(os.path.join(d, "report.txt"), "w", encoding="utf-8") as fh:
                fh.write("\n".join(notes) + "\n")
        return Report(self.name, ran=True, notes="；".join(notes))
