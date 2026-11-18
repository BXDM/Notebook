"""2.0 模块：由 1.x 的 main.py（人像）和 auto.py（自动校正）拆出，算法本身不改。

按文档的职责划分：
- 技术修复：拉直（Geometry）、去眩光（Deveil）、降噪（SceneDenoise / NonSkinDenoise）、锐化（Sharpen）
- 场景影调：曝光（Exposure）、去雾（Dehaze）、天空（Sky）、夜景光线（NightLight）、晚霞（Glow）、光影增强（Light）
- 色彩：白平衡与肤色（Color）、风格调色（Grade）
- 人像：瑕疵（Blemish）、精修（Retouch：补光、匀肤、D&B、磨皮、肤质平衡、美白）、瘦脸（Slim）

执行顺序由 order 决定（编排器排序）。有人脸的照片：人像链路 + 拉直（只纠歪，不做透视校正，免得拉伸身材）
+ 去雾、天空（人物区域受保护）；夜景光线、晚霞、风格调色仍只给没人脸的照片。
"""
import numpy as np
from engine.module import Module, Report, Request

# 曝光、去眩光改了亮度：皮肤蒙版要重算（在新亮度上分割），人脸位置、场景类型不变
TONE_INVALIDATES = ("person",)


def _face_only(ctx, why="没有人脸"):
    return (Request(goal="portrait"), "") if ctx.faces else (None, why)


def _scene_only(ctx, why="有人脸，走人像链路"):
    return (None, why) if ctx.faces else (Request(goal="scene"), "")


def person_prob(ctx):
    """人物概率图（人像分割，0~1，已羽化），几何改动前一直复用。"""
    if ctx.get("person_prob") is None:
        from scripts.light_enhance import _person_mask
        ctx.put("person_prob", _person_mask(ctx.image))
    return ctx.get("person_prob")


def protect_person(ctx, before, after):
    """背景类处理（天空、去雾）写回前，把人物区域换回处理前：人物由人像链路负责，背景处理不能改肤色和脸部明暗。
    分割概率放大 1.3 倍再截断，边缘的发丝也算人物（宁可人物边上留一点没处理的背景，也不让发丝被压暗）。"""
    pm = person_prob(ctx)
    if pm is None:
        return after
    w = np.clip(pm * 1.3, 0, 1)[..., None]
    return np.clip(after * (1 - w) + before * w + 0.5, 0, 255).astype(np.uint8)


# ---------------- 技术修复 ----------------
class Geometry(Module):
    name, region, geometry, order = "拉直", "global", True, 10

    def plan(self, ctx):
        if not ctx.opts.auto.straighten:
            return None, "关闭"
        return Request(goal="straighten"), ""

    def run(self, ctx, req):
        from scripts import geometry
        # 有人脸：只纠左右歪，不做仰拍透视校正（透视校正会把人拉长或压扁）
        geo = geometry.estimate(ctx.image, ctx.exif, keystone=0.0 if ctx.faces else ctx.opts.auto.keystone)
        ctx.put("geometry", geo)
        if not geo.ok:
            return Report(self.name, ran=False, skipped=geo.reason)
        ctx.update(geometry.apply(ctx.image, geo), "geometry", invalidates="all")
        return Report(self.name, ran=True, notes=f"歪 {geo.roll:+.2f}°、仰俯 {geo.pitch:+.2f}°，裁剪后 {geo.size[0]}x{geo.size[1]}")


class Deveil(Module):
    name, region, order = "去眩光", "person", 15

    def plan(self, ctx):
        return _face_only(ctx)

    def run(self, ctx, req):
        from scripts.deveil import deveil
        image, note = deveil(ctx.image, ctx.faces, ctx.popts.deveil)
        ctx.update(image, None, invalidates=TONE_INVALIDATES)
        return Report(self.name, ran=True, notes=note)


class SceneDenoise(Module):
    """无人脸照片：整图学习型降噪，放在提亮之前（噪声随后会被增益放大）。"""
    name, region, order, bands = "降噪", "global", 18, ("high",)

    def plan(self, ctx):
        if ctx.opts.auto.denoise <= 0:
            return None, "关闭"
        return _scene_only(ctx, "人像在精修后只对非皮肤降噪")

    def run(self, ctx, req):
        from scripts.learned_denoise import smart_denoise
        image, how = smart_denoise(ctx.image, None, ctx.opts.auto.gpu, ctx.iso)
        ctx.update(image, "denoise", invalidates=TONE_INVALIDATES)
        return Report(self.name, ran=True, notes=how)


class NonSkinDenoise(Module):
    """人像：精修之后对非皮肤区域（衣服、背景）降噪，皮肤已由磨皮处理。"""
    name, region, order, bands = "非皮肤降噪", "non-skin", 70, ("high",)

    def plan(self, ctx):
        if not ctx.faces:
            return None, "没有人脸"
        return Request(goal="technical_cleanup"), ""

    def run(self, ctx, req):
        from scripts.learned_denoise import smart_denoise
        if ctx.popts.denoise <= 0:
            ctx.update(ctx.image, "denoise")
            return Report(self.name, ran=False, skipped="降噪关闭")
        p = ctx.person
        image, how = smart_denoise(ctx.image, p.full(p.masks.skin, ctx.image.shape), ctx.popts.gpu, ctx.iso)
        ctx.update(image, "denoise")
        return Report(self.name, ran=True, notes=how)


class Sharpen(Module):
    name, region, order, bands = "锐化", "non-skin", 80, ("high",)

    def plan(self, ctx):
        return _face_only(ctx, "场景链路不锐化（1.x 行为）")

    def run(self, ctx, req):
        from scripts.sharpening import sharpen_image
        p = ctx.person
        image = sharpen_image(ctx.image, p.full(p.masks.skin, ctx.image.shape), p.scale, ctx.popts.sharpen,
                              hard_skin=ctx.profile != "r10")
        ctx.update(image, "sharpen")
        return Report(self.name, ran=True, notes=f"强度 {ctx.popts.sharpen}")


# ---------------- 场景影调 ----------------
class Exposure(Module):
    name, region, order, channels = "曝光", "global", 20, ("L",)

    def run(self, ctx, req):
        from scripts.analyze import analyze
        from scripts.exposure import apply_gain, gain_map
        if ctx.faces:
            o = ctx.popts
            if ctx.profile == "r10":
                ctx.update(ctx.image, "exposure")
                return Report(self.name, ran=False, skipped="R10 配置，保持 1.2 设定")
            if o.exposure <= 0:
                ctx.update(ctx.image, "exposure")
                return Report(self.name, ran=False, skipped="关闭")
            an = analyze(ctx.image, ctx.path, faces=ctx.faces)
            gain, k, why = gain_map(ctx.image, an, o.target, o.exposure, hi="luma")
            image = apply_gain(ctx.image, gain) if abs(k - 1) > 0.02 else ctx.image
        else:
            o = ctx.opts.auto
            an = analyze(ctx.image, ctx.path, use_landmarks=not o.no_landmarks)
            ctx.put("analysis", an)
            ctx.scene  # 场景类型在提亮前判断（与 1.x 一致）
            if o.exposure <= 0:
                return Report(self.name, ran=False, skipped="关闭")
            gain, k, why = gain_map(ctx.image, an, o.target, o.exposure, shadows=o.shadows)
            image = apply_gain(ctx.image, gain)
            ctx.put("gain", gain)
        ctx.update(image, "exposure", invalidates=TONE_INVALIDATES)
        return Report(self.name, ran=True, notes=f"×{k:.2f}（{why}）", channels=("L",))


class Dehaze(Module):
    name, region, order = "去雾", "global", 30

    def plan(self, ctx):
        if ctx.opts.auto.dehaze <= 0:
            return None, "关闭"
        if ctx.ev100 is not None and ctx.ev100 < 10:
            return None, f"EV {ctx.ev100:.1f}，不是白天户外"
        return Request(goal="dehaze"), ""

    def run(self, ctx, req):
        from scripts.dehaze import dehaze
        from scripts.sky import analyze_sky
        if not analyze_sky(ctx.image).found:
            return Report(self.name, ran=False, skipped="没有天空（室内或近景）")
        image, how = dehaze(ctx.image, ctx.opts.auto.dehaze)
        if ctx.faces:
            image, how = protect_person(ctx, ctx.image, image), how + "（人物区域不动）"
        ctx.update(image, "dehaze")
        return Report(self.name, ran=True, notes=how)


class Sky(Module):
    name, region, order = "天空", "sky", 32

    def plan(self, ctx):
        if ctx.opts.auto.sky <= 0:
            return None, "关闭"
        return Request(goal="sky"), ""

    def run(self, ctx, req):
        from scripts.sky import analyze_sky, recover_sky
        o = ctx.opts.auto
        image, sky_w, how = recover_sky(ctx.image, analyze_sky(ctx.image), o.sky, o.sky_darken)
        if ctx.faces and sky_w is not None:
            image, how = protect_person(ctx, ctx.image, image), how + "（人物区域不动）"
        ctx.put("sky_weight", sky_w)
        if sky_w is None:
            return Report(self.name, ran=False, skipped=how)
        ctx.update(image, "sky")
        return Report(self.name, ran=True, notes=how)


class NightLight(Module):
    name, region, order = "夜景光线", "global", 34

    def plan(self, ctx):
        if ctx.opts.auto.night <= 0:
            return None, "关闭"
        if ctx.faces:
            return None, "有人脸，走人像链路"
        return (Request(goal="night"), "") if ctx.scene.kind == "night" else (None, f"不是夜景（{ctx.scene.label}）")

    def run(self, ctx, req):
        from scripts.scene_adjust import night_light
        image, how = night_light(ctx.image, ctx.opts.auto.night)
        ctx.update(image, "night")
        return Report(self.name, ran=True, notes=how)


class Glow(Module):
    name, region, order = "晚霞", "sky", 35

    def plan(self, ctx):
        if ctx.opts.auto.glow <= 0:
            return None, "关闭"
        if ctx.faces:
            return None, "有人脸，走人像链路"
        return (Request(goal="glow"), "") if ctx.scene.glow > 0 else (None, "没有晚霞")

    def run(self, ctx, req):
        from scripts.scene_adjust import enhance_glow
        image, how = enhance_glow(ctx.image, ctx.scene, ctx.opts.auto.glow)
        ctx.update(image, "glow")
        return Report(self.name, ran=True, notes=how)


class Light(Module):
    """人像光影增强：只塑脸的低频明暗、背景压暗、白天高光柔辉（人像配置）。"""
    name, region, order, channels, bands = "光影", "person", 50, ("L",), ("low",)

    def run(self, ctx, req):
        from scripts.light_enhance import light_enhance
        if not ctx.faces or ctx.profile == "r10" or ctx.popts.light <= 0:
            ctx.update(ctx.image, "light")
            return Report(self.name, ran=False, skipped="没有人脸" if not ctx.faces else "R10 配置或关闭")
        p = ctx.person
        image, note = light_enhance(ctx.image, p.masks, p.roi, ctx.ev100, ctx.popts.light)
        ctx.update(image, "light")
        return Report(self.name, ran=True, notes=note.lstrip("，"))

    def plan(self, ctx):
        return _face_only(ctx)


# ---------------- 色彩 ----------------
class Color(Module):
    """白平衡 + 色阶（保肤色彩度），人像配置再做肤色色相校正。"""
    name, region, order, channels = "色彩", "global", 38, ("L", "ab")

    def plan(self, ctx):
        return _face_only(ctx, "场景链路由夜景/晚霞/调色模块处理")

    def run(self, ctx, req):
        from scripts.color_correction import color_correct, skin_hue
        p, o = ctx.person, ctx.popts
        face_px = p.crop(ctx.image)[p.masks.face_hard > 0]
        image = color_correct(ctx.image, o.color, face_px, 0.85 if ctx.profile == "r10" else 0.97)
        note = ""
        if ctx.profile != "r10" and o.skin_hue > 0:
            image, note = skin_hue(image, p.masks, p.roi, o.skin_hue)
        ctx.update(image, "color")
        return Report(self.name, ran=True, notes=f"强度 {o.color}{note}")


class Grade(Module):
    name, region, order = "风格调色", "global", 36

    def plan(self, ctx):
        g = ctx.opts.auto.grade
        if ctx.faces:
            return None, "有人脸，走人像链路"
        if g == "auto":
            g = "orange-teal" if ctx.scene.kind == "night" else "none"
        return (Request(goal=g), "") if g == "orange-teal" else (None, "默认不调色（2.0：不默认改氛围）")

    def run(self, ctx, req):
        from scripts.grade import orange_teal
        image, how = orange_teal(ctx.image, ctx.opts.auto.grade_strength)
        ctx.update(image, "grade")
        return Report(self.name, ran=True, notes=how)


# ---------------- 人像 ----------------
class Blemish(Module):
    name, region, order, bands = "瑕疵", "skin", 35, ("mid", "high")

    def plan(self, ctx):
        return _face_only(ctx)

    def run(self, ctx, req):
        from dataclasses import replace
        from scripts.blemish import remove_blemishes
        p, o = ctx.person, ctx.popts
        crop, found = remove_blemishes(p.crop(ctx.image), p.masks, p.faces, p.scale, o.blemish, o.freckles, o.moles)
        x0, y0 = p.roi[:2]
        ctx.put("blemishes", [replace(b, center=(b.center[0] + x0, b.center[1] + y0)) for b in found])
        ctx.update(p.paste(ctx.image, crop), "blemish")
        n = sum(b.weight > 0 for b in found)
        return Report(self.name, ran=True, notes=f"检出 {len(found)}，处理 {n}")


class Retouch(Module):
    """补光、匀肤、Dodge & Burn、磨皮、局部肤质平衡、美白（顺序与 1.x 相同）。"""
    name, region, order = "精修", "skin", 45

    def plan(self, ctx):
        return _face_only(ctx)

    def run(self, ctx, req):
        from main import whiten_amount
        from scripts.dodge_burn import dodge_burn
        from scripts.exposure import face_fill
        from scripts.frequency_separation import frequency_separation
        from scripts.skin_smoothing import skin_smoothing, texture_balance
        from scripts.whitening import whiten_skin
        p, o, r10 = ctx.person, ctx.popts, ctx.profile == "r10"
        m, notes = p.masks, []
        if not r10 and o.fill > 0:
            image, n = face_fill(ctx.image, m, p.roi, o.fill)
            ctx.update(image, None)
            notes.append(n.lstrip("，"))

        def step(fn, stage):
            ctx.update(p.paste(ctx.image, fn(p.crop(ctx.image))), stage)

        step(lambda c: frequency_separation(c, m, p.scale, o.even, 8.0 if r10 else 20.0), "even")
        step(lambda c: dodge_burn(c, m, p.faces, p.scale, o.dodge_burn, o.eye), "dodge_burn")
        step(lambda c: skin_smoothing(c, m, p.scale, o.smooth, o.grain), None)
        if not r10 and o.balance > 0:
            crop, n = texture_balance(p.crop(ctx.image), m, p.scale, o.balance)
            ctx.update(p.paste(ctx.image, crop), None)
            notes.append(n.lstrip("，"))
        ctx.update(ctx.image, "smooth")
        step(lambda c: whiten_skin(c, m.tone, whiten_amount(c, m, o.whiten)), "whiten")
        return Report(self.name, ran=True, notes="；".join(x for x in notes if x) or "完成")


class Slim(Module):
    name, region, order, geometry = "瘦脸", "person", 90, True

    def plan(self, ctx):
        if ctx.faces and ctx.popts.slim > 0:
            return Request(goal="slim"), ""
        return None, "关闭" if ctx.faces else "没有人脸"

    def run(self, ctx, req):
        from scripts.warping import slim_face
        p = ctx.person
        ctx.update(p.paste(ctx.image, slim_face(p.crop(ctx.image), p.faces, ctx.popts.slim)), "slim")
        return Report(self.name, ran=True, notes=f"强度 {ctx.popts.slim}")


def default_modules():
    return [Geometry(), Deveil(), SceneDenoise(), Exposure(), Dehaze(), Sky(), NightLight(), Glow(), Grade(),
            Blemish(), Color(), Retouch(), Light(), NonSkinDenoise(), Sharpen(), Slim()]
