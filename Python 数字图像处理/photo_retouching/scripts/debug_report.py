"""S1 质量检查：保存各阶段输出、蒙版叠加、瑕疵分类标注，以及关键部位 100% 前后对比裁剪。"""
import os
from collections import Counter

import cv2
import numpy as np

from scripts.save_image import save_image

# 关键部位 -> mediapipe 关键点下标
REGIONS = {
    "左脸颊": 50, "右脸颊": 280, "鼻翼": 129, "发际线": 10,
    "眼周": 145, "嘴角": 61, "下颌": 172,
}
KIND_COLORS = {"acne": (0, 0, 255), "spot": (0, 200, 255), "freckle": (255, 200, 0), "mole": (255, 0, 255)}


def mask_overlay(image, masks):
    ov = image.astype(np.float32)
    ov[..., 1] += masks.skin * 70
    ov[..., 0] += masks.face * 50
    p = masks.protect > 0
    ov[p] = ov[p] * 0.5 + np.array([0, 0, 255]) * 0.5
    return np.clip(ov, 0, 255).astype(np.uint8)


def blemish_overlay(image, blemishes):
    ov = image.copy()
    for b in blemishes:
        c = tuple(int(v) for v in b.center)
        r = int(b.radius * 2.5 + 4)
        cv2.circle(ov, c, r, KIND_COLORS[b.kind], 2 if b.weight > 0 else 1)
    return ov


def _label(img, text):
    """左上角写标签（OpenCV 不支持中文，用拼音/英文）。"""
    cv2.putText(img, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(img, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


_NAMES = {"左脸颊": "cheek-L", "右脸颊": "cheek-R", "鼻翼": "nose-wing", "发际线": "hairline",
          "眼周": "under-eye", "嘴角": "mouth-corner", "下颌": "jaw"}


def crops_sheet(before, after, faces, scale):
    """每个部位一行：原图 | 结果，均为 100% 像素。"""
    face = next((f for f in faces if f.points is not None), None)
    if face is None:
        return None
    size = int(max(64, scale * 0.28))
    h, w = before.shape[:2]
    rows = []
    for name, idx in REGIONS.items():
        cx, cy = face.points[idx].astype(int)
        x0, y0 = int(np.clip(cx - size // 2, 0, w - size)), int(np.clip(cy - size // 2, 0, h - size))
        pair = [_label(before[y0:y0 + size, x0:x0 + size].copy(), _NAMES[name] + " before"),
                _label(after[y0:y0 + size, x0:x0 + size].copy(), "after")]
        rows.append(np.hstack(pair + [np.full((size, 4, 3), 255, np.uint8)]))
    # 两列排布
    if len(rows) % 2:
        rows.append(np.zeros_like(rows[0]))
    half = len(rows) // 2
    return np.hstack([np.vstack(rows[:half]), np.vstack(rows[half:])])


def write_debug(out_dir, stem, stages, masks, blemishes, faces, scale, params):
    d = os.path.join(out_dir, stem)
    os.makedirs(d, exist_ok=True)
    names = list(stages)
    for i, name in enumerate(names):
        save_image(stages[name], os.path.join(d, f"{i:02d}_{name}.jpg"))
    original, final = stages[names[0]], stages[names[-1]]
    save_image(mask_overlay(original, masks), os.path.join(d, "mask_overlay.jpg"))
    save_image(blemish_overlay(original, blemishes), os.path.join(d, "blemishes.jpg"))
    sheet = crops_sheet(original, final, faces, scale)
    if sheet is not None:
        save_image(sheet, os.path.join(d, "crops_100pct.png"))
    counts = Counter(b.kind for b in blemishes)
    fixed = Counter(b.kind for b in blemishes if b.weight > 0)
    with open(os.path.join(d, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"蒙版方式: {masks.method}\n人脸: {len(faces)}，参考尺寸(脸宽): {scale:.0f}px\n")
        fh.write("瑕疵（检出/处理）: " + "，".join(f"{k} {counts[k]}/{fixed[k]}" for k in KIND_COLORS) + "\n")
        fh.write("颜色: acne 红, spot 橙, freckle 蓝, mole 紫；粗圈=已处理，细圈=保留\n\n参数:\n")
        for k, v in params.items():
            fh.write(f"  {k} = {v}\n")
        fh.write("\n瑕疵明细 (kind, x, y, 半径, dL, da, 周围起伏, 强度):\n")
        for b in blemishes:
            fh.write(f"  {b.kind:8s} {b.center[0]:7.0f} {b.center[1]:7.0f} {b.radius:5.1f} "
                     f"{b.dL:6.1f} {b.da:6.1f} {b.ring_std:6.1f} {b.weight:.1f}\n")
    return d
