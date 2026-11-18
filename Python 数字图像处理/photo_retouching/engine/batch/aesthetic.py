"""美学评分：LAION improved-aesthetic-predictor（Apache-2.0，github.com/christophschuhmann/improved-aesthetic-predictor），
OpenAI CLIP ViT-L/14 图像向量（MIT）→ 小 MLP，输出约 1~10 分（训练数据含 AVA 摄影美学评分）。
只用于连拍组内挑最好的一张（同一场景下比构图、姿态、表情），不跨场景比较、不用来判废片。
没选 Aesthetic Predictor V2.5：AGPL-3.0（MKit 是网络服务，接进去有传染问题），而且加载要 trust_remote_code。
"""
import os

import cv2
import numpy as np

WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "models", "laion_aesthetic_l14.pth")
CLIP = "openai/clip-vit-large-patch14"


def _mlp():
    import torch.nn as nn
    return nn.Sequential(nn.Linear(768, 1024), nn.Dropout(0.2), nn.Linear(1024, 128), nn.Dropout(0.2),
                         nn.Linear(128, 64), nn.Dropout(0.1), nn.Linear(64, 16), nn.Linear(16, 1))


def score(images_bgr, batch=8):
    """images_bgr：BGR 图列表（任意尺寸）。返回分数列表；模型不可用时返回 None。"""
    if not os.path.exists(WEIGHTS):
        return None
    import torch
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    clip = CLIPVisionModelWithProjection.from_pretrained(CLIP).eval()
    proc = CLIPImageProcessor.from_pretrained(CLIP)
    head = _mlp()
    sd = torch.load(WEIGHTS, map_location="cpu")
    head.load_state_dict({k.replace("layers.", ""): v for k, v in sd.items()})
    head.eval()
    out = []
    for i in range(0, len(images_bgr), batch):
        rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in images_bgr[i:i + batch]]
        px = proc(images=rgb, return_tensors="pt").pixel_values
        with torch.no_grad():
            e = clip(pixel_values=px).image_embeds
            e = e / e.norm(dim=-1, keepdim=True)
            out.extend(head(e).squeeze(-1).tolist())
    return [float(x) for x in out]
