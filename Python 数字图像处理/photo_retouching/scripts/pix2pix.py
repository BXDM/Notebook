#!/usr/bin/python3.12
# _*_ coding: utf-8 _*_
# @Time    : 2024/7/30 下午 7:52
# @Author  : Jason
# @IDE     : DataSpell
import torch
from PIL import Image
from torchvision import transforms
from models.pix2pix.model import Pix2Pix

def pix2pix_transform(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    input_image = transform(image).unsqueeze(0)

    model = Pix2Pix()
    model.load_state_dict(torch.load('models/pix2pix/pix2pix.pth'))
    model.eval()

    with torch.no_grad():
        output_image = model(input_image)

    output_image = output_image.squeeze().cpu().numpy()
    output_image = ((output_image + 1) * 127.5).astype(np.uint8)
    return Image.fromarray(output_image)
