import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from models.openpose import OpenPose

def load_openpose_model():
    # 初始化 OpenPose 模型
    model = OpenPose('models/openpose/model.pth')
    return model

def estimate_pose(image, model):
    # 使用 OpenPose 模型进行姿态估计
    keypoints = model.estimate(image)
    return keypoints

def adjust_body_parts(image, keypoints):
    # 基于关键点对身体部位进行调整（瘦身）
    adjusted_image = image.copy()

    # 示例：调整腰部
    left_waist = keypoints['left_hip']
    right_waist = keypoints['right_hip']

    # 假设调整腰部为缩小10%
    adjustment_factor = 0.9
    adjusted_image[:, left_waist[0]:right_waist[0]] *= adjustment_factor

    # 可以扩展为调整其他部位如手臂和腿部
    return adjusted_image

def process_with_openpose(image_path):
    image = cv2.imread(image_path)
    model = load_openpose_model()
    keypoints = estimate_pose(image, model)
    adjusted_image = adjust_body_parts(image, keypoints)
    return Image.fromarray(adjusted_image)
