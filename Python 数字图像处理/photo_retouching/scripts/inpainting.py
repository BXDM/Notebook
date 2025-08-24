import cv2
import numpy as np

def inpaint_image(image):
    # 创建一个掩码，标记需要修复的区域
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[50:100, 100:150] = 255  # 示例位置
    inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return inpainted
