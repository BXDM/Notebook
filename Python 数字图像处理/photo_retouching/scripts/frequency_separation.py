import cv2
import numpy as np

def frequency_separation(image):
    # 转换到浮点数
    image = np.float32(image) / 255.0
    low_pass = cv2.GaussianBlur(image, (21, 21), 3)
    high_pass = image - low_pass
    high_pass += 0.5  # 移动到0.0到1.0的范围
    high_pass = np.clip(high_pass, 0, 1)
    result = cv2.addWeighted(low_pass, 0.5, high_pass, 0.5, 0)
    return (result * 255).astype(np.uint8)
