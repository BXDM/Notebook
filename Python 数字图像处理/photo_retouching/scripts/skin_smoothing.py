import cv2

def skin_smoothing(image):
    # 使用双边滤波进行皮肤平滑
    smoothed = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    return smoothed
