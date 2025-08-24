import cv2
import numpy as np

def warp_image(image):
    # 使用仿射变换进行简单的瘦身处理
    rows, cols, _ = image.shape
    src_points = np.float32([[0, 0], [cols-1, 0], [0, rows-1]])
    dst_points = np.float32([[0, 0], [int(0.9 * (cols-1)), 0], [0, rows-1]])
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    warped = cv2.warpAffine(image, affine_matrix, (cols, rows))
    return warped
