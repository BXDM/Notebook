import cv2
from PIL import Image


def preprocess_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.resize(image,(1024, 1024))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)
