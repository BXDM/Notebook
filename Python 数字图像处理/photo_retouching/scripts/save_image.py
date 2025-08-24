import cv2
from PIL import Image

def save_image(image, output_path):
    image.save(output_path, quality=95)
