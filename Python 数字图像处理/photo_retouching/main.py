# Create an empty __init__.py file in the scripts folder
# scripts/__init__.py

# Adjust the import statements in main.py
import os
from scripts.preprocess import preprocess_image
from scripts.beautifygan import beautify_image
from scripts.frequency_separation import frequency_separation
from scripts.deepwarp import warp_image
from scripts.deepfillv2 import inpaint_image
from scripts.deblurgan import sharpen_image
from scripts.color_correction import color_correct
from scripts.save_image import save_image

def process_image(image_path, output_path):
    # 预处理
    image = preprocess_image(image_path)

    # 磨皮处理
    image = beautify_image(image)

    # 高低频分离处理
    image = frequency_separation(image)

    # 瘦身处理
    image = warp_image(image)

    # 发缝修复和祛痣处理
    image = inpaint_image(image)

    # 锐化处理
    image = sharpen_image(image)

    # 色彩校正
    image = color_correct(image)

    # 保存处理后的图像
    save_image(image, output_path)

def main():
    input_dir = 'data/raw/'
    output_dir = 'output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".raw"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            process_image(input_path, output_path)
            print(f"Processed {filename}")

if __name__ == "__main__":
    main()