# Copyright (c) Jason
# Date: 2024/7/27
# Python Version: 3.12.4
# IDE: visual studio code


import os
from rembg import remove
from PIL import Image
from tqdm import tqdm

def process_image(image_path, output_path):
    try:
        input_image = Image.open(image_path)
        output_image = remove(input_image)
        
        # Crop the image to remove blank areas
        bbox = output_image.getbbox()
        if bbox:
            output_image = output_image.crop(bbox)
        
        output_image.save(output_path)
        print(f"Processed: {output_path}")
        
        # Delete the original file if it's not the renamed PNG
        if image_path != output_path:
            os.remove(image_path)
            print(f"Deleted original file: {image_path}")
    except PermissionError:
        print(f"Permission denied: {image_path}")
    except Exception as e:
        print(f"Failed to process {image_path}: {e}")

def process_png_image(image_path):
    try:
        # Rename the PNG file temporarily
        temp_path = os.path.splitext(image_path)[0] + '_temp.png'
        os.rename(image_path, temp_path)
        
        # Process the renamed file and save it back to the original name
        process_image(temp_path, image_path)
        
        # Remove the temporary file if it still exists
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        print(f"Failed to process PNG {image_path}: {e}")

def process_all_images(folder):
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    with tqdm(total=len(image_files), desc="Processing Images", unit="image") as pbar:
        for image_file in image_files:
            image_path = os.path.join(folder, image_file)
            if image_file.lower().endswith('.png'):
                process_png_image(image_path)
            else:
                output_path = os.path.splitext(image_path)[0] + '.png'
                process_image(image_path, output_path)
            pbar.update(1)

if __name__ == "__main__":
    folder = r"C:\Users\BXDM\Desktop\Image Mapping"
    process_all_images(folder)