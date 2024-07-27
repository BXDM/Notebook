# Image Background Remover

This project is designed to remove the background from images using the `rembg` library and save the processed images. It supports various image formats including PNG, JPG, JPEG, and WEBP.

## Features

- Remove background from images
- Crop images to remove blank areas
- Process images in bulk
- Handle PNG files specifically to retain the original filename

## Requirements

- Python 3.12.4
- `rembg` library
- `Pillow` library
- `tqdm` library

## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required libraries:
    ```sh
    pip install rembg pillow tqdm
    ```

## Usage

1. Place the images you want to process in a folder.

2. Update the `folder` variable in the script to point to your folder:
    ```python
    folder = r"C:\Users\BXDM\Desktop\Image Mapping"
    ```

3. Run the script:
    ```sh
    python remove_background.py
    ```