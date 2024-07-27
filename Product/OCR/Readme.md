### README.md

# OCR自动化脚本 | OCR Automation Script

该仓库包含一个使用百度云API进行光学字符识别（OCR）的脚本 `OCR.py`。该脚本处理剪贴板中的图像并自动将识别出的文本保存到剪贴板，提供了一个非常方便的工作流程。  
This repository contains the `OCR.py` script, which leverages Baidu Cloud API for Optical Character Recognition (OCR). The script processes images from the clipboard and automatically saves the recognized text back to the clipboard, offering a highly convenient workflow.

## 功能 | Features

- **剪贴板图像处理**: 处理两种类型的剪贴板图像数据：  
  **Clipboard Image Processing**: Handles two types of clipboard image data:
   1. 直接复制的图像文件。  
      Directly copied image files.
   2. 复制的图像对象。  
      Copied image objects.

- **自动文本识别**: 执行时，脚本处理剪贴板中的图像并将识别出的文本保存到剪贴板。  
  **Automatic Text Recognition**: Upon execution, the script processes the image from the clipboard and saves the recognized text to the clipboard.

- **便捷使用**: 简化了从图像中提取文本的过程，非常方便用户使用。  
  **Convenient Usage**: Simplifies the process of text extraction from images, making it very convenient for users.

## 使用方法 | Usage

1. 将图像文件或图像对象复制到剪贴板。  
   Copy an image file or image object to the clipboard.
2. 运行 `OCR.py` 脚本。  
   Run the `OCR.py` script.
3. 识别出的图像文本将保存在剪贴板中。  
   The recognized text from the image will be available in the clipboard.

### 实现方法 | Implementation

#### 处理图像文件 | Processing Image Files

当复制一个图像文件到剪贴板时，脚本会读取该文件并发送到百度云OCR API进行处理。识别出的文本会被保存到剪贴板中供后续使用。  
When an image file is copied to the clipboard, the script reads the file and sends it to the Baidu Cloud OCR API for processing. The recognized text is then saved to the clipboard for subsequent use.

#### 处理图像对象 | Processing Image Objects

当复制一个图像对象到剪贴板时，脚本会从剪贴板中提取图像数据并发送到百度云OCR API进行处理。识别出的文本同样会被保存到剪贴板中。  
When an image object is copied to the clipboard, the script extracts the image data from the clipboard and sends it to the Baidu Cloud OCR API for processing. The recognized text is similarly saved to the clipboard.

享受这个简单高效的自动OCR处理脚本带来的便利吧！  
Enjoy the convenience of automatic OCR processing with this simple and efficient script!
