import io
import base64
import requests
from PIL import ImageGrab, Image
import pyperclip
import os

# 从百度获取Access Token
def fetch_baidu_token(api_key, secret_key):
    token_url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": api_key,
        "client_secret": secret_key,
    }
    response = requests.post(token_url, params=params)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        raise Exception("无法获取百度OCR token")

# 通过百度OCR识别图像中文本
def perform_ocr_baidu(image_data, token):
    ocr_url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic?access_token={token}"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    img_base64 = base64.b64encode(image_data).decode()
    data = {"image": img_base64}
    response = requests.post(ocr_url, headers=headers, data=data)
    if response.status_code == 200:
        words_result = response.json().get("words_result")
        if words_result:
            return "\n".join([item["words"] for item in words_result])
        else:
            return "未识别到任何文本内容"
    else:
        return "OCR识别失败"

# 处理剪切板内容
def process_clipboard(api_key, secret_key):
    try:
        # 尝试从剪切板获取图像或图像文件
        image = ImageGrab.grabclipboard()
        image_data = None

        if isinstance(image, Image.Image):
            # 如果找到图像对象，将其转换为字节数据，尝试保存为PNG或JPG
            with io.BytesIO() as output:
                image_format = "PNG" if image.mode == "RGBA" else "JPEG"
                image.save(output, format=image_format)
                image_data = output.getvalue()
        elif isinstance(image, list) and len(image) == 1 and isinstance(image[0], str) and os.path.isfile(image[0]):
            # 如果是图像文件，读取文件内容
            with open(image[0], "rb") as file:
                image_data = file.read()

        if image_data:
            # 获取百度token
            token = fetch_baidu_token(api_key, secret_key)
            # 通过百度OCR识别文本
            recognized_text = perform_ocr_baidu(image_data, token)
        else:
            recognized_text = "未识别到图像"
    except Exception as e:
        recognized_text = f"未识别到图像: {str(e)}"

    # 将识别到的文本或者错误信息写入剪切板
    pyperclip.copy(recognized_text)

# 百度OCR API的API Key和Secret Key
API_KEY = "dqaICpYSlRd3ewG3qW2jqAeI"
SECRET_KEY = "OlE7gkG8QnvzqceAzSEWzGaVIOGAQkaz"

# 执行主函数
process_clipboard(API_KEY, SECRET_KEY)