import fitz  # PyMuPDF
import io
import os
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import re

B_type_list = []

# pdf file type:
#     A: text 
#     B: image


def has_extractable_text(doc):
    page = doc.load_page(0)                        # 取 pdf file 的第一頁來判斷它是何種檔案
    text = page.get_text("text")                   # 讀取文本
     
    if text.strip():                               # 判斷有沒有讀到文本
        return True                                # True: A
    else:
        return False                               # False: B
    

def pdf_to_text(i, pdf_path, output_txt_path):
    doc = fitz.open(pdf_path)                      # 開啟 PDF 文件

    if has_extractable_text(doc):                  # A type pdf file
        A_type_converison(doc, output_txt_path)
    else:                                          # B type pdf file
        B_type_list.append(i)
        B_type_converison(pdf_path, output_txt_path)


def A_type_converison(doc, output_txt_path):
    extracted_text = ""                                              # 初始化空的字串來存放文本

    for page_num in range(len(doc)):                                 # 遍歷每個頁面，提取文本
        page = doc.load_page(page_num)                               # 讀取該頁面
        text = page.get_text("text")                                 # 提取該頁的文本，只提取純文字內容
        extracted_text += text + "\n"                                # 將文本累積到一個字符串中
    
    # 去除頁碼，例如 "第 X 頁，共 Y 頁" 或類似的頁碼格式
    extracted_text = re.sub(r'第\s*\d+\s*頁[\s，、]*共\s*\d+\s*頁', '', extracted_text)

    # 去除常見的條款或項目編號
    extracted_text = re.sub(r'^\s*[一二三四五六七八九十]{1,2}\s*[、．.：:]', '', extracted_text, flags=re.MULTILINE)  # 去除「一、二、三…」
    extracted_text = re.sub(r'^\s*\d+\s*[、．.：:]', '', extracted_text, flags=re.MULTILINE)  # 去除「1、2、3…」

    extracted_text = re.sub(r'\([一二三四五六七八九十]{1,2}\)', '', extracted_text)# 去除半形括號
    extracted_text = re.sub(r'\(\d+\)', '', extracted_text)

    extracted_text = re.sub(r'（[一二三四五六七八九十]+）', '', extracted_text)  # 去除全形括號
    extracted_text = re.sub(r'（\d+）', '', extracted_text)  

    extracted_text = re.sub(r'~\d+~', '', extracted_text)  # 匹配並刪除 ~59~ 類似格式
    extracted_text = re.sub(r'-\s*\d+\s*-', '', extracted_text)  # 匹配並刪除 - 5 - 類似格式

    # 去除金額（匹配格式如 $ 1,789 或 123,456 或 56,485,523 的數字）
    extracted_text = re.sub(r'\$?\s*\d{1,3}(?:,\d{3})+', '', extracted_text)

    # 去除單獨的金額（匹配格式如 $ 123 或 $ 12.15 或 $545）
    extracted_text = re.sub(r'\$\s*\d+(\.\d{1,2})?', '', extracted_text)

    # 去除所有空格和換行符
    extracted_text = re.sub(r'\s+', '', extracted_text)

    with open(output_txt_path, 'w', encoding='utf-8') as txt_file:   # 將提取的文本寫入到 txt 檔
        txt_file.write(extracted_text)

    print(f"TEXT: conversion done, the txt file is saved at {output_txt_path}")


def B_type_converison(pdf_path, output_txt_path):

    # 初始化空的字串來存放文本
    extracted_text = ""                                              

    # 將 PDF 文件每頁轉換為圖片
    pages = convert_from_path(pdf_path, dpi=300)

    # 去除紅色印章
    for i, page in enumerate(pages):

        # 將圖片轉換為 RGB 模式，並存儲為臨時文件，便於使用 OpenCV 進行處理
        page_rgb = page.convert('RGB')
        page_np = np.array(page_rgb)  # 將 PIL Image 轉換為 numpy 數組
        page_bgr = cv2.cvtColor(page_np, cv2.COLOR_RGB2BGR)  # 轉換為 OpenCV 的 BGR 色彩模式

        # 提取紅色通道
        _, _, R_channel = cv2.split(page_bgr)

        # 對紅色通道進行閾值處理，去除紅色區域
        _, RedThresh = cv2.threshold(R_channel, 150, 255, cv2.THRESH_BINARY)
        
        # 可選：進行膨脹或侵蝕操作來清理邊緣
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        erode = cv2.erode(RedThresh, element)

        # 使用遮罩將紅色區域變為白色
        page_bgr[erode == 255] = [255, 255, 255]  # 將紅色區域設置為白色

        # 將處理後的圖片轉回 PIL 格式
        processed_image = Image.fromarray(cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB))

        # 增強對比度和亮度
        enhancer = ImageEnhance.Contrast(processed_image)
        processed_image = enhancer.enhance(2)
        enhancer = ImageEnhance.Brightness(processed_image)
        processed_image = enhancer.enhance(1.5)

        # 轉為黑白模式
        processed_image = processed_image.convert('L')

        # 進行 OCR
        ocr_text = pytesseract.image_to_string(processed_image, lang='chi_tra')
        extracted_text += ocr_text + "\n"
  
    # 去除頁碼，例如 "第 X 頁，共 Y 頁" 或類似的頁碼格式
    extracted_text = re.sub(r'第\s*\d+\s*頁[\s，、]*共\s*\d+\s*頁', '', extracted_text)

    # 去除常見的條款或項目編號
    extracted_text = re.sub(r'^\s*[一二三四五六七八九十]{1,2}\s*[、．.：:]', '', extracted_text, flags=re.MULTILINE)  # 去除「一、二、三…」
    extracted_text = re.sub(r'^\s*\d+\s*[、．.：:]', '', extracted_text, flags=re.MULTILINE)  # 去除「1、2、3…」
    extracted_text = re.sub(r'\([一二三四五六七八九十]{1,2}\)', '', extracted_text)# 去除半形括號
    extracted_text = re.sub(r'\(\d+\)', '', extracted_text)

    extracted_text = re.sub(r'（[一二三四五六七八九十]+）', '', extracted_text)  # 去除全形括號
    extracted_text = re.sub(r'（\d+）', '', extracted_text)  

    extracted_text = re.sub(r'~\d+~', '', extracted_text)  # 匹配並刪除 ~59~ 類似格式
    extracted_text = re.sub(r'-\s*\d+\s*-', '', extracted_text)  # 匹配並刪除 - 5 - 類似格式

    # 635.557 應為 635,557
    extracted_text = re.sub(r'(\d{1,3})\.(\d{3})\b', r'\1,\2', extracted_text)

    # 去除金額（匹配格式如 $ 1,789 或 123,456 或 56,485,523 的數字）
    extracted_text = re.sub(r'\$?\s*\d{1,3}(?:,\d{3})+', '', extracted_text)

    # 去除單獨的金額（匹配格式如 $ 123 或 $ 12.15 或 $545）
    extracted_text = re.sub(r'\$\s*\d+(\.\d{1,2})?', '', extracted_text)

    # 去除所有空格和換行符
    extracted_text = re.sub(r'\s+', '', extracted_text)

    # 將提取的文本寫入到 txt 檔
    with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(extracted_text)
    
    print(f"IMAGE: conversion done, the txt file is saved at {output_txt_path}")



"""for i in range(0,1035):
    pdf_path = "競賽資料集\\reference\\finance\\" + str(i) + ".pdf"    # .pdf 檔位置
    out_path = "output_txt\\finance\\"              # 輸出 .txt 檔位置
    os.makedirs(out_path, exist_ok=True)
    output_txt_path = out_path + str(i) + ".txt"

    pdf_to_text(i, pdf_path, output_txt_path)
    

# record B type pdf file
image_pdf_path = "output_txt\\finance_image_pdf_file.txt"
text = ""
for i in range(len(B_type_list)):
    text = text + str(B_type_list[i]) + ".pdf" + '\n'

with open(image_pdf_path, 'w', encoding='utf-8') as txt_file:
    txt_file.write(text)"""


for i in range(1,644):
    pdf_path = "競賽資料集\\reference\\insurance\\" + str(i) + ".pdf"    # .pdf 檔位置
    out_path = "output_txt\\insurance\\"              # 輸出 .txt 檔位置
    os.makedirs(out_path, exist_ok=True)
    output_txt_path = out_path + str(i) + ".txt"

    pdf_to_text(i, pdf_path, output_txt_path)
    

# record B type pdf file
image_pdf_path = "output_txt\\insurance_image_pdf_file.txt"
text = ""
for i in range(len(B_type_list)):
    text = text + str(B_type_list[i]) + ".pdf" + '\n'

with open(image_pdf_path, 'w', encoding='utf-8') as txt_file:
    txt_file.write(text)

    
    
    
    
    
    