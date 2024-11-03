import fitz  # PyMuPDF
import io
import os
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
from PIL import Image, ImageEnhance


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
    

def pdf_to_text(pdf_path, output_txt_path):
    doc = fitz.open(pdf_path)                      # 開啟 PDF 文件

    if has_extractable_text(doc):                  # A type pdf file
        A_type_converison(doc, output_txt_path)
    else:                                          # B type pdf file
        B_type_converison(pdf_path, output_txt_path)


def A_type_converison(doc, output_txt_path):
    extracted_text = ""                                              # 初始化空的字串來存放文本

    for page_num in range(len(doc)):                                 # 遍歷每個頁面，提取文本
        page = doc.load_page(page_num)                               # 讀取該頁面
        text = page.get_text("text")                                 # 提取該頁的文本，只提取純文字內容
        extracted_text += text + "\n"                                # 將文本累積到一個字符串中

    with open(output_txt_path, 'w', encoding='utf-8') as txt_file:   # 將提取的文本寫入到 txt 檔
        txt_file.write(extracted_text)

    print(f"TEXT: conversion done, the txt file is saved at {output_txt_path}")


def B_type_converison(pdf_path, output_txt_path):

    # 初始化空的字串來存放文本
    extracted_text = ""                                              

    # 將 PDF 文件每頁轉換為圖片
    pages = convert_from_path('1.pdf', dpi=300)

    # 處理每一頁圖片並進行 OCR
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

        # 顯示處理後的圖片（可選）
        #processed_image.show()

        # 進行 OCR
        ocr_text = pytesseract.image_to_string(processed_image, lang='chi_tra')
        extracted_text += ocr_text + "\n"

        #print(f"Page {i + 1}:\n{text}\n")   

         # 將提取的文本寫入到 txt 檔
    with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(extracted_text)
    
    print(f"IMAGE: conversion done, the txt file is saved at {output_txt_path}")


for i in range(100):
    pdf_path = "./競賽資料集/reference/finance/" + str(i) + ".pdf"    # .pdf 檔位置
    out_path = "./output_txt/finance/"                               # 輸出 .txt 檔位置
    os.makedirs(out_path, exist_ok=True)
    
    output_txt_path = out_path + str(i) + ".txt"
    pdf_to_text(pdf_path, output_txt_path)
    
    
    
    
    
    
    