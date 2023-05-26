from csv import excel
import cv2
import numpy as np
import re
import os
import pandas as pd
import pytesseract
from keras.models import model_from_json
from keras import backend
from process_data import correct_skew, get_lines
import tkinter as tk
from tkinter import filedialog

"""
load_model(): load và return model RCNN CTC 
doc_bang_diem(): đầu vào là đường dẫn của MỘT ảnh bảng điểm và đầu ra là mã lớp thi là tên file excel, dataframe dữ liệu đọc được. 
recognize_lopthi(): chọn các ảnh của cùng một mã lớp thi và nhận diện 
recognize_folder(): đọc cả 1 folder
"""

characters = u"0123456789.n"


# Kí tự sang số
def label_to_num(label):
    num = []
    for character in label:
        num.append(characters.find(character))
    return np.array(num)


# Số sang kí tự
def num_to_label(num):
    label = ""
    for ch in num:
        if ch == -1:
            break
        else:
            label += characters[ch]
    return label


def load_model():
    with open('He ho tro nhap diem tu dong\Model\model_CRNNCTC_final.json', 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights('He ho tro nhap diem tu dong\Model\model_CRNNCTC_final.h5')
    return model


def doc_bang_diem(file_url, model):
    img = cv2.imread(file_url, cv2.IMREAD_GRAYSCALE)
    img_filename = file_url.split("/")[-1]
    print(img_filename)
    img = correct_skew(img)
    height, width = img.shape
    horizontal_coor, vertical_coor = get_lines(img)

    # Đọc mã lớp
    img_malop = img[horizontal_coor[0] - round(height * 0.05):horizontal_coor[0] - round(height * 0.017),
                vertical_coor[3]: vertical_coor[6]]
    text = pytesseract.image_to_string(img_malop)
    malop = re.search(r'\d\d\d\d\d\d', text)
    if malop:
        malop = malop.group()
    else:
        malop = ' ERROR ' + img_filename
    excel_filename = str(malop) + '.xlsx'
    print('MA LOP:', excel_filename)

    # Đọc mã số sinh viên
    list_MSSV = []
    for i in range(1, len(horizontal_coor) - 1):
        img_mssv = img[horizontal_coor[i]: horizontal_coor[i + 1], vertical_coor[1]: vertical_coor[2]]
        mssv = pytesseract.image_to_string(img_mssv)
        mssv = re.search(r'\d\d\d\d\d\d\d\d', mssv)
        if mssv:
            mssv = mssv.group()
        else:
            mssv = pytesseract.image_to_string(img_mssv)
        list_MSSV.append(mssv)

    # Đọc điểm
    list_Diem = []
    for i in range(1, len(horizontal_coor) - 1):
        img_diem = img[horizontal_coor[i]: horizontal_coor[i + 1] + round(height * 0.004),
                   vertical_coor[4]: vertical_coor[5]]
        img_diem = cv2.resize(img_diem, (100, 40))
        img_diem = img_diem / 255.0
        pred = model.predict(img_diem.reshape(1, 40, 100, 1))
        decoded = backend.get_value(
            backend.ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=True)[0][0])
        diem = num_to_label(decoded[0])
        diem = diem.replace("n", "")
        list_Diem.append(diem)

    df = pd.DataFrame(columns=['MSSV', 'Điểm'])
    list_MSSV = pd.Series(list_MSSV)
    list_Diem = pd.Series(list_Diem)
    df['MSSV'] = list_MSSV.values
    df['Điểm'] = list_Diem.values
    return excel_filename, df


def recognize_lopthi(model):
    root = tk.Tk()
    root.withdraw()
    print("Get input image path")
    files_url = list(filedialog.askopenfilenames())
    print("Get output folder path")
    output_folder = filedialog.askdirectory()
    for file_url in files_url:
        img_filename = file_url.split('/')[-1]
        print("Read ", file_url)
        excel_filename, df = doc_bang_diem(file_url, model)
        print("Excel file = ", excel_filename)
        if os.path.exists(os.path.join(output_folder, excel_filename)) is True:
            old_df = pd.read_excel(os.path.join(output_folder, excel_filename))
            old_df = pd.DataFrame(old_df)
            new_df = old_df.append(df)
            new_df.to_excel(os.path.join(output_folder, excel_filename), index=False)
        else:
            df.to_excel(os.path.join(output_folder, excel_filename), index=False)
        print('Write successfully ', img_filename)


def recognize_folder(model):
    print("Get input folder path: ")
    input_folder = filedialog.askdirectory()
    print("Get output folder path")
    output_folder = filedialog.askdirectory()
    for img_filename in os.listdir(input_folder):
        print('Read ', img_filename)
        excel_filename, df = doc_bang_diem(os.path.join(input_folder, img_filename), model)
        if os.path.exists(os.path.join(output_folder, excel_filename)) is True:
            old_df = pd.read_excel(os.path.join(output_folder, excel_filename))
            old_df = pd.DataFrame(old_df)
            new_df = old_df.append(df)
            new_df.to_excel(os.path.join(output_folder, excel_filename), index=False)
        else:
            df.to_excel(os.path.join(output_folder, excel_filename), index=False)
        print('Write successfully ', img_filename)
