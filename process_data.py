import numpy as np
import cv2
from scipy.ndimage import interpolation as inter


""""
Chứa các hàm tiền xử lý ảnh, bao gồm correct_skew (Chỉnh độ nghiêng của ảnh), get_lines (Nhận diện các dòng 
kẻ bảng điểm 
clustering_coordinates và average hỗ trợ phân cụm để trích rút ra tọa độ dòng kẻ hoàn chỉnh cuối cùng  
"""


def average(list):
    return round(sum(list) / len(list))


def clustering_coordinates(listCoor, label, height, width):
    if label == 0:
        max_distance = round(width * 0.03)  # vertical line
    if label == 1:
        max_distance = round(height * 0.008) # horizontal line
    cluster = []
    curr_point = listCoor[0]
    curr_cluster = [curr_point]
    for x in listCoor[1:]:
        if x <= curr_point + max_distance:
            curr_cluster.append(x)
        else:
            cluster.append(curr_cluster)
            curr_cluster = [x]
        curr_point = x
    cluster.append(curr_cluster)
    mean_cluster = []
    for item in cluster:
        mean_cluster.append(average(item))
    return np.array(mean_cluster)


def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)
    best_angle = angles[scores.index(max(scores))]
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


def img_to_binary(img):
    # blur = cv2.GaussianBlur(img, (3,3), 0)   # Lọc Gauss loại bỏ nhiễu
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  # Phân ngưỡng về ảnh đen trắng
    return thresh


def get_lines(img):
    height, width = img.shape
    img = img_to_binary(img)
    horiznotal_coor = []  # Các tọa độ ngang  y
    vertical_coor = []  # Các tọa độ thẳng  x
    SE_HorizontalSize = round(width * 0.04)
    SE_VerticalSize = round(height * 0.04)
    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (SE_HorizontalSize, 1))
    horizontal_mask = cv2.morphologyEx(img, cv2.MORPH_HITMISS, horizontal_kernel, iterations=1)
    lines = cv2.HoughLinesP(horizontal_mask, 1, np.pi / 180, 0, minLineLength=SE_HorizontalSize, maxLineGap=1)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        horiznotal_coor.append(y1)
        # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, SE_VerticalSize))
    vertical_mask = cv2.morphologyEx(img, cv2.MORPH_HITMISS, vertical_kernel, iterations=1)
    lines = cv2.HoughLinesP(vertical_mask, 1, np.pi / 180, 0, minLineLength=1, maxLineGap=SE_VerticalSize)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        vertical_coor.append(x1)
    horiznotal_coor = np.array(horiznotal_coor)
    vertical_coor = np.array(vertical_coor)
    horiznotal_coor = np.unique(horiznotal_coor)
    vertical_coor = np.unique(vertical_coor)
    new_horizontal_coor = clustering_coordinates(horiznotal_coor, 1, height, width)
    new_vertical_coor = clustering_coordinates(vertical_coor, 0, height, width)

    if new_horizontal_coor[0] < height * 0.1:
        new_horizontal_coor = np.delete(new_horizontal_coor, 0)

    return new_horizontal_coor, new_vertical_coor

