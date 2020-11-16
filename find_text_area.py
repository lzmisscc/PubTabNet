# coding:utf8
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess(gray, save_mid_image=False):
    # 1. Sobel算子，x方向求梯度
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    # 2. 二值化
    ret, binary = cv2.threshold(
        sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # 3. 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))

    # 4. 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)

    # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    erosion = cv2.erode(dilation, element1, iterations=1)

    # 6. 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations=2)

    # 7. 存储中间图片
    if save_mid_image:
        cv2.imwrite("test_result_images/binary.png", binary)
        cv2.imwrite("test_result_images/dilation.png", dilation)
        cv2.imwrite("test_result_images/erosion.png", erosion)
        cv2.imwrite("test_result_images/dilation2.png", dilation2)

    return dilation2


def findTextRegion(org, img, source):
    region = []

    # 1. 查找轮廓
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bbox = []
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        # area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        bbox.append([x, y, w, h])
    return bbox


def detect(img):
    # 1.  转化成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 形态学变换的预处理，得到可以查找矩形的图片
    dilation = preprocess(gray)

    # 3. 查找和筛选文字区域
    bbox = findTextRegion(gray, dilation, source=img)
    return bbox


if __name__ == '__main__':
    # 读取文件
    imagePath = "table_ocr_dataset/train/PMC1626454_002_00_00006.png"  # sys.argv[1]
    img = cv2.imread(imagePath)
    bbox = detect(img)
    print(bbox)
