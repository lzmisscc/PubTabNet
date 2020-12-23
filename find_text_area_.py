# coding:utf8
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import average


def preprocess(gray, save_mid_image=True, imagePath=None):
    # 1. Sobel算子，x方向求梯度
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    # 2. 二值化
    ret, binary = cv2.threshold(
        sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    h = []
    # print(binary)
    for l in binary:
        l = np.count_nonzero(l)
        h.append(l)
    # print(h)
    h = np.array(h)
    # print(f"水平分布：{(h - np.average(h))>0}")
    # 3. 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))

    # 4. 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=3)

    # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    erosion = cv2.erode(dilation, element1, iterations=3)

    # 6. 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations=10000)
    h = []
    # print(dilation2)
    for l in dilation2:
        l = np.count_nonzero(l)
        h.append(l)
    # print(f"水平分布：{h}")
    left, right = 0, 0
    while h and h[0] == 0:
        left += 1
        h.pop(0)
    while h and h[-1] == 0:
        right += 1
        h.pop()
    hs = []
    while h:
        if h[0] !=0:
            left += 1 
            h.pop(0)
        else:
            if left >= 5:
                hs.append(left)
            while h and h[0] == 0:
                left += 1
                h.pop(0)
            # hs.append(left)
        

    # print(imagePath,)
    # print("图像SIZE：", gray.shape, "是否多行:", 0 in h)
    # print("打印位置:", hs)
    # 7. 存储中间图片
    if save_mid_image:
        cv2.imwrite("test_result_images/binary.png", binary)
        cv2.imwrite("test_result_images/dilation.png", dilation)
        cv2.imwrite("test_result_images/erosion.png", erosion)
        cv2.imwrite("test_result_images/dilation2.png", dilation2)
    # if 0 in h:
    return hs


def findTextRegion(org, img, source,):
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


def detect(img, imagePath):
    # 1.  转化成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 形态学变换的预处理，得到可以查找矩形的图片
    _h = preprocess(gray, imagePath=imagePath)
    if not _h:
        return []
    # 3. 查找和筛选文字区域
    # bbox = findTextRegion(gray, dilation, source=img)
    h, w = gray.shape
    _h = [0] + _h + [h]
    bboxes = []
    for i in range(len(_h)-1):
        bboxes.append([0, _h[i], w, _h[i+1]])

    return bboxes


if __name__ == '__main__':
    from PIL import Image, ImageDraw
    from math import ceil
    import glob
    import os.path as osp
    # 读取文件
    # imagePath = "./data/val/PMC5664063_009_00_00004.png"  # sys.argv[1]
    # print(imagePath)
    # img = cv2.imread(imagePath)
    # h,w,c = img.shape
    # bbox = detect(img, imagePath=imagePath)
    # exit(0)
    for imagePath in glob.glob("table_ocr/data/val/PMC5461533_016_00*.png"):
        print(imagePath)
        img = cv2.imread(imagePath)
        h, w, c = img.shape
        # if h > 15:
        print(imagePath, h)
        bbox = detect(img, imagePath=imagePath)

        im = Image.fromarray(img)
        d = ImageDraw.Draw(im, )
        for i in bbox:
            # l_t = min(i[0], i[2]), min(i[1], i[3])
            # b_r = max(i[0], i[2]), max(i[1], i[3])
            d.rectangle(i, outline=(0, 255, 100))
        print(bbox)
        im.save(f"test_result_images/{osp.basename(imagePath)}")
