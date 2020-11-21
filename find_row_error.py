from __future__ import barry_as_FLUFL
import enum
import PIL
from shapely.geometry import polygon
import jsonlines
import logging
from PIL import Image, ImageDraw, ImagePath
import os
import numpy as np
import re

# 重新计算每个单元格的bbox，更加精确的bbox文件。

logging.basicConfig(
    level=logging.INFO
)
reader = jsonlines.open('/data/liuzhuang/DataSet/pubtabnet/PubTabNet_2.0.0.jsonl', 'r').iter
data_path = "/data/liuzhuang/DataSet/pubtabnet/"


def get_bbox(points: list) -> list:
    # [[], ...,[]] -> [x_min,y_min,x_max,y_max]
    points = np.array(points)
    return list(map(int, [
        np.min(points[:, 0]),
        np.min(points[:, 1]),
        np.max(points[:, 2]),
        np.max(points[:, 3]),
    ]))


def A_B(A, B):
    x1, y1 = A[0], A[2]
    x2, y2 = B[0], B[2]
    if x1 > y2 or x2 > y1:
        return False
    else:
        return min(x1, x2), 0, max(y1, y2), 0

def B_A(A, B):
    x1, y1 = A[1], A[3]
    x2, y2 = B[1], B[3]
    if x1 > y2 or x2 > y1:
        return False
    else:
        return 0, min(x1, x2), 0, max(y1, y2)

def polygon(gt: dict, save: bool = False, classes: str = 'row') -> PIL.Image:
    cols_bbox = []
    rows_bbox = []

    if save:
        img_path = os.path.join(data_path, gt['split'], gt['filename'])
        pil_img_row = Image.open(img_path)
        pil_img_col = pil_img_row.copy()
    html = ''.join(gt['html']['structure']['tokens'])

    thead = re.findall(r'<thead>(.*?)</thead>', html)
    thead_nums = len(re.findall(r'<td.*?</td>', thead[0]))

    tr = re.findall(r'<tr>(.*?)</tr>', html)

    td = re.findall(r'<td.*?</td>', html)

    tbody = re.findall(r'<tbody>(.*?)</tbody>', html)
    tbody_nums = len(re.findall(r'<td.*?</td>', tbody[0]))

    cells = gt['html']['cells']

    table_bbox = get_bbox([i['bbox']
                           for i in cells if 'bbox' in i])
    table_W, table_H = [table_bbox[0],
                        table_bbox[2]], [table_bbox[1], table_bbox[3]]
    # tr 投影解决高度的问题。

   
    new_bbox = []
    tmp_bbox = [i['bbox'] for i in cells if 'bbox' in i]
    fix_bbox = []
    error_bbox = []
    # cell的阈值大于0.05会被踢出bboxes
    for bbox in tmp_bbox:
        tmp = []
        for B in tmp_bbox:
            iou_score = IOU(B, bbox)
            if iou_score > 0.05 and iou_score != 1.0:
                tmp.append(iou_score)
        if len(tmp) > 2:
            print("tmp", len(tmp))
        if tmp:
            new_bbox.append(tmp)
            error_bbox.append(bbox)
        else:
            fix_bbox.append(bbox)
    if len(error_bbox) > 3:
        return 
    if save:
        copy_cell = pil_img_row.copy()
        draw = ImageDraw.Draw(copy_cell)         
        for box in [i['bbox'] for i in cells if 'bbox' in i]:
            draw.rectangle(box, outline=(255, 0, 255))
    if save:
        copy_fix = pil_img_row.copy()
        draw = ImageDraw.Draw(copy_fix)
        for box in fix_bbox:
            draw.rectangle(box, outline=(255, 0, 255))

    # return 
        
    

    bboxes = fix_bbox

    # for label, bbox in zip(td, cells):
    #     if 'bbox' in bbox:
    #         bboxes.append(bbox['bbox'])
    # if not bboxes:
    #     return None

    td_nums = 0
    if save:
        copy = pil_img_row.copy()
        draw = ImageDraw.Draw(copy, mode='RGBA')

    res_bbox = []
    for id, tr_line in enumerate(tr):
        num = len(re.findall(r'<td.*?</td>', tr_line))
        bbox = [i['bbox']
                for i in cells[td_nums:td_nums+num] if 'bbox' in i]
        bbox = list(filter(lambda box: box not in error_bbox, bbox))
        if not bbox:
            continue
        bbox = get_bbox(bbox)
        bbox = [table_W[0], bbox[1], table_W[1], bbox[3]]
        res_bbox.append(bbox)
        td_nums += num
    height_lt = []
    for i in res_bbox:
        tmp = []
        for j in res_bbox:
            height_score = height_IOU(i,j)
            tmp.append(height_score)
        height_lt.append(max(tmp))
    # 高度阈值小于0.01,不保存
    if height_lt and max(height_lt) > 0.01:
        return 


    for bbox in res_bbox:
        rows_bbox.append({
            'bbox': bbox,
            'category_id': 0,
        })
        if save:
            draw.rectangle(bbox, outline=(255, 0, 255))
    if save:
        copy.save(f"erro_row_picture/{gt['filename'].strip('.png')}_row.png")
        copy_cell.save(f"erro_row_picture/{gt['filename'].strip('.png')}.png")
        copy_fix.save(f"erro_row_picture/{gt['filename'].strip('.png')}_fix.png")



    return rows_bbox 


def IOU(bbox_A, bbox_B):
    min_H = min(bbox_A[3], bbox_B[3]) - max(bbox_A[1], bbox_B[1])
    min_W = min(bbox_A[2], bbox_B[2]) - max(bbox_A[0], bbox_B[0])
    if min_H <= 0 or min_W <= 0:
        return 0.0
    max_H = max(bbox_A[3], bbox_B[3]) - min(bbox_A[1], bbox_B[1])
    max_W = max(bbox_A[2], bbox_B[2]) - min(bbox_A[0], bbox_B[0])
    return min_H * min_W / (max_H * max_W)

def height_IOU(bbox_A,  bbox_B):
    if bbox_A == bbox_B:
        return 0.0
    min_H = min(bbox_A[3],bbox_B[3]) - max(bbox_A[1], bbox_B[1])
    if min_H < 0:
        return 0.0
    max_H = max(bbox_A[3], bbox_B[3]) - min(bbox_A[1], bbox_B[1])
    return min_H / max_H


# exit(height_IOU([1,1,2,2], [1.5,1.5,5,5]))

if __name__ == "__main__":
    # logging
    for index, line in enumerate(reader()):
        # if line['filename'] not in ['PMC1421383_002_00.png', 'PMC1421415_008_00.png', 'PMC1064888_003_01.png', ]:
        #     continue
        if index > 10000:
            break
        logging.info(f"{index}\t->{line['filename']}")
        polygon(line, save=True)
