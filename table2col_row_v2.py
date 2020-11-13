from __future__ import barry_as_FLUFL
from logging import log
from os import close
import PIL
from numpy.core.numeric import outer
from shapely.geometry import polygon
import jsonlines
import logging
from PIL import Image, ImageDraw, ImagePath
import os
import numpy as np
import re

logging.basicConfig(
    level=logging.INFO
)
reader = jsonlines.open('examples/PubTabNet_Examples.jsonl', 'r').iter
data_path = "examples/"


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
        img_path = os.path.join(data_path, gt['filename'])
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
    if save:
        draw = ImageDraw.Draw(pil_img_row, mode='RGBA')

    bboxes = []

    for label, bbox in zip(td, cells):
        if 'bbox' in bbox:
            bboxes.append(bbox['bbox'])
    if not bboxes:
        return None

    rows = [bboxes.pop()]
    while bboxes:
        bbox = bboxes.pop()
        flag = False
        for index, val in enumerate(rows):
            res = B_A(bbox, val)
            if res:
                flag = True
                rows[index] = res
        if not flag:
            rows.append(bbox)
    for bbox in rows:
        bbox = [table_W[0], bbox[1], table_W[1], bbox[3]]
        rows_bbox.append({
            'bbox': bbox,
            'category_id': 0,
        })
        if save:
            draw.rectangle(bbox, outline=(255, 0, 255))
    if save:
        pil_img_row.save(f"vis_row/{gt['filename']}")

    # td 进行投影决定列的宽度。
    if save:
        draw = ImageDraw.Draw(pil_img_col, mode='RGBA')

    td_nums = 0
    bboxes = []

    for label, bbox in zip(td, cells):
        if 'bbox' in bbox and not re.findall(r'colspan="([0-9]+?)"', label):
            bboxes.append(bbox['bbox'])
    if not bboxes:
        return None
    cols = [bboxes.pop()]
    while bboxes:
        bbox = bboxes.pop()
        flag = False
        for index, val in enumerate(cols):
            res = A_B(bbox, val)
            if res:
                flag = True
                cols[index] = res
        if not flag:
            cols.append(bbox)
    for bbox in cols:
        cols_bbox.append({
            'bbox': [bbox[0], table_H[0], bbox[2], table_H[1]],
            'category_id': 1,
        })
        if save:
            draw.rectangle([bbox[0], table_H[0], bbox[2], table_H[1]],
                           fill=(0, 255, 255, 150), outline=(0, 0, 255))
    if save:
        pil_img_col.save(f"vis_col/{gt['filename']}")

    return rows_bbox + cols_bbox


if __name__ == "__main__":
    # logging
    for index, line in enumerate(reader()):
        logging.info(f"{index}\t->{line['filename']}")
        polygon(line, save=True)
