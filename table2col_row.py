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


def get_bbox(points: list) -> list:
    # [[], ...,[]] -> [x_min,y_min,x_max,y_max]
    points = np.array(points)
    return list(map(int, [
        np.min(points[:, 0]),
        np.min(points[:, 1]),
        np.max(points[:, 2]),
        np.max(points[:, 3]),
    ]))


def polygon(gt: dict) -> PIL.Image:
    img_path = os.path.join("examples/", gt['filename'])
    pil_img = Image.open(img_path)
    draw = ImageDraw.Draw(pil_img, mode='RGBA')

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
    # tr 统计td的数量进行bbox的对齐。
    td_nums = 0
    for tr_line in tr:
        num = len(re.findall(r'<td.*?</td>', tr_line))
        bbox = get_bbox([i['bbox']
                         for i in cells[td_nums:td_nums+num] if 'bbox' in i])
        bbox = [table_W[0], bbox[1], table_W[1], bbox[3]]
        # draw.rectangle(bbox, fill=(0, 255, 0, 150), outline=(255, 0, 255))
        td_nums += num

    # td 进行投影决定列的宽度。
    td_nums = 0
    cols = {}

    for tr_line in tr:
        num = len(re.findall(r'<td.*?</td>', tr_line))
        index = 0
        for label, bbox in zip(re.findall(r'<td.*?</td>', tr_line), [i for i in cells[td_nums:td_nums+num]]):
            if index not in cols:
                cols[index] = []
            x = re.findall(r'colspan="([0-9]+?)"', label)
            y = re.findall(r'rowspan="([0-9]+?)"', label)
            if y:
                logging.info(f'y:{y}')
            else:
                y = 0 
            if not x:
                if 'bbox' in bbox:
                    cols[index].append(bbox['bbox'])
                index += 1
            else:
                index += sum(map(int, x))

        td_nums += num
    for key,values in cols.items():
        values = np.array(values)
        draw.rectangle([int(np.min(values[:, 0])), table_H[0], int(np.max(values[:, 2])), table_H[1]], fill=(255, 255, 0, 200), outline=(0, 0, 255))
    pil_img.save(f"vis_col_row/{gt['filename']}")
    return pil_img


for index, line in enumerate(reader()):
    logging.info(f"{index}\t->{line['filename']}")
    polygon(line)
