from __future__ import barry_as_FLUFL
import PIL
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
    if not points:
        return None
    points = np.array(points)
    return list(map(int, [
        np.min(points[:, 0]),
        np.min(points[:, 1]),
        np.max(points[:, 2]),
        np.max(points[:, 3]),
    ]))


def polygon(gt: dict, save=False) -> dict:
    if save:
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
    # thead
    thead_bbox = get_bbox([i['bbox']
                           for i in cells[0:thead_nums] if 'bbox' in i])
    if thead_bbox == None:
        return None
    thead_bbox = [table_W[0], thead_bbox[1], table_W[1], thead_bbox[3]]
    if save:
        draw.rectangle(thead_bbox, fill=(255, 0, 0, 150))
    tbody_bbox = get_bbox(
        [i['bbox'] for i in cells[thead_nums:len(td)] if 'bbox' in i])
    if tbody_bbox == None:
        return None
    tbody_bbox = [table_W[0], tbody_bbox[1], table_W[1], tbody_bbox[3]]
    if save:
        draw.rectangle(tbody_bbox, fill=(
            0, 0, 255, 150), outline='blue')
        pil_img.save(f"vis_thead_tbody/{gt['filename']}")
    return [
        {
            'bbox': thead_bbox,
            'category_id': 0,
        },
        {
            'bbox': tbody_bbox,
            'category_id': 1,

        }
    ]


if __name__ == "__main__":
    for index, line in enumerate(reader()):
        logging.info(f"{index}\t->{line['filename']}")
        polygon(line, save=True)
