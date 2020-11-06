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


def polygon(gt: dict) -> PIL.Image:
    img_path = os.path.join("examples/", gt['filename'])
    pil_img = Image.open(img_path)
    draw = ImageDraw.Draw(pil_img, mode='RGBA')
    # 找出表格的宽和高
    cells = gt['html']['cells']
    bboxes = []
    for cell in cells:
        if 'bbox' not in cell:
            continue
        bboxes.append(cell['bbox'])
    bboxes = np.array(bboxes)
    left, top, right, buttom = np.min(bboxes[:, 0]), np.min(
        bboxes[:, 1]), np.max(bboxes[:, 2]), np.max(bboxes[:, 3])

    draw.rectangle([left, top, right, buttom], fill=(255, 0, 0, 150))
    W, H = right-left, buttom-top

    # 找出每行和列的关系
    rows, cell_nums = 0, 0
    table = []
    structure = gt['html']['structure']['tokens']
    structure_len = len(structure)
    for index, cell in enumerate(structure):
        if '<tr' in cell:
            table.append([])
        elif '<td' in cell:
            if index < structure_len and 'colspan' in structure[index+1]:
                # 计算出列的数量
                nums = re.search(r'[0-9]+', structure[index+1])
                if nums:
                    nums = int(nums.group())
                    # table[-1].append([[nums] for _ in range(nums)])
                    table[-1].append([nums])
                continue
            elif index < structure_len and 'rowspan' in structure[index+1]:
                # 计算出列的数量
                nums = re.search(r'[0-9]+', structure[index+1])
                if nums:
                    nums = int(nums.group())
                    table[-1].append([-nums])
                continue
            table[-1].append([1])
    print(*table, sep='\n')

    table = np.array(table, np.int)

#     pil_img.save(f"vis_col_row/{gt['filename']}")

#     return pil_img


for index, line in enumerate(reader()):
    logging.info(f"{index}->{line['filename']}")
    polygon(line)
