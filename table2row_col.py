from __future__ import barry_as_FLUFL
import PIL
from shapely.geometry import polygon
import jsonlines
import logging
from PIL import Image, ImageDraw, ImagePath
import os
import numpy as np

logging.basicConfig(
    level=logging.INFO
)


def read_json_lines():
    with jsonlines.open('examples/PubTabNet_Examples.jsonl', 'r') as reader:
        imgs = list(reader)
    return imgs


def polygon(gt: dict) -> PIL.Image:
    img_path = os.path.join("examples/", gt['filename'])
    pil_img = Image.open(img_path)
    draw = ImageDraw.Draw(pil_img)
    W, H = pil_img.size
    cells = gt['cells']
    bboxs = []
    for cell in cells:
        if not cell['bbox']:
            continue
        bboxs.append(cell['bbox'])


