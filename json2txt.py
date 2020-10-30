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

reader = jsonlines.open("", "r")


def get_text(img: dict) -> list:
    text_list = []
    for cell in img['html']['cells']:
        if 'tokens' not in cell:
            continue
        text_list.append(''.join(cell['tokens']+['\n']))
        
    if img['split'] == 'train':
        with open(os.path.join("train", img['filename']), 'w') as f:
            f.writelines(text_list)
    elif img['split'] == 'val':
        with open(os.path.join("val", img['filename']), 'w') as f:
            f.writelines(text_list)
    elif img['split'] == 'test':
        with open(os.path.join("test", img['filename']), 'w') as f:
            f.writelines(text_list)
