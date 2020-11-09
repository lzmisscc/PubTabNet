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


reader = jsonlines.open(
    '/data/liuzhuang/DataSet/pubtabnet/PubTabNet_2.0.0.jsonl', 'r')
reader = reader.iter


def polygon(img: dict, flag: str = 'train') -> PIL.Image:

    img_path = os.path.join(
        f"/data/liuzhuang/DataSet/pubtabnet/{flag}", img['filename'])
    pil_img = Image.open(img_path)
    draw = ImageDraw.Draw(pil_img)
    points = []
    for point in img['html']['cells']:
        if 'bbox' not in point:
            continue
        points.append(point['bbox'])
        xxyy = ImagePath.Path(point['bbox']).getbbox()
        draw.rectangle(xxyy, fill=None, outline='blue', )
    pil_img.save(f"vis_rectangle/{img['filename']}")


if __name__ == "__main__":
    img = reader()
    for index, i in enumerate(img):
        if i['split'] != 'train':
            continue
        if index > 10000:
            break
        logging.info(f"\t{index}->{i['filename']}")
        polygon(i)
