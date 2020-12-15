import os
from os.path import join
import json
from PIL import Image
import base64
# import cv2
import numpy as np
# from ocr.model import predict2 as ocr
import jsonlines
import os.path as osp

from tqdm.cli import main
from table2col_row import polygon as col_row
import tqdm
reader = jsonlines.open("pubtabnet/PubTabNet_2.0.0.jsonl").iter()
td = tqdm.tqdm(reader)


def imageToStr(image):
    with open(image, 'rb') as f:
        image_byte = base64.b64encode(f.read())
        # print(type(image_byte))
    image_str = image_byte.decode('ascii')  # byte类型转换为str
    # print(type(image_str))
    return image_str


with open('1.json', 'r') as f:
    L = json.load(f)

templates = {
    "label": "1",
    "line_color": None,
    "fill_color": None,
    "points": [
        [
            158.21739130434776,
            130.6086956521739
        ],
        [
            598.4347826086955,
            177.3478260869565
        ]
    ],
    "shape_type": "rectangle",
    "flags": {}
}


def txt2json(img_path, save_name, lines,):

    points = []
    for line in lines:
        x_min, y_min, x_max, y_max = line
        point = [[x_min, y_min], [x_max, y_max]]
        templates["points"] = point
        templates["label"] = "col"
        points += [templates.copy()]
    L["shapes"] = points
    L["imagePath"] = os.path.basename(img_path)
    L["imageData"] = imageToStr(img_path)
    L["imageWidth"], L["imageHeight"] = Image.open(img_path).size

    with open(save_name, 'w') as f:
        json.dump(L, f)


if __name__ == "__main__":
    import glob
    png = glob.glob("fliter_image_cols_20201208/*/*.png")
    png = [osp.basename(i) for i in png]
    for la in td:
        if la['filename'] not in png:
            continue
        td.set_description(la['filename'])
        bboxes = []
        col = col_row(la)
        if not col:
            continue
        for i in col:
            if i['category_id'] == 1:
                bboxes.append(i['bbox'])
        img_path = osp.join("fliter_image_cols_20201208", la['split'], la['filename'])
        txt2json(img_path, img_path.replace('.png', '.json'), bboxes)

    