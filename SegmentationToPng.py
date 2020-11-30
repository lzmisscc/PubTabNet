import os
import json
import numpy as np
from pycocotools.coco import COCO
from cocostuff.cocostuffhelper import cocoSegmentationToPng


split = 'val'
coco = COCO(annotation_file=f"coco2png/table_json/table_{split}.json")
keys = coco.imgs.keys()

for key in keys:
    img = coco.imgs[key]['file_name']
    print(coco.imgs[key])
    cocoSegmentationToPng(coco, key, pngPath=f"coco2png/{split}/{img}", includeCrowd=False)
# print(coco)