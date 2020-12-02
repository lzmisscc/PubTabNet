import os
import json
import numpy as np
from pycocotools.coco import COCO
from cocostuff.cocostuffhelper import cocoSegmentationToPng
from multiprocessing import Process
import tqdm

split = 'train'
coco = COCO(annotation_file=f"coco2png/table_json/table_{split}.json")
keys = coco.imgs.keys()
keys = list(keys)

def cell(keys):
    for key in tqdm.tqdm(keys):
        img = coco.imgs[key]['file_name']
        # print(coco.imgs[key])
        cocoSegmentationToPng(coco, key, pngPath=f"coco2png/{split}/{img}", includeCrowd=False)

n = 50
partition = [keys[i:i+n] for i in range(0,len(keys), n)]
poolings = []
for l in partition:
    s = Process(target=cell, args=(l, ))
    s.start()
    poolings.append(s)
for pool in poolings:
    pool.join()
print("End!")