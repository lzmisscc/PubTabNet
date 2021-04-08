import os
import json
import jsonlines
import re
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import os.path as osp 
import tqdm

def check_bbox(bboxes):
    bboxes_orgin = bboxes.copy()
    bboxes_after = sorted(bboxes, key=lambda x: (x[0], x[1]))
    if bboxes_orgin == bboxes_after:
        return True
    else:
        return False


def meger_bbox(bboxes):
    bboxes = np.array(bboxes)
    return list((np.min(bboxes[:, 0]), np.min(bboxes[:, 1]), np.max(bboxes[:, 2]), np.max(bboxes[:, 3])))


def draw(j):
    filename = j['filename']
    im = Image.open(osp.join('pubtabnet', j['split'], filename))
    im_draw = ImageDraw.Draw(im)
    cells = j['html']['cells']
    i = 0
    for cell in cells:
        if 'bbox' in cell:
            im_draw.rectangle(cell['bbox'], outline=(255, 0, 255))
            im_draw.text(cell['bbox'][:2], str(i).zfill(5), fill='red')
            i += 1 

    im.save(f'show_0201/show.jpg')
Reader = jsonlines.open("pubtabnet/PubTabNet_2.0.0.jsonl", "r").iter()
for i in tqdm.tqdm(Reader):
    if i['filename'] == 'PMC3599964_003_00.png':
        draw(i)
exit()

count = 0
all_count = 0

Reader = jsonlines.open("pubtabnet/PubTabNet_2.0.0.jsonl", "r").iter()
for j in Reader:
    filename = j['filename']
    cells = j['html']['cells']
    html = ''.join(j['html']['structure']['tokens'])
    trs = re.findall('<tr>.*?</tr>', html)
    for tr in trs:
        tds = re.findall('<td.*?</td>', tr)
        bboxes = []
        for td in tds:
            t = cells.pop(0)
            if 'bbox' in t:
                bboxes.append(t['bbox'])
        # check bboxes
        all_count += 1
        if not check_bbox(bboxes):
            count += 1
            print(filename, count, all_count, count / all_count * 100)
            break
