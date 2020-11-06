# 'images': [
#     {
#         'file_name': 'COCO_val2014_000000001268.jpg',
#         'height': 427,
#         'width': 640,
#         'id': 1268
#     },
#     ...
# ],

# 'annotations': [
#     {
#         'segmentation': [[192.81,
#                           247.09,
#                           ...
#                           219.03,
#                           249.06]],  # if you have mask labels
#         'area': 1035.749,
#         'iscrowd': 0,
#         'image_id': 1268,
#         'bbox': [192.81, 224.8, 74.73, 33.43],
#         'category_id': 16,
#         'id': 42986
#     },
#     ...
# ],

# 'categories': [
#     {'id': 0, 'name': 'car'},
# ]
from multiprocessing import pool
import json
from os import read
import PIL
import jsonlines
import logging
from PIL import Image, ImageDraw, ImagePath
import os
import tqdm
from multiprocessing import Process


logging.basicConfig(
    level=logging.INFO
)


class Point2Coco:

   # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return [min_x, min_y, max_x - min_x, max_y - min_y]
    # 计算面积

    def _get_area(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return (max_x - min_x+1) * (max_y - min_y+1)
    # segmentation

    def _get_seg(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        h = max_y - min_y
        w = max_x - min_x
        a = []
        a.append([min_x, min_y, min_x, min_y+0.5*h, min_x, max_y, min_x+0.5*w, max_y,
                  max_x, max_y, max_x, max_y-0.5*h, max_x, min_y, max_x-0.5*w, min_y])
        return a


def main():
    P2C = Point2Coco()
    images, annotations, categories = [], [], []
    categories += [
        dict(id=0, name='cell'),
    ]
    dataset_img_path = "/home/work/DataSet/pubtabnet/"

    logging.info("Start".center(15, "!"))

    def gen_coco(flag: str):
        annotations, images = [], []
        bbox_id = 0
        reader = jsonlines.open(
            '/home/work/DataSet/pubtabnet/PubTabNet_2.0.0.jsonl', 'r')

        for id, img in tqdm.tqdm(enumerate(reader.iter())):
            if flag != img['split']:
                continue
            im = Image.open(os.path.join(
                dataset_img_path, flag, img['filename']))
            W, H = im.size
            # logging.info(f"{id}->{img['filename']}")
            images.append(
                {
                    'file_name': img['filename'],
                    'height': H,
                    'width': W,
                    'id': id,
                },
            )
            for box in img['html']['cells']:
                if 'bbox' not in box:
                    continue
                bbox_id += 1
                point = box['bbox']
                point_xywh = [
                    min(point[0], point[2]), min(point[1], point[3]),
                    max(point[0], point[2])-min(point[0], point[2]),
                    max(point[1], point[3])-min(point[1], point[3]),
                ]
                annotations.append(
                    {
                        # if you have mask labels
                        'segmentation': P2C._get_seg(point),
                        'area': P2C._get_area(point),
                        'iscrowd': 0,
                        'image_id': id,
                        'bbox': point_xywh,
                        'category_id': 0,
                        'id': bbox_id
                    },
                )
        return annotations, images

    def get_flag(flag):
        annotations, images = gen_coco(flag)
        with open(f'table_json/table_coco_{flag}.json', 'w') as f:
            json.dump(dict(
                images=images,
                annotations=annotations,
                categories=categories,
            ), f)

        logging.info(f"Finish {flag}".center(15, '!'))
    pools = []
    for flag in ['train', ]:
        pool = Process(target=get_flag, args=(flag, ))
        pools.append(pool)
        pool.start()
    for pool in pools:
        pool.join()

if __name__ == "__main__":
    main()
