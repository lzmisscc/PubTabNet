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
from table2thead_tbody import polygon as thead_tbody
from coco import main
from multiprocessing import pool
import json
import jsonlines
import logging
from PIL import Image
import os
import tqdm
from multiprocessing import Process
from table2col_row import polygon as col_row

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


def func(args):
    return args['html']['cells']


class COCO:
    def __init__(self, flag='train') -> None:
        self.dataset_img_path = "/data/liuzhuang/DataSet/pubtabnet/"
        self.images, self.annotations, self.categories = [], [], []
        self.categories += [
            dict(id=0, name='thead'),
            dict(id=1, name='tbody'),
        ]
        self.flag = flag
        self.reader = jsonlines.open(
            '/data/liuzhuang/DataSet/pubtabnet/PubTabNet_2.0.0.jsonl', 'r').iter

    def main(self, func=func) -> None:
        P2C = Point2Coco()

        def gen_coco(gt: list) -> list:
            bbox_id = 0

            for id, img in enumerate(gt):
                if img['split'] != self.flag:
                    continue
                logging.info(f'{id}-->\t{img["filename"]}')
                im = Image.open(os.path.join(
                    self.dataset_img_path, self.flag, img['filename']))
                W, H = im.size
                if self.flag == 'train':
                    if len(self.images) > 10000:
                        break
                elif self.flag == 'val':
                    if len(self.images) > 5000:
                        break
                else:
                    if len(self.images) > 10000:
                        break
                tmp = func(img)
                if not tmp:
                    continue
                for box in func(img):
                    if 'bbox' not in box:
                        continue
                    bbox_id += 1
                    point = box['bbox']
                    point_xywh = [
                        min(point[0], point[2]), min(point[1], point[3]),
                        max(point[0], point[2])-min(point[0], point[2]),
                        max(point[1], point[3])-min(point[1], point[3]),
                    ]
                    self.annotations.append(
                        {
                            # if you have mask labels
                            'segmentation': P2C._get_seg(point),
                            'area': P2C._get_area(point),
                            'iscrowd': 0,
                            'image_id': id,
                            'bbox': point_xywh,
                            'category_id': box.get('category_id', 0),
                            'id': bbox_id
                        },
                    )
                self.images.append(
                        {
                            'file_name': img['filename'],
                            'height': H,
                            'width': W,
                            'id': id,
                        },
                    )
        logging.info("Start".center(15, "!"))
        all_ = self.reader()
        gen_coco(all_)
        self.json_save()

    def json_save(self, ):
        # return annotations, images
        with open(f'table_json/thead_tbody_coco_{self.flag}.json', 'w') as f:
            json.dump(dict(
                images=self.images,
                annotations=self.annotations,
                categories=self.categories,
            ), f)

        logging.info(f"Finish {self.flag}".center(15, '!'))


if __name__ == "__main__":
    COCO(flag='val').main(col_row)
