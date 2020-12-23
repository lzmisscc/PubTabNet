"""
TODO:得到干净的单元格数据

"""

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
from table import main as col
import tqdm

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
        self.dataset_img_path = "pubtabnet/"
        self.images, self.annotations, self.categories = [], [], []
        self.categories += [
            # dict(id=100, name='col'),
            dict(id=1, name='col'),
        ]
        self.flag = flag
        self.reader = jsonlines.open(
            'pubtabnet/PubTabNet_2.0.0.jsonl', 'r').iter

    def main(self, func=func) -> None:
        P2C = Point2Coco()

        def gen_coco(gt: list) -> list:
            bbox_id = 0
            skip_nums = 0
            gt = tqdm.tqdm(gt)
            for id, img in enumerate(gt):
                if img['split'] != self.flag:
                    continue
                # bboxes = [i['bbox'] for i in func(img) if 'bbox' in i]
                # if not gt_fliter(bboxes):
                #     skip_nums += 1
                #     # print("skip_nums:", skip_nums)
                #     # print("id:", id)
                #     continue
                # gt.set_description(f"total:{id},skip_nums:{skip_nums}")
                logging.info(f'{id}-->\t{img["filename"]}')
                im = Image.open(os.path.join(
                    self.dataset_img_path, self.flag, img['filename']))
                W, H = im.size
                if self.flag == 'train':
                    if len(self.images) >= 10000:
                        break
                elif self.flag == 'val':
                    if len(self.images) >= 10000:
                        break
                else:
                    if len(self.images) >= 1000:
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
                            'category_id': box.get('category_id', 100),
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
        with open(f'table_json_1204/table_col_1216_{self.flag}.json', 'w') as f:
            json.dump(dict(
                images=self.images,
                annotations=self.annotations,
                categories=self.categories,
            ), f)

        logging.info(f"Finish {self.flag}".center(15, '!'))


def IOU(BBOX_A, BBOX_B) -> bool:
    if BBOX_A == BBOX_B:
        return 1.
    min_H = min(BBOX_A[3], BBOX_B[3]) - max(BBOX_B[1], BBOX_A[1])
    min_W = min(BBOX_A[2], BBOX_B[2]) - max(BBOX_B[0], BBOX_A[0])
    if min_H <= 0 or min_W <= 0:
        return 0.
    max_H = max(BBOX_A[3], BBOX_B[3]) - min(BBOX_A[1], BBOX_B[1])
    max_W = max(BBOX_A[2], BBOX_B[2]) - min(BBOX_A[0], BBOX_B[0])
    return min_H*min_W / (max_H*max_W)


def gt_fliter(bboxes):
    for A in bboxes:
        for B in bboxes:
            if A == B:
                continue
            score = IOU(A, B)
            if score > 0.05:
                return False
    return True
            
    
# res = IOU([0, 0, 2, 1], [0.1, 0, 1, 1])
# exit(res)


if __name__ == "__main__":
    # COCO(flag='train').main(col)
    COCO(flag='val').main(col)
