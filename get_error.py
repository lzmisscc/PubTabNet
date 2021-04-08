from logging import log
from PIL.ImageDraw import ImageDraw
import jsonlines
import os
from PIL import Image
import re
import logging

logging.basicConfig(level=logging.INFO, )


class Get_Error:
    def __init__(self, flag="train") -> None:
        self.read = jsonlines.open(
            "/data/liuzhuang/DataSet/pubtabnet/PubTabNet_2.0.0.jsonl", "r")
        self.data_path = os.path.join(
            "/data/liuzhuang/DataSet/pubtabnet/", flag)
        self.flag = flag

    def get_none_head(self, ):
        for json in self.read.iter():
            if json['split'] != self.flag:
                continue
            html = ''.join(json['html']['structure']['tokens'])
            get_head = re.findall('<thead>(.*?)</thead>', html)
            td_num = len(re.findall('<td', ''.join(get_head)))
            bboxes = []
            for cell in json['html']['cells'][0:td_num]:
                if 'bbox' in cell:
                    bboxes.append(cell['bbox'])
            if bboxes == []:
                img = Image.open(os.path.join(
                    self.data_path, json['filename']))
                img.save(f'error_image/{json["filename"]}')
                logging.info(os.path.join(self.data_path, json['filename']))

        return

    def get_across_col_row(self, ):
        from table2col_row import polygon
        for json in self.read.iter():
            row = [r['bbox']
                   for r in polygon(json) if r['category_id'] == 0]
            cells = json['html']['cells']
            bboxes = [i['bbox']
                      for i in cells if 'bbox' in i]
            for i in bboxes:
                num = 0
                for j in row:
                    if IOU(i, j):
                        num += 1
                    else:
                        continue

                    if num > 3:
                        img = Image.open(os.path.join(
                            self.data_path, json['filename']))
                        draw = ImageDraw(img, mode='RGBA')

                        for x in row:
                            draw.rectangle(x, outline='blue')
                        for y in bboxes:
                            draw.rectangle(y, fill=(0, 255, 0, 150))
                        draw.rectangle(i, fill=(255, 255, 0, 155))

                        img.save(f'error_across_image/{json["filename"]}')
                        logging.info(f"num:\t{num}")
                        logging.info(os.path.join(
                            self.data_path, json['filename']))
                        break
                if num > 3:
                    break
        return


def IOU(A, B) -> bool:
    if min(A[2], B[2]) - max(A[0], B[0]) > 0 and min(A[3], B[3]) - max(A[1], B[1]) > 0:
        if (min(A[2], B[2]) - max(A[0], B[0])) * (min(A[3], B[3]) - max(A[1], B[1])) > 3500:
            return False
        else:
            return True
    else:
        return False


if __name__ == "__main__":
    # Get_Error('train').get_none_head()
    Get_Error('train').get_across_col_row()
    # logging.info(IOU([1, 1, 1.5, 1.5], [1, 1, 1, 1], ))
