import jsonlines
import os
from PIL import Image
import re
import logging

logging.basicConfig(level=logging.INFO, filename='find.log')


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
                img = Image.open(os.path.join(self.data_path, json['filename']))
                img.save(f'error_image/{json["filename"]}')
                logging.info(os.path.join(self.data_path, json['filename']))

        return

    def get_across_col_row(self, ):
        return


if __name__ == "__main__":
    Get_Error('train').get_none_head()
