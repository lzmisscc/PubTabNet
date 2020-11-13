import jsonlines
from PIL import Image, ImageDraw
import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, )
data_path = "/data/liuzhuang/DataSet/pubtabnet/"
save_data_path = "table_ocr_dataset"
reader = jsonlines.open(
    "/data/liuzhuang/DataSet/pubtabnet/PubTabNet_2.0.0.jsonl").iter()


def multi__lines_image_text(image: Image, ):
    return image


logging.info('START'.center(50, '-'))

train_txt, val_txt = [], []
for id, json in enumerate(reader):
    text = json['html']['cells']
    try:
        img = Image.open(os.path.join(
            data_path, json['split'], json['filename']))
    except:
        img = Image.open(os.path.join(data_path,  json['filename']))
        logging.info("Not ERROR")
    finally:
        logging.info(f"{id}\t{json['filename']}")

        for index, _ in enumerate(text):
            if 'bbox' not in _:
                continue
            image, text = img.crop(_['bbox']), ''.join(_['tokens'])
            image = multi__lines_image_text(image)

            save_name = json['filename'][:-4] + '_' + str(index).zfill(5) + '.png'
            image.save(os.path.join(save_data_path, json['split'], save_name))
            if json['split'] == 'train':
                train_txt.append(f'{save_name}\t{text}\n')
            elif json['split'] == 'val':
                val_txt.append(f'{save_name}\t{text}\n')
            else:
                logging.info(f"ERROR SPLIT {json['split']}")

logging.info("Save".center(50, '-'))

with open(os.path.join(save_data_path, 'train.txt'), 'w') as f:
    f.writelines(train_txt)

with open(os.path.join(save_data_path, 'val.txt'), 'w') as f:
    f.writelines(val_txt)

logging.info("END".center(50, '-'))
