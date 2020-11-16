import jsonlines
from PIL import Image, ImageDraw
import os
import logging
import numpy as np
from find_text_area import detect

logging.basicConfig(level=logging.INFO, )
data_path = "examples/"
save_data_path = "table_ocr_dataset"
reader = jsonlines.open(
    "examples/PubTabNet_Examples.jsonl").iter()


def multi__lines_image_text(image: Image, filename: str):
    source = image.copy()
    size = source.size
    image = np.array(image, np.uint8)
    bbox = detect(image)
    if len(bbox) == 1:
        return source
    elif len(bbox) > 1:
        logging.info(f"multi__lines_image_text:\t{filename}")
        bbox.sort(key=lambda x:x[1])
        bbox = np.array(bbox)
        w = np.sum(bbox[:, 2])
        max_h = h = np.max(bbox[:, 3])
        canvas = Image.new('RGB', (w, h), (255, 255, 255))
        W = 0
        for box in list(bbox):
            x, y, w, h = box
            canvas.paste(source.crop((x, y, x+w, y+h)), (x+W, 0, x+W+w, h))
            W += w

        # canvas.paste(source, (0, max_h, size[0], max_h+size[1]))
        return canvas
    else:
        return source


logging.info('START'.center(50, '-'))

train_txt, val_txt = [], []
for id, json in enumerate(reader):
    text = json['html']['cells']
    try:
        img = Image.open(os.path.join(
            data_path, json['split'], json['filename']))
    except:
        img = Image.open(os.path.join(data_path,  json['filename']))
    finally:
        logging.info(f"{id}\t{json['filename']}")

        for index, _ in enumerate(text):
            if 'bbox' not in _:
                continue
            image, text = img.crop(_['bbox']), ''.join(_['tokens'])

            save_name = json['filename'][:-4] + \
                '_' + str(index).zfill(5) + '.png'
            image = multi__lines_image_text(image, save_name+'\t'+text)
            image.save(os.path.join(save_data_path, json['split'], save_name))
            if json['split'] == 'train':
                train_txt.append(f'{save_name}\t{text}\n')
            elif json['split'] == 'val':
                val_txt.append(f'{save_name}\t{text}\n')
            else:
                logging.info(f"ERROR SPLIT {json['split']}")

logging.info("Save_Train".center(50, '-'))
with open(os.path.join(save_data_path, 'train.txt'), 'w') as f:
    f.writelines(train_txt)

logging.info("Save_Test".center(50, '-'))
with open(os.path.join(save_data_path, 'val.txt'), 'w') as f:
    f.writelines(val_txt)

logging.info("END".center(50, '-'))
