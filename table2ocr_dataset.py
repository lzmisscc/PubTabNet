import jsonlines
from PIL import Image, ImageDraw
import os
import logging
import numpy as np
from find_text_area import detect

logging.basicConfig(level=logging.INFO, )
data_path = "/data/liuzhuang/DataSet/pubtabnet/" # 图片路径
save_data_path = "table_ocr_dataset"             # 保存路径 //train //val
reader = jsonlines.open(
    "/data/liuzhuang/DataSet/pubtabnet/PubTabNet_2.0.0.jsonl").iter()
# jsonl 路径

def multi__lines_image_text(image: Image, filename: str):
    source = image.copy()
    size = source.size
    image = np.array(image, np.uint8)
    bbox = detect(image)
    if len(bbox) == 1:
        return source
    elif len(bbox) > 1:
        # logging.info(f"multi__lines_image_text:\t{filename}")
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
class Gen_Txt:
    def __init__(self) -> None:
        self.train_txt, self.val_txt = [], []


    def func(self, reader):
        for id, json in enumerate(reader):
            text = json['html']['cells']
            try:
                img = Image.open(os.path.join(
                    data_path, json['split'], json['filename']))
            except:
                img = Image.open(os.path.join(data_path,  json['filename']))
            finally:
                # logging.info(f"{id}\t{json['filename']}")

                for index, _ in enumerate(text):
                    if 'bbox' not in _:
                        continue
                    image, text = img.crop(_['bbox']), ''.join(_['tokens'])

                    save_name = json['filename'][:-4] + \
                        '_' + str(index).zfill(5) + '.png'
                    image = multi__lines_image_text(image, save_name+'\t'+text)
                    image.save(os.path.join(save_data_path, json['split'], save_name))
                    if json['split'] == 'train':
                        self.train_txt.append(f'{save_name}\t{text}\n')
                    elif json['split'] == 'val':
                        self.val_txt.append(f'{save_name}\t{text}\n')
                    else:
                        logging.info(f"ERROR SPLIT {json['split']}")
    def save(self, ):
        from multiprocessing import Process
        read = list(reader)
        logging.info(f"Reading, {len(read)}条")
        poolings = []
        i = 0
        for i in range(0, len(read)//100000):
            p = Process(target=self.func, args=(read[i*100000:(i+1)*100000], ))
            p.start()
            poolings.append(p)
        p = Process(target=self.func, args=(read[(i+1)*100000:], ))
        p.start()
        poolings.append(p)
        for i in poolings:
            i.join()


        logging.info("Save_Train".center(50, '-'))
        with open(os.path.join(save_data_path, 'train.txt'), 'w') as f:
            f.writelines(self.train_txt)

        logging.info("Save_Test".center(50, '-'))
        with open(os.path.join(save_data_path, 'val.txt'), 'w') as f:
            f.writelines(self.val_txt)

        logging.info("END".center(50, '-'))

if __name__ == "__main__":
    Gen_Txt().save()