from __future__ import barry_as_FLUFL
import jsonlines
import logging
import os
from PIL import Image

logging.basicConfig(
    level=logging.INFO
)

reader = jsonlines.open(
    "/data/ouyangshizhuang/data/table_match/pubtabnet/PubTabNet_2.0.0.jsonl", "r")

reader = iter(reader)


def get_text(img: dict, dataset_path: str = "/data/ouyangshizhuang/data/table_match/pubtabnet/") -> list:
    flag = img['split']
    image_path = os.path.join(dataset_path, flag, img['filename'])
    image = Image.open(image_path)
    text_list = []
    points = []
    corp_imgs = []
    for cell in img['html']['cells']:
        if 'bbox' not in cell:
            continue
        text_list.append(''.join(cell['tokens']))
        points.append(cell['bbox'])
        corp_imgs.append(image.crop(cell['bbox']))

    return {
        'text_list': text_list,
        'corp_imgs': corp_imgs,
        'points': points,
        'filename': img['filename'],
        'flag': img['split']
    }


def main(flag='train'):
    data = next(reader)
    while data['split'] != flag:
        data = next(reader)

    result = get_text(data)
    logging.info(result['filename'])
    index = 0
    for text, crop_img in zip(result['text_list'], result['corp_imgs']):
        index += 1
        pre = str(index).zfill(5)
        crop_save_name = result['filename'].replace('.png', f'_{pre}.png')
        crop_txt_save_name = crop_save_name.replace('.png', '.txt')
        logging.info([result['flag'], crop_txt_save_name])
        with open(os.path.join('dataset', result['flag'], crop_txt_save_name), 'w') as f:
            f.write(text)
        crop_img.save(os.path.join(
            'dataset', result['flag'], crop_save_name))
    return


if __name__ == "__main__":
    for i in range(100):
        main('val')
