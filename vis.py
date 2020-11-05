from __future__ import barry_as_FLUFL
import PIL
from shapely.geometry import polygon
import jsonlines
import logging
from PIL import Image, ImageDraw, ImagePath
import os
import numpy as np

logging.basicConfig(
    level=logging.INFO
)


def read_json_lines():
    with jsonlines.open('examples/PubTabNet_Examples.jsonl', 'r') as reader:
        imgs = list(reader)
    return imgs


def polygon(img: dict) -> PIL.Image:
    img_path = os.path.join("examples/", img['filename'])
    pil_img = Image.open(img_path)
    draw = ImageDraw.Draw(pil_img)
    points = []
    for point in img['html']['cells']:
        if 'bbox' not in point:
            continue
        points.append(point['bbox'])
        xxyy = ImagePath.Path(point['bbox']).getbbox()
        draw.rectangle(xxyy, fill=None, outline='blue', )
    pil_img.show()
    return pil_img


def table(img: dict) -> PIL.Image:
    img_path = os.path.join("examples/", img['filename'])
    pil_img = Image.open(img_path)
    draw = ImageDraw.Draw(pil_img)

    lt = []
    points = img['html']['cells']
    tmp = []
    for html_label in img['html']['structure']['tokens']:
        if html_label == '</td>':
            # 收集矩形框
            point = points.pop(0)
            if 'bbox' in point:
                tmp.append(point['bbox'])
                # 绘制蓝色，<td>
                xxyy = ImagePath.Path(point['bbox']).getbbox()
                draw.rectangle(xxyy, fill='blue', outline=None, )

            while lt[-1] != '<td>':
                if lt[-1] == '<td':
                    lt.pop()
                    break
                lt.pop()
        elif html_label == '</tr>':
            # 解决左上、右下
            if tmp:
                tmp = np.array(tmp).reshape(-1, 2)
                point = [
                    np.min(tmp[:, 0]), np.min(tmp[:, 1]),
                    np.max(tmp[:, 0]), np.max(tmp[:, 1]),
                ]
            else:
                point = None

            # 绘制红色矩形框
            xxyy = ImagePath.Path(point).getbbox()
            draw.rectangle(xxyy, fill=None, outline='red', )
            tmp = []

            while lt[-1] != '<tr>':
                lt.pop()
        elif html_label == '</head>':
            while lt[-1] != '<head>':
                lt.pop()
        else:
            lt.append(html_label)

    pil_img.save(f"vis/{img['filename']}")
    return


if __name__ == "__main__":
    img = read_json_lines()
    for index, i in enumerate(img):
        logging.info(f"\t{index}->{i['filename']}")
        table(i)
