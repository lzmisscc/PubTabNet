import json
import numpy as np
import skimage.io
import cv2
import matplotlib.pyplot as plt
import PIL.Image
import PIL.ImageDraw
import numpy

data = json.load(open('table_coco.json', 'r'))


def polygons_to_mask(img_shape, polygons):
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask
