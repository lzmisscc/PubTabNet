import glob
import fire
import os
import logging
logging.basicConfig(level=logging.INFO)
args = fire.Fire


def match(path):
    json = glob.glob(os.path.join(path, '*.json'))
    png = glob.glob(os.path.join(path, '*.png'))

    for p in png:
        flag = False
        for j in json:
            if os.path.basename(p).endswith('.png') in j:
                flag = True
                break
        if not flag:
            logging.info(
                "Not match png\t{}".format(p, ))
    return


args(match)
