from cnocr import CnOcr
import fire
from PIL import Image
import os
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
ocr = CnOcr().ocr


def main(file_path):
    result = ocr(file_path)
    logging.info(f"Result\t{result}")
    print(result)


fire.Fire(main)
