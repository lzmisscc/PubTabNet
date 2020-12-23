import glob
import os.path as osp
from PIL import Image
import math
import tqdm
# def get_w_h(name):
#     im = Image.open(name)
#     w,h = im.size
#     f.write(f"{osp.basename(name)},{w},{h}\n")
#     del im

# if __name__ == "__main__":
#     from multiprocessing import Pool
#     import tqdm
#     # data = glob.glob("../table_ocr/data/train/*.png")
#     f = open("table_ocr/abs_train.txt", "r").readlines()
#     data = []
#     for i in f:
#         name, _ = i.strip("\n").split("\t")
#         data.append(name)

#     data  =tqdm.tqdm(data)
#     f = open("statics_w_h_1217.list", "w")
#     with Pool(90) as P:
#         P.map(get_w_h, data)


with open("statics_w_h_1217.list", "r") as f:
    data = f.readlines()
data = tqdm.tqdm(data)
s = {}
total = 0
for line in data:
    name,w,h = line.strip("\n").split(",")
    w = int(w)
    h = int(h)
    ratio = math.ceil(math.ceil(32/h*w) / 32)
    if ratio in  s:
        s[ratio] += 1
    else:
        s[ratio] = 1
    total += 1
s = sorted(s.items(), key=lambda x:x[1])
s = [(i[0], round(i[1]/total*100, 3)) for i in s]
print(*s, total, sep="\n")
    

