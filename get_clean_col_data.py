import enum
import jsonlines
import os
import os.path as osp
import glob
import json
import re
import tqdm
from PIL import Image
reader = jsonlines.open("pubtabnet/PubTabNet_2.0.0.jsonl").iter()
tq = tqdm.tqdm(reader)
for json_data in tq:
    # if json_data['split'] != 'val':
    #     continue
    cells = json_data["html"]["cells"]
    htmls = "".join(json_data["html"]["structure"]["tokens"])


    trs = re.findall("<tr>.*?</tr>", htmls)
    hash_id = {}
    id = 0
    td_lens = []
    for tr_id, tr in enumerate(trs):
        tds = re.findall("(<td.*?>.*?</td>)", tr)
        td_lens.append(len(tds))
        hash_id[tr_id] = []

        while tds:
            td = tds.pop(0)
            cell = cells.pop(0)
            hash_id[tr_id].append((id, td, cell))
            id += 1


    max_len_tr = max(td_lens)
    res = filter(lambda x: len(x[1])==max_len_tr, hash_id.items())
    res = list(res)
    

    cols_bbox = [[] for i in range(len(res[0][1]))]
    for tr in res:
        for id, td in enumerate(tr[1]):
            cols_bbox[id].append(td[2]['bbox'] if 'bbox' in td[2] else [])


