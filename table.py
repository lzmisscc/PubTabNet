import numpy as np
import jsonlines
import re
from bs4 import BeautifulSoup as bs
from html import escape
from PIL import Image, ImageDraw
import os.path as osp
import random


class MyTable:
    def __init__(self, *args, gt) -> None:
        self.data_path = "pubtabnet/"
        self.gt = gt
        self.html = gt['html']['structure']['tokens']
        self.html = ''.join(self.html)

        self.cells = gt['html']['cells']
        self.bbox = [box['bbox'] for box in self.cells if 'bbox' in box]
        self.text = [t['tokens'] for t in self.cells if t['tokens']]

        self.filename = gt['filename']
        self.split = gt['split']
        self.image = Image.open(
            osp.join(self.data_path, self.split, self.filename)).convert('RGBA')
        self.tds = re.findall(r"<td.*?/td>", self.html)
        self.trs = re.findall(r"<tr.*?/tr>", self.html)
        self.ids = len(self.tds)
        self.row_length, self.col_length = self.get_length()
        self.table_mat = [
            ['*' for j in range(self.col_length)] for i in range(self.row_length)]
        self.get_table()  # 得到table矩阵

        # print(*self.table_mat, sep='\n')
        # print("len:", len(self.bbox))
        # print(self.filename)

    def get_table(self,):
        table_mat = self.table_mat.copy()
        col, row = 0, 0
        cells = self.cells
        count_None = 0
        for id, td in enumerate(self.tds):
            find_rows_num = re.findall('rowspan="([0-9]+?)"', td)
            find_cols_num = re.findall('colspan="([0-9]+?)"', td)
            find_rows_num = int(find_rows_num[0]) if find_rows_num else 1
            find_cols_num = int(find_cols_num[0]) if find_cols_num else 1
            for i in range(find_rows_num):
                for j in range(find_cols_num):
                    if 'bbox' in cells[id]:
                        table_mat[row+i][col+j] = id - count_None
                    else:
                        table_mat[row+i][col+j] = None
            if 'bbox' not in cells[id]:
                count_None += 1
            flag = 0
            for i_ in range(row, self.row_length):
                for j_ in range(0, self.col_length):
                    if table_mat[i_][j_] == '*':
                        row = i_
                        col = j_
                        flag = 1
                        break
                if flag:
                    flag = 0
                    break
        self.table_mat = table_mat

    def get_length(self):
        # first_td = self.trs[0]
        col_length_ld = []
        for first_td in self.trs:
            first_td = re.findall("<td.*?></td>", first_td)
            col_length = 0
            for cell in first_td:
                find = re.findall('colspan="([0-9]+?)"', cell)
                find_num = int(find[0]) if find else 1
                col_length += find_num
            col_length_ld.append(col_length)
        # row_length = 0
        # find_num = 1
        # for tr in self.trs:
        #     if find_num > 1:
        #         find_num -= 1
        #         continue
        #     find = re.findall('rowspan="([0-9]+?)"', tr)
        #     find_num = int(find[0]) if find else 1
        #     row_length += find_num
        row_length = len(self.trs)
        col_length = max(col_length_ld)
        return row_length, col_length

    def create_same_col_matrix(self):
        #  创建列矩阵，将列对齐
        all_col = []
        tmp = list(zip(*self.table_mat))
        for i in tmp:
            all_col.append([j for j in i if j != None])
        self.all_col = all_col
        return self.create_same_matrix(all_col, ids=len(self.bbox))

    def create_same_row_matrix(self):
        # 创建行矩阵，将行对齐
        # all_row = self.table_mat
        all_row = []
        for i in self.table_mat:
            all_row.append([j for j in i if j != None])
        return self.create_same_matrix(all_row, ids=len(self.bbox))

    def create_same_cell_matrix(self):
        # 创建单元格矩阵
        all_cell = []
        # for id, td in enumerate(self.tds):
        #     all_cell.append([id])
        for i in self.table_mat:
            for j in i:
                if j != None:
                    all_cell.append([j])

        return self.create_same_matrix(all_cell, ids=len(self.bbox))

    def create(self):
        same_col_mat, same_row_mat, same_cell_mat = \
            self.create_same_col_matrix(),\
            self.create_same_row_matrix(), \
            self.create_same_cell_matrix()
        html = self.create_html()
        tablecategory = 4
        im = self.image
        bbox = []
        for id, (box, text) in enumerate(zip(self.bbox, self.text)):
            bbox.append([id, ''.join(text)] + box)
        return same_cell_mat, same_col_mat, same_row_mat, len(self.bbox), html, tablecategory, im, bbox

    def create_same_matrix(self, arr, ids):
        '''Given a list of lists with each list consisting of all ids considered same, this function
         generates a matrix '''
        matrix = np.zeros(shape=(ids, ids))
        for subarr in arr:
            for element in subarr:
                matrix[element, subarr] = 1
        return matrix

    def create_html(self):
        # 创建完整的HTLM标签
        # Helper function to read in tables from the annotations

        def format_html(img):
            ''' Formats HTML code from tokenized annotation of img
            '''
            html_code = img['html']['structure']['tokens'].copy()
            to_insert = [i for i, tag in enumerate(
                html_code) if tag in ('<td>', '>')]
            for i, cell in zip(to_insert[::-1], img['html']['cells'][::-1]):
                if cell['tokens']:
                    cell = [escape(token) if len(token) ==
                            1 else token for token in cell['tokens']]
                    cell = ''.join(cell)
                    html_code.insert(i + 1, cell)
            html_code = ''.join(html_code)
            html_code = '''<html>
                        <head>
                        <meta charset="UTF-8">
                        <style>
                        table, th, td {
                            border: 1px solid black;
                            font-size: 10px;
                        }
                        </style>
                        </head>
                        <body>
                        <table frame="hsides" rules="groups" width="100%%">
                            %s
                        </table>
                        </body>
                        </html>''' % html_code

            # prettify the html
            soup = bs(html_code)
            html_code = soup.prettify()
            return html_code

        return format_html(self.gt)
# class fliter_head:
#     def __init__(self, gt):
#         self.html =
#         self.thead = re.sub(r"(.*?)<thead>(.*?)</thead>(.*)", r"\2",self.html)
#         self.thead_tds = re.findall("<td.*?</td>", self.thead)
#         self.thead_length = len(self.thead_tds)
#         self.html = re.sub(r"(.*?)<thead>(.*?)</thead>(.*)", r"\3",self.html)
#         self.cells = self.cells[:self.thead_length]
#         self.bbox = self.bbox[:self.thead_length]
#         self.text = self.text[:self.thead_length]


def get_bbox(points: list) -> list:
    # [[], ...,[]] -> [x_min,y_min,x_max,y_max]
    points = np.array(points)
    return list(map(int, [
        np.min(points[:, 0]),
        np.min(points[:, 1]),
        np.max(points[:, 2]),
        np.max(points[:, 3]),
    ]))

samples = list(range(256))

def main(gt, save_flag=True):
    # for gt in Reader:
        # if gt['filename'] not in ['PMC2575590_004_00.png', 'PMC6008895_006_01.png', 'PMC4683521_005_00.png']:
        #     continue

        # 引入去除表头
    try:
        t = MyTable(gt=gt)
        col_same = t.create_same_col_matrix()
        all_col = t.all_col
        bbox = t.bbox
        table_mat = t.table_mat
        fliter_bbox = []
        for i in table_mat:
            for j in i:
                if j != None:
                    if len(i)-i[::-1].index(j)-1 != i.index(j):
                        fliter_bbox.append(j)
        filename = gt['filename']
        im = Image.open(f"pubtabnet/{gt['split']}/{filename}")
        draw = ImageDraw.Draw(im)
        cols_bbox = []
        l_, t_, r_, b_ = get_bbox(bbox)
        for id, b in enumerate(all_col):
            box = [bbox[i] for i in b]
            # l_, t_, r_, b_ = get_bbox(box)
            # _, H = r_-l_, b_-t_

            box = [bbox[i] for i in b if i not in fliter_bbox]
            l_, _, r_, _ = get_bbox(box)
            # W, _ = r_-l_, b_-t_
            # b = bbox[b[-1]]
            draw.rectangle([l_, t_, r_, b_], outline=tuple(random.sample(samples, 3)))
            cols_bbox.append({
                    'bbox': [l_, t_, r_, b_],
                    'category_id': 1,
                })
        if save_flag:
            im.save(f"get_new_data/{filename}")
        return cols_bbox
    except Exception:
        return None

if __name__ == "__main__":
    import tqdm
    from multiprocessing import Pool
    
    Reader = jsonlines.open("pubtabnet/PubTabNet_2.0.0.jsonl", "r").iter()
    Reader = [Reader.__next__() for i in range(100000)]
    Reader = tqdm.tqdm(list(Reader))

    with Pool(40) as P:
        P.map(main, Reader)
