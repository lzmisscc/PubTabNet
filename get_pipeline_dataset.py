import jsonlines
import os

# Helper function to read in tables from the annotations
from bs4 import BeautifulSoup as bs
from html import escape
import json
import tqdm


def format_html(img):
    ''' Formats HTML code from tokenized annotation of img
    '''
    html_code = img['html']['structure']['tokens'].copy()
    to_insert = [i for i, tag in enumerate(html_code) if tag in ('<td>', '>')]
    for i, cell in zip(to_insert[::-1], img['html']['cells'][::-1]):
        if cell['tokens']:
            cell = [escape(token) if len(token) ==
                    1 else token for token in cell['tokens']]
            cell = ''.join(cell)
            html_code.insert(i + 1, cell)

    html_code = ''.join(html_code)
    html_code = '''<html><head><meta charset="UTF-8"><style>table, th, td {  border: 1px solid black;  font-size: 10px;}</style></head><body><table frame="hsides" rules="groups" width="100%%">  %s</table></body></html>''' % html_code

    # prettify the html
    # soup = bs(html_code)
    # html_code = soup.prettify()
    return html_code


def main(split="val"):
    res = {}
    data = jsonlines.open("pubtabnet/PubTabNet_2.0.0.jsonl").iter()
    data = tqdm.tqdm(data)
    for img in data:
        filename = img['filename']
        data.set_description(filename)
        if split != img['split']:
            continue

        file = dict(
            html=format_html(img),
            type=split,
        )
        res[filename] = file
    with open(f"src/{split}.json", "w") as f:
        json.dump(res, f)
    print(f"{split} END!")
    return


if __name__ == "__main__":
    main()
    # main("train")
    # data = jsonlines.open("pubtabnet/PubTabNet_2.0.0.jsonl").iter()
    # for i in data:
    #     if i['split'] == 'test':
    #         print(i)