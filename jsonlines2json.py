import jsonlines
import json
import os
import os.path as osp

save_path = "pubtabnet/json_train"
split = 'train'


def trans(gt):
    im_name = gt['filename']
    # print(im_name)

    if gt['split'] != split:
        return 

    with open(f"{save_path}/{im_name.replace('.png', '.json')}", 'w') as f:
        json.dump(gt, f,)


def main(reader):
    from multiprocessing import Pool
    with Pool(5) as P:
        P.map_async(trans, reader)


def f(x):
    return x**2

async def run(reader):
    await asyncio.gather(*(list(map(trans, reader))))


def json2md():
    with open("src/train.json", "r") as f:
        j = json.load(f)
    for key,values in j.items():
        with open("src/md/" + key.replace(".png", ".md"), "w") as f:
            f.write(values['html'])

json2md()
exit()
    
if __name__ == "__main__":
    from multiprocessing import Process, Manager
    import time
    import asyncio
    import tqdm
    # 异步
    # reader = jsonlines.open("pubtabnet/PubTabNet_2.0.0.jsonl", 'r').iter()
    # reader = [reader.__next__() for i in range(10000)]
    # start = time.time()
    # asyncio.run(run(reader))
    # print("Cost:", time.time()-start)

    # 多线程
    # reader = jsonlines.open("pubtabnet/PubTabNet_2.0.0.jsonl", 'r').iter()
    # start = time.time()
    # main(reader)
    # print("Cost:", time.time()-start)
    func = lambda x: [trans(i) for i in tqdm.tqdm(x)]
    reader = list(jsonlines.open("pubtabnet/PubTabNet_2.0.0.jsonl", 'r').iter())
    start = time.time()
    pools = 5
    bs = len(reader) // pools
    pooling = []
    data = [reader[i:i+bs] for i in range(0, len(reader), bs)]
    for i in data:
        p = Process(target=func, args=(i, ))
        pooling.append(p)
        p.start()
    [p.join() for p in pooling]
    print("Cost:", time.time()-start)

    # 单线程
    # reader = jsonlines.open("pubtabnet/PubTabNet_2.0.0.jsonl", 'r').iter()
    # start = time.time()
    # list(map(trans, reader))
    # print("D Cost", time.time()-start)
