import os
os.environ["http_proxy"] = "http://127.0.0.1:7890" 
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from runner import init, main
from modules import *

if __name__ == "__main__":
    
    # model configs
    # model_name = 'gpt2-xl'
    model_name = 'facebook/opt-6.7b'
    
    init(model_name, 'fig1.txt')

    tasks = []

    # fig 1 (15 points)

    for ratio in [1e-8, 3e-7, 1e-7, 3e-6, 1e-6]:
        for percent in [1e-8]:
            if percent > ratio: continue
            tasks.append(([
                rquan(ratio=ratio, encoding_1='flint'),
                adderrorq(error_percent=percent, error_on='outlier'),
            ], f'ratio:{ratio}'))
            tasks.append(([
                rquan(ratio=ratio, encoding_1='int4'),
                adderrorq(error_percent=percent, error_on='outlier'),
            ], f'ratio:{ratio}'))

    for _ in range(10):
        for task, key in tasks:
            main(task, key)
