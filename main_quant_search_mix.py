import os
os.environ["http_proxy"] = "http://127.0.0.1:7890" 
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

from runner import init, main
from modules import *

if __name__ == "__main__":
    # model configs
    # model_name = 'gpt2-xl'
    model_name = 'facebook/opt-6.7b'

    init(model_name, 'quant_search_mix.txt')

    tasks = []

    ratio = 0.001
    while ratio < 0.005:
        tasks.append(([
            rquan(ratio=ratio, encoding_1='int8'),
            hashq()
        ], f'{ratio}'))
        ratio += 0.0002
    
    for task, key in tasks:
        main(task, key)
