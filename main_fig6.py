import os
os.environ["http_proxy"] = "http://127.0.0.1:7890" 
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

from runner import init, main
from modules import *

ratio_best = 0.0015
ratio_best_int4 = 0.002

if __name__ == "__main__":
    # model configs
    # model_name = 'gpt2-xl'
    model_name = 'facebook/opt-6.7b'

    init(model_name, 'fig6.txt')

    tasks = []

    simplified_errs = [0, 1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2]
    
    tasks.append([
        rquan(ratio=ratio_best_int4, encoding_1='int4'),
        hashq()
    ])
    
    for err in simplified_errs:
        for bit_low in [3, 2, 1]:
            tasks.append([
                rquan(ratio=ratio_best_int4, encoding_1='int4'),
                hashq(),
                adderror(error_percent=err),
                fullerr(correct_error_low=bit_low)
            ])


    for task in tasks:
        main(task, 'one')
