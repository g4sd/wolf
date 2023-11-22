import os
os.environ["http_proxy"] = "http://127.0.0.1:7890" 
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

from runner import init, main
from modules import *

ratio_best = 0.0015
ratio_best_int4 = 0.002

if __name__ == "__main__":
    # model configs
    # model_name = 'gpt2-xl'
    model_name = 'facebook/opt-6.7b'

    init(model_name, 'fig4.1.txt')

    tasks = []

    errs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

    # fig 3 (20 points)

    for err in errs:
        tasks.append([
            rquan(ratio=ratio_best_int4, encoding_1='int4'),
            hashq(),
            adderror(error_percent=err)
        ])
        tasks.append([
            rquan(ratio=ratio_best_int4, encoding_1='int4'),
            hashq(),
            adderror(error_percent=err),
            fullerr()
        ])
        tasks.append([
            rquan(ratio=ratio_best_int4, encoding_1='int4'),
            hashq(),
            adderror(error_percent=err),
            weightzeroerr_v2()
        ])
        tasks.append([
            rquan(ratio=ratio_best_int4, encoding_1='int4'),
            hashq(),
            adderror(error_percent=err),
            wesco(col=32)
        ])
    
    for task in tasks:
        main(task, 'one')
