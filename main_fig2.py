import os
os.environ["http_proxy"] = "http://127.0.0.1:7890" 
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from runner import init, main
from modules import *

if __name__ == "__main__":
    # model configs
    # model_name = 'gpt2-xl'
    model_name = 'facebook/opt-6.7b'

    init(model_name, 'fig2.txt')

    tasks = []

    ratio_best_int4 = 0.002
    simplified_errs = [1e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2]

    # fig 2 (33 points)

    err_ons = ['outlier', 'non-outlier']

    tasks.append(([
        int4rquan(),
    ], 'r0.005_int4'))
    '''tasks.append(([
        olivewise(sigmaq(), encoding_1='flint'),
    ], 'os_flint'))
    tasks.append(([
        rquan(ratio=1, encoding_1='flint'),
    ], 'r1_flint'))'''

    for err in simplified_errs:
        for err_on in err_ons:
            tasks.append(([
                int4rquan(),
                adderrorq(error_percent=err, error_on=err_on),
            ], 'r0.005_int4'))
            '''
            tasks.append(([
                rquan(ratio=1, encoding_1='flint'),
                adderrorq(error_percent=err, error_on=err_on),
            ], 'r1_flint'))
            tasks.append(([
                olivewise(sigmaq(), encoding_1='flint'),
                adderrorq(error_percent=err, error_on=err_on),
            ], 'os_flint'))'''
    
    for _ in range(10):
        for task, key in tasks:
            main(task, key)
