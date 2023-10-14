import os
os.environ["http_proxy"] = "http://127.0.0.1:7890" 
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ['CUDA_VISIBLE_DEVICES'] = '3,2'

from runner import init, main
from modules import *

if __name__ == "__main__":
    # model configs
    # model_name = 'gpt2-xl'
    model_name = 'facebook/opt-6.7b'

    init(model_name, 'fig3.txt')

    tasks = []

    ratio_best_int4 = 0.002
    simplified_errs = [1e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2]

    # fig 3 (20 points)

    for err in simplified_errs:
        for bit_high in [4, 3, 2, 1]:
            tasks.append([
                # sigmaq(encoding_1='flint'),                       
                int4rquan(),
                adderror(error_percent=err, error_high=bit_high, err_low=0),
            ])

    for _ in range(10):
        for task in tasks:
            main(task, 'one')
