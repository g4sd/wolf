import os
os.environ["http_proxy"] = "http://127.0.0.1:7890" 
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ['CUDA_VISIBLE_DEVICES'] = '3,2'

from runner import init, main
from modules import *

if __name__ == "__main__":
    # model configs
    # model_name = 'gpt2-xl'
    model_name = 'gpt2-xl'

    init(model_name, 'gpt.txt')

    main([
        rquan(ratio=0.005, encoding_1='int4'),
        hashq()
    ], 'int4')

    main([
        rquan(ratio=0.005, encoding_1='flint'),
        hashq()
    ], 'flint')

    main([
        rquan(ratio=0.005, encoding_1='int8'),
        hashq()
    ], 'mix')

    main([
        rquan(ratio=0.005, encoding_1='int8', encoding_2='int8'),
        hashq()
    ], 'int8')
