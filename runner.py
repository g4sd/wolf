import torch

from transformers.pytorch_utils import Conv1D
from datasets import load_dataset
from lm_eval import evaluator
from lm_evaluate_adaptor import LMEvalAdaptor

import gc, json
from modules import qmgr
import modules as ms

import os
os.environ["http_proxy"] = "http://101.6.70.170:7890" 
os.environ["https_proxy"] = "http://101.6.70.170:7890"

eval_tasks = None
model = None
tokenizer = None
g_model_name = None
org_sd = None

log = None

def print_(str):
    print(str)
    log.write(str + '\n')
    log.flush()

def print_task(task):
    str = json.dumps({module.__class__.__name__: {k: v for k, v in module.config.items() if k != 'device'} for module in task})
    print_(f'config = {str}')

TEST = True

def init(model_name, log_name):
    global eval_tasks, model, tokenizer, g_model_name, log, org_sd
    
    # model_name = 'bigscience/bloomz-7b1'
    eval_tasks = "piqa,arc_challenge,boolq"
    kwargs = {"torch_dtype": torch.float16}
    tokenizer_kwargs = {
        "use_fast": True,
        "revision": "main",
        "use_auth_token": None,
    }

    if TEST:
        import os
        eval_tasks = 'wikitext'
        #model_name = r'C:\Users\Administrator\.cache\huggingface\hub\models--facebook--opt-125m\snapshots\3d2b5f275bdf882b8775f902e1bfdb790e2cfc32'
        model_name = 'facebook/opt-125m'
        #os.environ["http_proxy"] = "http://127.0.0.1:1080" 
        #os.environ["https_proxy"] = "http://127.0.0.1:1080"

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        ms.g_device = torch.device('cuda:0')
        
        from transformers.models.opt.modeling_opt import OPTForCausalLM
        from transformers import AutoTokenizer
        print("Loading model...")
        model = OPTForCausalLM.from_pretrained(model_name, **kwargs).cuda('cuda:0')
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

    # load model and tokenizer
    elif model_name == "bigscience/bloomz-7b1":
        model_name = '/home/cc/huggingface/hub/models--bigscience--bloomz-7b1/snapshots/2f4c4f3ebcf171dbbe2bae989ea2d2f3d3486a97/'
        from transformers.models.bloom.modeling_bloom import BloomForCausalLM
        from transformers.models.bloom.tokenization_bloom_fast import BloomTokenizerFast
        print("Loading model...")
        model = BloomForCausalLM.from_pretrained(model_name, **kwargs).cuda('cuda:0')
        print("Loading tokenizer...")
        tokenizer = BloomTokenizerFast.from_pretrained(model_name, **tokenizer_kwargs)
    elif model_name == "gpt2-xl":
        model_name = '/home/cc/huggingface/hub/models--gpt2-xl/snapshots/33cdb5c0db5423c1879b1b9f16c352988e8754a8/'
        from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
        from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
        print("Loading model...")
        model = GPT2LMHeadModel.from_pretrained(model_name, **kwargs).cuda('cuda:0')
        print("Loading tokenizer...")
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name, **tokenizer_kwargs)
    elif model_name == "facebook/opt-6.7b" or model_name == "facebook/opt-125m":
        if model_name == 'facebook/opt-125m':
            model_name = '/home/cc/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6/'
        else:
            model_name = '/home/cc/huggingface/hub/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0/'
        from transformers.models.opt.modeling_opt import OPTForCausalLM
        from transformers import AutoTokenizer
        print("Loading model...")
        model = OPTForCausalLM.from_pretrained(model_name, **kwargs).cuda('cuda:0')
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    
    g_model_name = model_name
    log = open(log_name, 'a')
    org_sd = {k: v.cpu() for k, v in model.state_dict().items()}

state_dict = {}

def main(modules, cacheKey):

    print_task(modules)
    
    for name, m in model.named_modules():
        if 'lm_head' in name:
            continue
        if isinstance(m, torch.nn.Linear) or isinstance(m, Conv1D):
            
            key = f'{cacheKey}_{name}'
            
            q = qmgr(modules, print_)
            
            q.state = state_dict.get(key, None)
            
            state = q.process(m.weight.data.to(ms.g_device))
            
            state_dict[key] = q.state
            
            val = state['tensor'].to('cuda:0').to(torch.float16)
            
            qmse = (m.weight.data - val) ** 2
            
            print_(f'{name} qmse: {qmse.max()}, {qmse.mean()}')

            m.weight.data = val
            
            del state
            gc.collect()
    
    lm_eval_model = LMEvalAdaptor(g_model_name, model, tokenizer)
    # evaluation function
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=eval_tasks.split(","),
        batch_size=1,
        no_cache=True,
        num_fewshot=0
    )

    print_(evaluator.make_table(results))
    
    model.load_state_dict(org_sd)

def eval():
    lm_eval_model = LMEvalAdaptor(g_model_name, model, tokenizer)
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=eval_tasks.split(","),
        batch_size=1,
        no_cache=True,
        num_fewshot=0,
    )