from spare_attn.solution2.build_model_chat import LLamaChat
from spare_attn.solution2.modeify_llama import enable_llama_approx_attention_eval
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uuid
from datasets import load_dataset
from tqdm import tqdm

from spare_attn.solution2.build_model_chat import LLamaChat
from spare_attn.solution2.modeify_llama import enable_llama_approx_attention_eval
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uuid
from datasets import load_dataset
import numpy as np


def gen_kv_cache(prompt, model, tokenizer, max_length=32000, sp='kv_cache_dir'):
    tokenized_prompt = tokenizer(
        prompt, truncation=False, return_tensors="pt"
    ).input_ids[0]
    if len(tokenized_prompt) > max_length:
        half = int(max_length / 2)
        prompt = tokenizer.decode(
            tokenized_prompt[:half], skip_special_tokens=True
        ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
    input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model(
            input_ids=input.input_ids[:, :],
            past_key_values=None,
            use_cache=True,
        )
        past_key_values = output.past_key_values
    # layer kv bsz
    save_kv(past_key_values, sp)


def save_kv(kvcache, sp):
    name = str(uuid.uuid4()) + '.npy'
    tensor = []
    for l in kvcache:
        # print(l.shape)
        # L[0]只存储K
        tensor.append(l[0])
    lkv = torch.stack(tensor).cpu().to(float).numpy()
    # np.save('/nfs/FM/ydq/kv_cache/'+name, lkv)
    np.save(f'./{sp}/' + name, lkv)
    return name


if __name__ == '__main__':

    # model_name = "/ms/FM/ydq/kvcache/Qwen2.5-7B-Instruct-1M"
    model_name = "/ms/FM/ydq/kvcache/Llama-2-7B-32K-Instruct"
    cache_save_path='Llama-2-7B-32K-Instruct_k_cache'
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        _attn_implementation='eager'
    )
    model = model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    radio_bag = []
    enable_llama_approx_attention_eval(model, attn_sum=1, radio_bag=radio_bag)
    p = '/nfs/hw-data/ms/FM/ydq/kvcache/duo-attention/THUDM/LongBench/LongBench.py'
    data = load_dataset(p, 'narrativeqa', split="test")
    print('len ', len(data))
    for i in tqdm(range(3)):
        d = data[i]
        gen_kv_cache(d['context'], model, tokenizer, sp=cache_save_path)
