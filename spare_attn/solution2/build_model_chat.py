import sys
sys.path.append('/nfs/hw-data/ms/FM/ydq/kvcache/duo-attention/')
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from tqdm import tqdm
import numpy as np
import random
import argparse

from duo_attn.patch import enable_duo_attention_eval

from duo_attn.utils import (
    to_device,
    load_attn_pattern,
    sparsify_attention_heads,
)
from duo_attn.patch.tuple_kv_cache import enable_tuple_kv_cache, enable_tuple_kv_cache_for_llama
import time


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    generation_config = GenerationConfig.from_pretrained(path)
    eos_token_ids = generation_config.eos_token_id
    if not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]

    model = model.eval()
    return model, tokenizer, eos_token_ids

from spare_attn.solution2.simulation_quant_k import Quanter
class LLamaChat:
    def __init__(self,model_path,attn_sum,quant_path,modify,dims=4):
        seed_everything(42)
        device_list = [0]
        #model_path='/nfs/hw-data/ms/FM/ydq/kvcache/Llama-2-7b-chat-hf'
        # llama_model2path='/nfs/hw-data/ms/FM/ydq/kvcache/Llama-2-7B-32K-Instruct'
        self.model_path=model_path
        self.radio_bag=[]
        self.modify=modify
        model, tokenizer, eos_token_ids = load_model_and_tokenizer(model_path)
        #TOVA需要
        self.init_past_key_values=None
        if modify=="ours":
            from spare_attn.solution2.modeify_llama import enable_llama_approx_attention_eval
            enable_llama_approx_attention_eval(model,attn_sum=attn_sum,radio_bag=self.radio_bag)
        elif modify=="sllm":
            from spare_attn.solution2.sllm_modeify_llama import enable_llama_approx_attention_eval
            enable_llama_approx_attention_eval(model, attn_sum=attn_sum, radio_bag=self.radio_bag)
        elif modify=="h2o":
            from spare_attn.solution2.h2o_modeify_llama import enable_llama_approx_attention_eval
            enable_llama_approx_attention_eval(model, attn_sum=attn_sum, radio_bag=self.radio_bag)
        elif modify=="tova":
            from spare_attn.solution2.TOVA import TOVACache, enable_tova_caching
            enable_tova_caching(model)
            # cache_size meaning min size
            self.init_past_key_values =TOVACache(cache_size=128,radio=0.5)
        else:
            print("-"*50)
            print("this test dont modify attn fwd")

        self.model = to_device(model, device_list, enable_tp=True)
        self.tokenizer = tokenizer
        self.eos_token_ids = eos_token_ids

        if quant_path:
            self.quanter = Quanter(quant_path,dims)
            self.is_quant=True
        else:
            self.quanter = None
            self.is_quant = False

    def chat(self, prompt, max_gen=2, decoding_simulation_length=1):
        #prompt = build_chat(prompt)
        input = self.tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
        simulation_start_idx = input.input_ids.shape[-1] - decoding_simulation_length


        with torch.no_grad():
            output = self.model(
                input_ids=input.input_ids[:, :simulation_start_idx],
                past_key_values=self.init_past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            # 压缩
            # s = time.time()
            if self.is_quant:
                past_key_values=self.quanter.quant(past_key_values)
            # e = time.time()
            # print('----------',e-s)
            if decoding_simulation_length > 0:
                for idx, input_id in enumerate(input.input_ids[0, simulation_start_idx:]):
                    output = self.model(
                        input_ids=input_id.unsqueeze(0).unsqueeze(0),
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = output.past_key_values
            logit = output.logits[:, -1, :]
            pred_token_idx = logit.argmax(dim=-1).unsqueeze(1)
            generated_content = [pred_token_idx.item()]
            generated_logits = [logit.detach().cpu().numpy()]
            for _ in range(max_gen - 1):
                outputs = self.model(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                logit = outputs.logits[:, -1, :]
                pred_token_idx = logit.argmax(dim=-1).unsqueeze(1)
                generated_content += [pred_token_idx.item()]
                generated_logits += [logit.detach().cpu().numpy()]
                if pred_token_idx.item() in self.eos_token_ids:
                    break

        pred = self.tokenizer.decode(generated_content, skip_special_tokens=True)
        return pred, generated_logits

    def attn_analysis(self, prompt, decoding_simulation_length=1):
        #prompt = build_chat(prompt)
        input = self.tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
        simulation_start_idx = input.input_ids.shape[-1] - decoding_simulation_length
        with torch.no_grad():
            output = self.model(
                input_ids=input.input_ids[:, :simulation_start_idx],
                past_key_values=None,
                use_cache=True,
                output_attentions=True
            )
            past_key_values = output.past_key_values

            attns=output.attentions
            # 压缩
            # s = time.time()
            if self.is_quant:
                past_key_values=self.quanter.quant(past_key_values)
            # e = time.time()
            # print('----------',e-s)
            attn_list=[]
            if decoding_simulation_length > 0:
                for idx, input_id in enumerate(input.input_ids[0, simulation_start_idx:]):
                    output = self.model(
                        input_ids=input_id.unsqueeze(0).unsqueeze(0),
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_attentions=True
                    )
                    past_key_values = output.past_key_values
                    attn_list.append(output.attentions)
        return attns,attn_list

class QwenChat(LLamaChat):

    def __init__(self,model_path,attn_sum,quant_path,modify=True,dims=4):
        seed_everything(42)
        device_list = [0]
        #model_path='/nfs/hw-data/ms/FM/ydq/kvcache/Llama-2-7b-chat-hf'
        # llama_model2path='/nfs/hw-data/ms/FM/ydq/kvcache/Llama-2-7B-32K-Instruct'
        self.model_path=model_path
        self.radio_bag=[]
        model, tokenizer, eos_token_ids = load_model_and_tokenizer(model_path)
        if modify:
            from spare_attn.solution2.modeify_qwen import enable_qwen_approx_attention_eval
            enable_qwen_approx_attention_eval(model,attn_sum=attn_sum,radio_bag=self.radio_bag)
        self.model = to_device(model, device_list, enable_tp=True)
        self.tokenizer = tokenizer
        self.eos_token_ids = eos_token_ids
        if quant_path:
            self.quanter = Quanter(quant_path,dims)
            self.is_quant=True
        else:
            self.is_quant = False

    def chat(self, prompt, max_gen=2, decoding_simulation_length=1):
        #prompt = build_chat(prompt)
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        input = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        simulation_start_idx = input.input_ids.shape[-1] - decoding_simulation_length
        with torch.no_grad():
            output = self.model(
                input_ids=input.input_ids[:, :simulation_start_idx],
                past_key_values=None,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            # 压缩
            # s = time.time()
            if self.is_quant:
                past_key_values=self.quanter.quant(past_key_values)
            # e = time.time()
            # print('----------',e-s)
            if decoding_simulation_length > 0:
                for idx, input_id in enumerate(input.input_ids[0, simulation_start_idx:]):
                    output = self.model(
                        input_ids=input_id.unsqueeze(0).unsqueeze(0),
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = output.past_key_values
            logit = output.logits[:, -1, :]
            pred_token_idx = logit.argmax(dim=-1).unsqueeze(1)
            generated_content = [pred_token_idx.item()]
            generated_logits = [logit.detach().cpu().numpy()]
            for _ in range(max_gen - 1):
                outputs = self.model(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                logit = outputs.logits[:, -1, :]
                pred_token_idx = logit.argmax(dim=-1).unsqueeze(1)
                generated_content += [pred_token_idx.item()]
                generated_logits += [logit.detach().cpu().numpy()]
                if pred_token_idx.item() in self.eos_token_ids:
                    break

        pred = self.tokenizer.decode(generated_content, skip_special_tokens=True)
        return pred, generated_logits