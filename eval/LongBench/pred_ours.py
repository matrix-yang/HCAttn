import os
from datasets import load_dataset
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
from duo_attn.patch.tuple_kv_cache import enable_tuple_kv_cache


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
    )
    parser.add_argument("--task", type=str, help="task name", default="none")
    parser.add_argument("--decoding_simulation_length", type=int, default=0)
    parser.add_argument("--attn_sum", type=float, default=0.5)
    parser.add_argument("--no_quant", action="store_true")
    parser.add_argument("--quant_path", type=str, default="none")
    parser.add_argument("--quant_dims", type=int, default=4)
    return parser.parse_args(args)


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "llama-2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    elif "llama-3" in model_name.lower():
        response = (
            response.split(".assistant")[0]
            .split("\n\nQuestion")[0]
            .split("</s>")[0]
            .strip()
        )
    elif "Llama-2-7B-32K-Instruct" in model_name:
        response = (
            response.split("(Document")[0]
            .split("\n\nQuestion")[0]
            .split("\n\nAnswer")[0]
            .split("(Passage")[0]
            .strip()
        )
    return response


def get_pred(
        model,
        tokenizer,
        eos_token_ids,
        data,
        max_length,
        max_gen,
        prompt_format,
        dataset,
        model_name,
        decoding_simulation_length,
        quanter
):
    preds = []
    pbar = tqdm(data)
    for idx, json_obj in enumerate(pbar):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in [
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "lcc",
            "repobench-p",
        ]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
        pbar.set_description(f"Generating for {idx}, len = {input.input_ids.shape[-1]}")
        simulation_start_idx = input.input_ids.shape[-1] - decoding_simulation_length
        with torch.no_grad():
            output = model(
                input_ids=input.input_ids[:, :simulation_start_idx],
                past_key_values=None,
                use_cache=True,
            )
            past_key_values = output.past_key_values

            if not args.no_quant:
                # print('---------------use quant------------------')
                past_key_values = quanter.quant(past_key_values)
                # print('mem use before clear', torch.cuda.memory_allocated())
                # torch.cuda.empty_cache()
                # print('mem use after clear',torch.cuda.memory_allocated())
            else:
                pass
                # print('---------------no quant------------------')

            if decoding_simulation_length > 0:
                for idx, input_id in enumerate(
                        input.input_ids[0, simulation_start_idx:]
                ):
                    output = model(
                        input_ids=input_id.unsqueeze(0).unsqueeze(0),
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = output.past_key_values

            pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_content = [pred_token_idx.item()]
            for _ in range(max_gen - 1):
                outputs = model(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                past_key_values = outputs.past_key_values
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_content += [pred_token_idx.item()]
                if pred_token_idx.item() in eos_token_ids:
                    break
        print('mem use after decoder', torch.cuda.memory_allocated())
        del past_key_values
        del outputs
        del pred_token_idx
        del input
        torch.cuda.empty_cache()
        print('del past_key_values after infer', torch.cuda.memory_allocated())

        pred = tokenizer.decode(generated_content, skip_special_tokens=True)
        pred = post_process(pred, model_name)
        print(f"Prediction: {pred}")
        preds.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
        )
    return preds


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


from spare_attn.solution2.build_model_chat import LLamaChat

if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    model2path = json.load(open("eval/LongBench/config/model2path.json", "r"))
    model2maxlen = json.load(open("eval/LongBench/config/model2maxlen.json", "r"))
    device_list = [i for i in range(torch.cuda.device_count())]
    model_name = args.model
    # define your model
    model_path = model2path[model_name]
    attn_sum = args.attn_sum
    quant_path = args.quant_path
    if quant_path is None:
        quant_path = '/ms/FM/ydq/notebook/duo_attn/no_norm_4bits_8196.npy'
    print(f'use quant {quant_path} quant dims {args.quant_dims}')
    llama_chat = LLamaChat(model_path, attn_sum, quant_path,modify=True,dims=args.quant_dims)
    model = llama_chat.model
    tokenizer = llama_chat.tokenizer
    eos_token_ids = llama_chat.eos_token_ids
    # model, tokenizer, eos_token_ids = load_model_and_tokenizer(
    #     model2path[model_name], model_name
    # )
    # model = to_device(model, device_list, enable_tp=True)
    max_length = model2maxlen[model_name]
    if args.task == 'none':
        datasets = [
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "lcc",
            "repobench-p",
        ]
    else:
        datasets = [args.task]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("eval/LongBench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("eval/LongBench/config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("eval/LongBench/pred"):
        os.makedirs("eval/LongBench/pred")
    if not os.path.exists("eval/LongBench/pred_e"):
        os.makedirs("eval/LongBench/pred_e")

    print('init mem use --', torch.cuda.memory_allocated())
    for dataset in datasets:
        lb_path = '/ms/FM/ydq/kvcache/duo-attention/THUDM/LongBench/LongBench.py'
        data = load_dataset(lb_path, dataset, split="test")
        if not os.path.exists(f"eval/LongBench/pred/{model_name}"):
            os.makedirs(f"eval/LongBench/pred/{model_name}")
        if not args.no_quant:
            quant_name=quant_path.split('/')[-1][:-4]
            out_path = f"eval/LongBench/pred/{model_name}/{dataset}-attn_{attn_sum}_{quant_name}.jsonl"
        else:
            out_path = f"eval/LongBench/pred/{model_name}/{dataset}-attn_{attn_sum}-no_quant.jsonl"

        if os.path.exists(out_path):
            print(f'{out_path} is exists pass dataset {dataset}')
            continue

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred(
            model,
            tokenizer,
            eos_token_ids,
            data,
            max_length,
            max_gen,
            prompt_format,
            dataset,
            model_name,
            args.decoding_simulation_length,
            quanter=llama_chat.quanter
        )
        print("{} ues cache radio: {:.5f}".format(dataset, np.mean(llama_chat.radio_bag)))
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write("\n")
    print("Average ues cache radio: {:.5f}".format(np.mean(llama_chat.radio_bag)))
