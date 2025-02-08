import argparse
import os
import numpy as np
import pandas as pd
import time

choices = ["A", "B", "C", "D"]


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

from tqdm import tqdm
def eval(ntrain, subject, engine, dev_df, test_df,choices_idx,model):
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1]-2]
    rrs=[]
    for i in tqdm(range(test_df.shape[0])):
        # get prompt and make sure it fits
        k = ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]
        c,logits=model.chat(prompt,max_gen=1)

        lprobs = logits[0][0,choices_idx]
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(lprobs)]
        probs = softmax(np.array(lprobs))

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return cors, acc, all_probs

def get_choices_idx(choices,tokenizer):
    words = choices
    word_ids = [tokenizer.encode(word, add_special_tokens=False) for word in words]
    idxs=[]
    for word, word_id in zip(words, word_ids):
        idxs.append(word_id[0])
    return idxs

from spare_attn.solution2.build_model_chat import LLamaChat

if __name__ == '__main__':
    #示例样本数
    ntrain = 5
    data_dir = '/ms/FM/ydq/kvcache/mmlu/data'
    save_dir = 'mmlu_results'
    engine = 'test'
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(data_dir, "test")) if "_test.csv" in f])

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    model_path='/ms/FM/ydq/kvcache/Llama-2-7B-32K-Instruct'

    #量化设置
    quant_path=None
    quant_dims =4
    #稀疏化设置
    modify=False
    attn_sum = 1

    llama_chat = LLamaChat(model_path, attn_sum, quant_path, modify=modify, dims=quant_dims)

    choices_idx = get_choices_idx(choices,llama_chat.tokenizer)


    if not os.path.exists(os.path.join(save_dir, "results_{}".format(engine))):
        os.mkdir(os.path.join(save_dir, "results_{}".format(engine)))

    all_cors = []
    for subject in subjects:
        dev_df = pd.read_csv(os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None)[:ntrain]
        test_df = pd.read_csv(os.path.join(data_dir, "test", subject + "_test.csv"), header=None)

        cors, acc, probs = eval(ntrain, subject, engine, dev_df, test_df, choices_idx,llama_chat)
        all_cors.append(cors)

        test_df["{}_correct".format(engine)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(engine, choice)] = probs[:, j]
        test_df.to_csv(os.path.join(save_dir, "results_{}".format(engine), "{}.csv".format(subject)), index=None)
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))