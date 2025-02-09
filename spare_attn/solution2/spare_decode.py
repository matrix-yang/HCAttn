import torch
import numpy as np
from flash_attn import flash_attn_func, flash_attn_with_kvcache


def decode_token(query_states_ori, key_states_ori, value_states_ori, sum_value, radio_bag):
    if sum_value >= 1:
        return decode_token_ori(query_states_ori, key_states_ori, value_states_ori)
    #print('----ori', query_states_ori.shape, key_states_ori.shape)
    query_states = query_states_ori.transpose(1, 2).squeeze(0)
    key_states = key_states_ori.transpose(1, 2).squeeze(0)
    value_states = value_states_ori.transpose(1, 2).squeeze(0)
    # bsz 1 heads 32 seqlen 128
    # print('----------------',query_states.device,query_states.shape,query_states.device,key_states.shape)
    q_heads = query_states.shape[0]
    k_heads = key_states.shape[0]
    seq_len = key_states.shape[1]
    dims= query_states.shape[2]
    if q_heads != k_heads:
        #print('--',query_states.shape,key_states.shape)
        nrep=int(q_heads/k_heads)
        key_states=key_states[:,None,:,:].expand(k_heads,nrep,seq_len,dims).reshape(q_heads, seq_len, dims)
        #print('--',key_states.shape)
        value_states = value_states[:,None,:,:].expand(k_heads,nrep,seq_len,dims).reshape(q_heads, seq_len, dims)
    attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * 0.088388

    #print('----------------', query_states.shape, key_states.shape, attention_scores.shape)
    # 32 1000
    attention_probs = torch.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    # print('attention_probs',attention_probs.shape)
    # 获取降序排序的索引
    sorted_indices = torch.argsort(attention_probs, descending=True, dim=-1)
    # print(sorted_indices.shape)
    # search_index_ls = [16, 32, 64, 96, 128, 256, 384, 512, 768, attention_probs.shape[-1]]
    step = max(1,int(attention_probs.shape[-1] / 100))
    search_index_ls = np.arange(1, attention_probs.shape[-1], step)
    search_index_ls = np.append(search_index_ls, attention_probs.shape[-1])
    for i in search_index_ls:
        result = torch.gather(attention_probs, dim=-1, index=sorted_indices[:, :, :i])
        # print(result.shape,result.sum(-1))
        #print('----------', sorted_indices.shape, result.shape)
        if all(result.sum(-1) >= sum_value):
            break
    selected_indices = sorted_indices[:, :, :i]
    # print('selected_indices',selected_indices)

    selected_indices = selected_indices.sort(dim=-1).values
    # print('selected_indices sort',selected_indices)
    # print('radio',i,key_states.shape[-2],i/key_states.shape[-2])
    radio_bag.append(i / key_states.shape[-2])
    # 按原始顺序去除这些索引
    # print(len(selected_indices),selected_indices)
    # print(selected_indices.shape)
    # print(value_states.shape)
    attn_weights = torch.gather(attention_probs, dim=-1, index=selected_indices)
    # print('attn_weights',attn_weights.shape)
    v_index = selected_indices.squeeze(1).unsqueeze(-1).expand(-1, -1, 128)
    # print('v_index',v_index.shape)
    select_value_states = torch.gather(value_states, dim=-2, index=v_index)
    # print(attn_weights.shape,value_states.shape)
    # print('select_value_states',v_index.shape)
    # print(select_value_states.shape,select_value_states[1,255]==value_states[1,v_index[1,255,0]])
    attn_output = torch.matmul(attn_weights, select_value_states)
    attn_output = attn_output.unsqueeze(0).transpose(1, 2)
    # print(attn_output.shape)
    return attn_output


def decode_token_ori(query_states, key_states, value_states):
    # query = query_states.transpose(1, 2)
    # key = key_states.transpose(1, 2)
    # value = value_states.transpose(1, 2)
    # # print(query_states.shape,key_states.shape)
    # # bsz 1 heads 32 seqlen 128
    #
    # attn_weights = torch.matmul(query, key.transpose(2, 3))
    # attn_weights /= 11.3137
    #
    # # 32 1000
    # attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    # attn_output = torch.matmul(attn_weights, value)
    # attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = flash_attn_func(
        query_states,
        key_states,
        value_states,
        causal=True,
        dropout_p=0.0,
    )
    return attn_output
