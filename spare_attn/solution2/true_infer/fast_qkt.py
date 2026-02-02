import torch
import numpy as np
#from flash_attn import flash_attn_func, flash_attn_with_kvcache
import time


def fast_qK(q, ki, C):
    bsz,head,qn,_=q.shape
    g,cid_len,dim=C.shape
    k_len=ki.shape[-2]
    q = q.reshape(bsz * head * qn, g, dim).unsqueeze(-2)
    q_ = torch.matmul(q.float(), C.transpose(-1, -2).float())
    q_ = q_.reshape(bsz, head, qn, g, cid_len)
    q_expand = q_.unsqueeze(-3)
    q_expand = q_expand.expand(-1, -1, -1, k_len, -1, -1)
    ki_expand = ki.unsqueeze(-3).unsqueeze(-1).expand(-1, -1, qn, -1, -1, -1)
    ki_expand=ki_expand.to(torch.int64)
    qk = torch.gather(q_expand, dim=-1, index=ki_expand).squeeze(-1)
    attention_scores = qk.sum(-1) * 0.088388
    attention_probs = torch.softmax(attention_scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return attention_probs

def fast_qK_gqa(q, ki, C):
    bsz,head,qn,_=q.shape
    g,cid_len,dim=C.shape
    _,khead,k_len,_=ki.shape
    q = q.reshape(bsz * head,qn, g, dim).transpose(-2, -3)
    q_ = torch.matmul(q, C.transpose(-1, -2))
    #bsz*head,len,g,C_len
    q_=q_.transpose(-2, -3).reshape(bsz, head, qn, g, cid_len)
    q_expand = q_[:, :,:, None, :, :].expand(-1, -1, -1, k_len, -1, -1)
    ki=ki[:, :, None, :, :].expand(bsz, khead, int(head/khead), k_len, g)
    ki=ki.reshape(bsz, head, k_len, g)
    #bsz ,head,qn,kn,g,1
    ki_expand = ki.unsqueeze(-3).unsqueeze(-1).expand(-1, -1, qn, -1, -1, -1)
    ki_expand=ki_expand.to(torch.int64)
    qk = torch.gather(q_expand, dim=-1, index=ki_expand).squeeze(-1)
    attention_scores = qk.sum(-1) * 0.088388
    attention_probs = torch.softmax(attention_scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return attention_probs

def sparse_v_sum(atten, values, th):
    bsz, head, q_len, k_len = atten.shape
    dim=values.shape[-1]
    sort_attn, idx = torch.sort(atten, dim=-1, descending=True)
    cum_sum = torch.cumsum(sort_attn, dim=-1)
    topN = (cum_sum > th).int().argmax(dim=-1).max()
    select_attn = sort_attn[:, :, :, :topN].cpu()
    idx_cpu=idx[:, :, :, :topN].cpu()
    v_index = idx_cpu.unsqueeze(-1).expand(-1, -1, -1, -1, dim)
    values_expand = values.unsqueeze(-3).expand(-1, -1, q_len, -1, -1)
    select_value = torch.gather(values_expand, dim=-2, index=v_index)
    attn_output = torch.matmul(select_attn.unsqueeze(-2), select_value).squeeze(-2)
    return attn_output.cuda()

def sparse_v_sum_v2(attention_probs,value_states,sum_value):
    if sum_value>=1:
        attention_probs = attention_probs.cpu()
        attn_output = torch.matmul(attention_probs, value_states)
        return attn_output.cuda()
    attention_probs=attention_probs.squeeze(0)
    value_states = value_states.squeeze(0)
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
    # 按原始顺序去除这些索引
    # print(len(selected_indices),selected_indices)
    # print(selected_indices.shape)
    # print(value_states.shape)
    attn_weights = torch.gather(attention_probs, dim=-1, index=selected_indices).cpu()
    # print('attn_weights',attn_weights.shape)
    v_index = selected_indices.squeeze(1).unsqueeze(-1).expand(-1, -1, 128).cpu()
    # print('v_index',v_index.shape)
    select_value_states = torch.gather(value_states, dim=-2, index=v_index)
    # print(attn_weights.shape,value_states.shape)
    # print('select_value_states',v_index.shape)
    # print(select_value_states.shape,select_value_states[1,255]==value_states[1,v_index[1,255,0]])
    attn_output = torch.matmul(attn_weights, select_value_states)
    #attn_output2 = torch.bmm(attn_weights, select_value_states)
    #print(attn_output==attn_output2)
    attn_output = attn_output.unsqueeze(0).transpose(1, 2)
    # print(attn_output.shape)
    return attn_output.cuda()

def fast_fwd(q, ki, values,C,th):
    attention_probs = fast_qK(q, ki, C)
    #attn_out = sparse_v_sum(attention_probs, values, th=th)
    attn_out = sparse_v_sum_v2(attention_probs, values, th)
    return attn_out


def fast_qK_sim(q, ki, C):
    bsz,head,qn,_=q.shape
    g,cid_len,dim=C.shape
    _,khead,k_len,_=ki.shape
    attention_probs = torch.rand((bsz,head,qn,k_len),device='cuda',dtype=q.dtype)
    #attention_probs = torch.softmax(attention_scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return attention_probs

def fast_fwd_for_len(q, ki, values,C,th):
    # attention_probs = fast_qK_sim(q, ki, C)
    # attention_probs = attention_probs.cpu()
    batch, num_key_value_heads, slen, head_dim = values.shape

    values = values[:, :, None, :, :].expand(batch, num_key_value_heads, 4, slen, head_dim)
    values = values.reshape(batch, 32, slen, head_dim)

    end = int(0.16 * slen)
    # attention_probs = attention_probs[:, :, :, :end]
    values= values[:, :, :end,:]
    #attn_out = torch.matmul(attention_probs, values)
    attn_out = torch.zeros(q.shape, device='cuda', dtype=q.dtype)
    return attn_out
