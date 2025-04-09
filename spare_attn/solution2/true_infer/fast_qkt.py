import torch
import numpy as np
from flash_attn import flash_attn_func, flash_attn_with_kvcache
import time


def fast_qK(q, ki, C):
    bsz,head,qn,_=q.shape
    g,cid_len,dim=C.shape
    k_len=ki.shape[-2]
    q = q.reshape(bsz * head * qn, g, dim).unsqueeze(-2)
    q_ = torch.matmul(q, C.transpose(-1, -2))
    q_ = q_.reshape(bsz, head, qn, g, cid_len)
    q_expand = q_.unsqueeze(-3)
    q_expand = q_expand.expand(-1, -1, -1, k_len, -1, -1)
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

def fast_fwd(q, ki, values,C,th):
    attention_probs = fast_qK(q, ki, C)
    attn_out = sparse_v_sum(attention_probs, values, th=th)
    return attn_out

