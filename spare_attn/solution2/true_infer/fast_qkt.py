import torch
import numpy as np
#from flash_attn import flash_attn_func, flash_attn_with_kvcache
import time


# def fast_qK(q, ki, C):
#     bsz,head,qn,_=q.shape
#     g,cid_len,dim=C.shape
#     k_len=ki.shape[-2]
#     q = q.reshape(bsz * head * qn, g, dim).unsqueeze(-2)
#     q_ = torch.matmul(q.float(), C.transpose(-1, -2).float())
#     q_ = q_.reshape(bsz, head, qn, g, cid_len)
#     q_expand = q_.unsqueeze(-3)
#     q_expand = q_expand.expand(-1, -1, -1, k_len, -1, -1)
#     ki_expand = ki.unsqueeze(-3).unsqueeze(-1).expand(-1, -1, qn, -1, -1, -1)
#     ki_expand=ki_expand.to(torch.int64)
#     qk = torch.gather(q_expand, dim=-1, index=ki_expand).squeeze(-1)
#     attention_scores = qk.sum(-1) * 0.088388
#     attention_probs = torch.softmax(attention_scores, dim=-1, dtype=torch.float32).to(q.dtype)
#     return attention_probs

def fast_qK(q, ki, C):
    bsz, head, qn, _ = q.shape
    g, cid_len, dim = C.shape
    k_len = ki.shape[-2]
    
    # 优化张量操作，减少中间张量
    q_reshaped = q.reshape(bsz * head * qn, g, dim)
    q_matmul = torch.matmul(q_reshaped, C.transpose(-1, -2))
    q_ = q_matmul.reshape(bsz, head, qn, g, cid_len)
    
    # 正确扩展维度
    q_expand = q_.unsqueeze(-3).expand(-1, -1, -1, k_len, -1, -1)
    ki_expand = ki.unsqueeze(-3).unsqueeze(-1).expand(-1, -1, qn, -1, -1, -1).to(torch.int64)
    
    # 使用更高效的 gather 操作
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
    """稀疏注意力计算，将挑选后的 attention_probs 传到 CPU 上计算，结果传回 GPU"""
    bsz, head, q_len, k_len = atten.shape
    dim = values.shape[-1]
    
    # 在 GPU 上进行排序和阈值选择
    sort_attn, idx = torch.sort(atten, descending=True, dim=-1)
    cum_sum = torch.cumsum(sort_attn, dim=-1)
    topN = (cum_sum > th).int().argmax(dim=-1).max()
    
    # 将挑选后的 attention_probs 和索引传到 CPU
    select_attn = sort_attn[:, :, :, :topN].cpu()
    idx_cpu = idx[:, :, :, :topN].cpu()
    
    # 在 CPU 上进行计算
    v_index = idx_cpu.unsqueeze(-1).expand(-1, -1, -1, -1, dim)
    values_expand = values.unsqueeze(-3).expand(-1, -1, q_len, -1, -1)
    select_value = torch.gather(values_expand, dim=-2, index=v_index)
    attn_output = torch.matmul(select_attn.unsqueeze(-2), select_value).squeeze(-2)
    
    # 将结果传回 GPU
    return attn_output.cuda()

def sparse_v_sum_v2(attention_probs, value_states, sum_value):
    """优化的稀疏注意力计算，减少 CPU-GPU 数据传输"""
    if sum_value >= 1:
        # 确保 value_states 在 GPU 上
        value_states_gpu = value_states.cuda()
        attn_output = torch.matmul(attention_probs, value_states_gpu)
        return attn_output
    
    # 保持在 GPU 上操作
    bsz, head, q_len, kv_len = attention_probs.shape
    dim = value_states.shape[-1]
    
    # 优化排序和选择过程
    sorted_indices = torch.argsort(attention_probs, descending=True, dim=-1)
    
    # 使用二分查找找到合适的阈值
    def find_threshold(attn_probs, target_sum):
        low, high = 1, attn_probs.shape[-1]
        best = high
        
        while low <= high:
            mid = (low + high) // 2
            topk_probs = torch.gather(attn_probs, dim=-1, index=sorted_indices[..., :mid])
            cum_sum = topk_probs.sum(dim=-1)
            
            if torch.all(cum_sum >= target_sum):
                best = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return best
    
    # 找到合适的阈值
    threshold = find_threshold(attention_probs, sum_value)
    
    # 选择前 threshold 个值
    selected_indices = sorted_indices[..., :threshold]
    selected_attn = torch.gather(attention_probs, dim=-1, index=selected_indices)
    
    # 确保 value_states 在 GPU 上
    value_states_gpu = value_states.cuda()
    
    # 扩展 value_states 维度以匹配
    value_expanded = value_states_gpu.unsqueeze(2).expand(-1, -1, q_len, -1, -1)
    selected_values = torch.gather(
        value_expanded, 
        dim=3, 
        index=selected_indices.unsqueeze(-1).expand(-1, -1, -1, -1, dim)
    )
    
    # 计算注意力输出
    attn_output = torch.matmul(selected_attn.unsqueeze(-2), selected_values).squeeze(-2)
    
    return attn_output

def fast_fwd(q, ki, values,C,th):
    attention_probs = fast_qK(q, ki, C)
    attn_out = sparse_v_sum(attention_probs, values, th=th)
    #attn_out = sparse_v_sum_v2(attention_probs, values, th)
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
