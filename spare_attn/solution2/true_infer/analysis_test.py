import os
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
from fast_qkt import fast_fwd
from spare_attn.solution2.quanters.ki_quanter import MultiGroupQuanter
import time

def get_rand_tensor(shape, dtype=torch.bfloat16, device='cuda'):
    return torch.rand(shape, device=device, dtype=dtype)

def prepare_inputs():
    bsz, head, kv_len, head_dim = 1,32,512,128
    group_num,vector_len,vectir_dim = 32,4096,4
    C=(group_num,vector_len,vectir_dim)
    return get_rand_tensor((bsz, head, 1, head_dim)), \
           get_rand_tensor((bsz, head, kv_len, head_dim)), \
           get_rand_tensor((bsz, head, kv_len, head_dim)), \
           get_rand_tensor((bsz, head, kv_len, head_dim)), \
           get_rand_tensor(C)


def native_torch_attention(q, k, v):
    head_dim = q.size(-1)
    scale = head_dim ** -0.5
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output, attn_weights


if __name__ == '__main__':
    device='cuda'
    Cdtype=torch.bfloat16
    nq,q, k, v, C, = prepare_inputs()
    q1=q[:,:,-1:,:]
    # Test native torch attention with q, k, v
    a=time.time()
    attn_output, attn_weights = native_torch_attention(q1, k, v)
    b=time.time()
    print(f"Native torch attention time: {b-a}")

    # Convert k to ki
    qunater=MultiGroupQuanter(C,C_device=device,C_dtype=Cdtype)
    ki = qunater.quant(k)
    vcpu=v.cpu()
    a=time.time()
    fast_output = fast_fwd(q1, ki, vcpu, C, th=0.9)
    b=time.time()
    print(f"Fast forward attention time: {b-a}")
