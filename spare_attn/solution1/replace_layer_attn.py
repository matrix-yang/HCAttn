from numba import jit
import numpy as np
from numba.typed import List
from numba import njit, prange
import time
import torch
import torch.nn.functional as F


def modify_cache_by_layer(past_key_values, dist_th=1):
    new_past_key_values = []
    # bsz head len 128
    k = torch.cat([past_key_values[i][0] for i in range(32)])
    v = torch.cat([past_key_values[i][1] for i in range(32)])

    selected_k = []
    for i in range(32):
        if i % 2 == 0:
            selected_k.append(k[i])
            selected_k.append(k[i])

    selected_k = torch.stack(selected_k, dim=0)
    selected_v = v
    # unsqueeze bsz
    new_past_key_values = [(selected_k[i].unsqueeze(0), selected_v[i].unsqueeze(0)) for i in range(selected_k.shape[0])]
    d = time.time()
    radio=0.5
    #print(f'K shape {k.shape} cal sim {b - a} cal cmp {c - b} replace{d - c} all {d - a}')
    return new_past_key_values, np.mean(radio)
