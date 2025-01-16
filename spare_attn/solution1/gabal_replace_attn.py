from numba import jit
import numpy as np
from numba.typed import List
from numba import njit, prange
import time
import torch
import torch.nn.functional as F


def cal_key_cosine_fast(key_states):
    # bsz,32, 1343, 128
    size = key_states.shape
    key_states_copy = key_states.reshape((size[0] * size[1], size[2], size[3]))
    key_states_copy = F.normalize(key_states_copy, p=2, dim=-1)
    # 32*1343 128
    # hs=hs.reshape((size[0]*size[1],size[2]))
    sim = torch.zeros((size[0] * size[1], size[2], size[2]))
    chunk_size=16
    for i in range(0,size[2],chunk_size):
        chunk=key_states_copy[:,i:i+chunk_size,:]
        chunk = F.normalize(chunk, p=2, dim=-1)
        simt = torch.bmm(chunk, key_states_copy.transpose(1, 2))
        #print(simt.shape,sim[:,i:i+32,i:i+32])
        sim[:,i:i+chunk_size,:]=simt.cpu()

    rs = (1 - sim)
    # print(rs[0,0,:20,:20])
    return rs


def cal_key_sim_fast(key_states):
    # bsz,32, 1343, 128
    size = key_states.shape
    key_states_copy = key_states.reshape((size[0] * size[1], size[2], size[3]))
    # 32*1343 128
    # hs=hs.reshape((size[0]*size[1],size[2]))
    sim = torch.zeros((size[0] * size[1], size[2], size[2]))
    for i in range(0,size[2],16):
        chunk=key_states_copy[:,i:i+32,:]
        chunk = F.normalize(chunk, p=2, dim=-1)
        simt = torch.bmm(chunk, chunk.transpose(1, 2))
        #print(simt.shape,sim[:,i:i+32,i:i+32])
        sim[:,i:i+32,i:i+32]=simt.cpu()

    # key_states_copy = F.normalize(key_states_copy, p=2, dim=-1)
    # sim = torch.bmm(key_states_copy, key_states_copy.transpose(1, 2))
    # sim = sim.reshape(size[0], size[1], size[2], size[2])
    # 模长的差
    dis = torch.zeros((size[0] * size[1], size[2], size[2]))
    for i in range(0,size[2],16):
        chunk = key_states_copy[:, i:i + 32, :]
        key_states_mod = torch.norm(chunk, p=2, dim=-1)
        dist = (key_states_mod.unsqueeze(1) - key_states_mod.unsqueeze(2)).abs()
        dis[:,i:i+32,i:i+32]=dist.cpu()
    ##bsz,32, 1343, 128
    # key_states=key_states.detach()
    # dis =(key_states.unsqueeze(2) - key_states.unsqueeze(3)).sum(-1)
    rs = (1 - sim) * dis
    # print(rs[0,0,:20,:20])
    return rs

@njit
def compute_mean1(arr):
    # 获取数组的形状
    rows = arr.shape[0]
    cols = arr.shape[1]

    # 初始化结果数组
    result = np.zeros(rows, dtype=arr.dtype)

    # 计算沿着最后一个轴的平均值
    for i in range(rows):
        for j in range(cols):
            result[i] += arr[i, j]

    # 除以列数以得到平均值
    for i in range(rows):
        result[i] /= cols

    return result


@njit
def compute_mean2(arr):
    return np.mean(arr)


@njit
def argmax(arr):
    # 假设arr是一维数组
    max_index = 0
    max_value = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > max_value:
            max_value = arr[i]
            max_index = i
    return max_index


@jit(nopython=True)
def guess_window2(sim, s, l, th):
    for tl in range(l, 0, -1):
        chunk = sim[s:s + tl, s:s + tl]
        mni = compute_mean1(chunk)
        mn = compute_mean2(mni)
        if mn < th:
            return chunk.shape[0], s + argmax(mni)
        elif tl == 1:
            return tl, s
    return 1, s


@jit(nopython=True)
def get_head_repeat_index(mat, th=2, max_len=16, sink=8, recent=128):
    wls = List()  # 假设wls的最大长度等于mat的第一个维度
    tis = List()  # 假设tis的最大长度等于mat的第一个维度
    size = mat.shape
    index = sink
    while index < size[-1] - recent:
        guess_l = min(max_len, size[-1] - recent - index)
        wl, ti = guess_window2(mat, index, guess_l, th)
        tis.append(ti)
        wls.append(wl)
        index += wl

    # print(wls)
    assert sum(wls) == size[-1] - recent - sink
    zip_ratio = (len(wls) + recent + sink) / size[-1]
    expanded_list = [value for value, repeat in zip(tis, wls) for _ in range(repeat)]
    expanded_list = [i for i in range(sink)] + expanded_list + [i for i in range(size[-1] - recent, size[-1], 1)]
    return expanded_list, zip_ratio


@jit(nopython=True)
def get_clone_index(sim, th):
    size = sim.shape
    sim = sim.reshape(-1, size[-2], size[-1])
    repeat = List()
    radio = List()
    for i in range(sim.shape[0]):
        row=sim[i]

        expanded_list, zip_ratio = get_head_repeat_index(sim[i], th)
        repeat.append(expanded_list)
        radio.append(zip_ratio)
    return repeat, radio

@jit(nopython=True)
def get_head_repeat_index(mat, th=2, sink=8, recent=128):
    size=mat.shape
    mat=mat[sink:size[1]-recent,sink:size[1]-recent]
    mids=List()
    for m in mat:
        idxs = List()
        for i,ele in enumerate(m):
            if ele<th:
               idxs.append(i)
        mids.append(idxs)

    repeat_index=np.full(size[-1]-recent-sink, -1)
    now_id=0
    for l in mids:
        for i in l:
            if repeat_index[i] ==-1:
                repeat_index[i]=now_id
        now_id+=1
    repeat_index_offset=repeat_index+sink
    repeat_index_final=[i for i in range(sink)]+[i for i in repeat_index_offset]+[i for i in range(size[-1] - recent, size[-1], 1)]
    return repeat_index_final


@jit(nopython=True)
def get_clone_index(mat, th):
    size=mat.shape
    repeat=np.full(mat.shape[:2], -1)
    print(repeat.shape)
    for i in range(size[0]):
        mids=get_head_repeat_index(mat[i], th=0.5,sink=8, recent=8)
        #mids=get_index2(mat[i],th)
        repeat[i]=mids
    return repeat

def modify_cache_by_kcos(past_key_values, dist_th=1):
    new_past_key_values = []
    # bsz head len 128
    k = torch.cat([past_key_values[i][0] for i in range(32)])
    v = torch.cat([past_key_values[i][1] for i in range(32)])
    a = time.time()
    s = cal_key_cosine_fast(k).detach().to(torch.float).cpu().numpy()
    b = time.time()
    # bsz head len
    repeat, radio = get_clone_index(s, th=dist_th)
    c = time.time()
    repeat = torch.tensor(repeat).reshape(k.shape[:-1]).cuda()
    selected_k = torch.gather(k, 2, repeat.unsqueeze(-1).expand(-1, -1, -1, 128)).cuda()
    selected_v = torch.gather(v, 2, repeat.unsqueeze(-1).expand(-1, -1, -1, 128)).cuda()
    # unsqueeze bsz
    new_past_key_values = [(selected_k[i].unsqueeze(0), selected_v[i].unsqueeze(0)) for i in range(selected_k.shape[0])]
    d = time.time()
    #print(f'K shape {k.shape} cal sim {b - a} cal cmp {c - b} replace{d - c} all {d - a}')
    return new_past_key_values, np.mean(radio)
