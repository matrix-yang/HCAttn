from numba import jit
import numpy as np
from numba.typed import List
from numba import njit, prange
import time
import torch
import torch.nn.functional as F

def cal_key_sim(key_states, q_len_dim=2):
    # 模长的差
    #key_states = key_states.detach().cpu()
    #dis = key_states.unsqueeze(q_len_dim) - key_states.unsqueeze(q_len_dim + 1)
    #dis_key = torch.norm(dis, p=2, dim=-1)
    chunks = key_states.chunk(64, dim=0)
    dis_list = []
    for chunk in chunks:
        dis = chunk.unsqueeze(q_len_dim).detach().cpu() - chunk.unsqueeze(q_len_dim + 1).detach().cpu()
        dis_norm = torch.norm(dis, p=2, dim=-1)
        dis_list.append(dis_norm)
    dis_key = torch.cat(dis_list, dim=0)
    return dis_key

def cal_key_diff_mod_fast(key_states):
    # bsz,32, 1343, 128
    size = key_states.shape
    key_states_copy = key_states.reshape((size[0] * size[1], size[2], size[3]))
    # 32*1343 128
    # hs=hs.reshape((size[0]*size[1],size[2]))
    sim = torch.zeros((size[0] * size[1], size[2], size[2]))
    for i in range(0,size[2],16):
        chunk=key_states_copy[:,i:i+32,:]
        dist = chunk.unsqueeze(1) - chunk.unsqueeze(2)
        #print(dist.shape, sim.shape)
        dist =torch.norm(dist, p=2, dim=-1)
        #print(dist.shape,sim.shape)
        sim[:,i:i+32,i:i+32]=dist.cpu()
    return sim

def cal_key_value_sim(key_states, value_states):
    # bsz,32, 1343, 128
    size = key_states.shape
    key_states_copy = key_states.detach().cpu().reshape((size[0] * size[1], size[2], size[3]))
    # 32*1343 128
    # hs=hs.reshape((size[0]*size[1],size[2]))
    key_states_copy = F.normalize(key_states_copy, p=2, dim=-1)
    sim = torch.bmm(key_states_copy, key_states_copy.transpose(1, 2))
    sim = sim.reshape(size[0], size[1], size[2], size[2])
    # 模长的差
    key_states_copy1 = key_states.detach()
    key_states_mod = torch.norm(key_states_copy1, p=2, dim=-1)
    dis = (key_states_mod.unsqueeze(2) - key_states_mod.unsqueeze(3)).abs()
    # v 的l1的绝对值的距离
    vc = value_states.detach().cpu()
    v = vc.unsqueeze(2) - vc.unsqueeze(3)
    vl1 = v.abs().mean(dim=-1)

    ##bsz,32, 1343, 128
    # key_states=key_states.detach()
    # dis =(key_states.unsqueeze(2) - key_states.unsqueeze(3)).sum(-1)
    rs = (1 - sim) * dis * vl1.to(dis.device)
    # print(rs[0,0,:20,:20])
    return rs

def cal_key_cosine_fast(key_states):
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


def cal_key_value_sim_fast(key_states, value_states):
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

    # v 的l1的绝对值的距离
    value_states_copy = value_states.reshape((size[0] * size[1], size[2], size[3]))
    vl1 = torch.zeros((size[0] * size[1], size[2], size[2]))
    for i in range(0, size[2], 16):
        vc=value_states_copy[:, i:i + 32, :]
        v = vc.unsqueeze(1) - vc.unsqueeze(2)
        vl1t = v.abs().mean(dim=-1)
        vl1[:,i:i+32,i:i+32]=vl1t.cpu()

    ##bsz,32, 1343, 128
    # key_states=key_states.detach()
    # dis =(key_states.unsqueeze(2) - key_states.unsqueeze(3)).sum(-1)
    rs = (1 - sim) * dis * vl1.to(dis.device)
    # print(rs[0,0,:20,:20])
    return rs

def cal_key_sim_or(key_states):
    # bsz,32, 1343, 128
    size = key_states.shape
    key_states_copy = key_states.reshape((size[0] * size[1], size[2], size[3]))
    # 32*1343 128
    # hs=hs.reshape((size[0]*size[1],size[2]))
    sim = torch.zeros((size[0] * size[1], size[2], size[2]))
    diff_mod = torch.zeros((size[0] * size[1], size[2], size[2]))
    for i in range(0,size[2],16):
        chunk=key_states_copy[:,i:i+32,:]
        chunk = F.normalize(chunk, p=2, dim=-1)
        simt = torch.bmm(chunk, chunk.transpose(1, 2))
        #print(simt.shape,sim[:,i:i+32,i:i+32])
        sim[:,i:i+32,i:i+32]=simt.cpu()

        dis = chunk.unsqueeze(1) - chunk.unsqueeze(2)
        diff_mod[:,i:i+32,i:i+32] = torch.norm(dis, p=2, dim=-1).cpu()

    rs = (1 - sim) * diff_mod
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
def get_head_repeat_index(mat, th=2, max_len=16, sink=32, recent=128):
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
        expanded_list, zip_ratio = get_head_repeat_index(sim[i], th)
        repeat.append(expanded_list)
        radio.append(zip_ratio)
    return repeat, radio





def modify_cache(past_key_values, dist_th=1):
    new_past_key_values = []
    # bsz head len 128
    k = torch.cat([past_key_values[i][0] for i in range(32)])
    v = torch.cat([past_key_values[i][1] for i in range(32)])
    a = time.time()
    s = cal_key_diff_mod_fast(k).detach().to(torch.float).cpu().numpy()
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
    print(f'K shape {k.shape} cal sim {b - a} cal cmp {c - b} replace{d - c} all {d - a}')
    return new_past_key_values, np.mean(radio)

def modify_cache_by_kv(past_key_values, dist_th=1):
    new_past_key_values = []
    # bsz head len 128
    k = torch.cat([past_key_values[i][0] for i in range(32)])
    v = torch.cat([past_key_values[i][1] for i in range(32)])
    a = time.time()
    s = cal_key_value_sim_fast(k,v).detach().to(torch.float).cpu().numpy()
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
    print(f'K shape {k.shape} cal sim {b - a} cal cmp {c - b} replace{d - c} all {d - a}')
    return new_past_key_values, np.mean(radio)

def modify_cache_by_kfast(past_key_values, dist_th=1):
    new_past_key_values = []
    # bsz head len 128
    k = torch.cat([past_key_values[i][0] for i in range(32)])
    v = torch.cat([past_key_values[i][1] for i in range(32)])
    a = time.time()
    s = cal_key_sim_fast(k).detach().to(torch.float).cpu().numpy()
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

def modify_k_by_kfast(past_key_values, dist_th=1):
    new_past_key_values = []
    # bsz head len 128
    k = torch.cat([past_key_values[i][0] for i in range(32)])
    v = torch.cat([past_key_values[i][1] for i in range(32)])
    a = time.time()
    s = cal_key_sim_fast(k).detach().to(torch.float).cpu().numpy()
    b = time.time()
    # bsz head len
    repeat, radio = get_clone_index(s, th=dist_th)
    c = time.time()
    repeat = torch.tensor(repeat).reshape(k.shape[:-1]).cuda()
    selected_k = torch.gather(k, 2, repeat.unsqueeze(-1).expand(-1, -1, -1, 128)).cuda()
    selected_v = v.cuda()
    # unsqueeze bsz
    new_past_key_values = [(selected_k[i].unsqueeze(0), selected_v[i].unsqueeze(0)) for i in range(selected_k.shape[0])]
    d = time.time()
    #print(f'K shape {k.shape} cal sim {b - a} cal cmp {c - b} replace{d - c} all {d - a}')
    return new_past_key_values, np.mean(radio)


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

def modify_cache_by_kor(past_key_values, dist_th=1):
    new_past_key_values = []
    # bsz head len 128
    k = torch.cat([past_key_values[i][0] for i in range(32)])
    v = torch.cat([past_key_values[i][1] for i in range(32)])
    a = time.time()
    s = cal_key_sim_or(k).detach().to(torch.float).cpu().numpy()
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


import multiprocessing
def mpi_modeify_cache(past_key_values):
    pool = multiprocessing.Pool(processes=32)
    k = [past_key_values[i][0] for i in range(32)]
    v = [past_key_values[i][1] for i in range(32)]
    rs = pool.map(cal_key_sim, k)
    s=torch.stack(rs)
    repeat, radio = get_clone_index(s, th=dist_th)
    # c = time.time()
    repeat = torch.tensor(repeat).reshape(k.shape[:-1]).cuda()
    selected_k = torch.gather(k, 2, repeat.unsqueeze(-1).expand(-1, -1, -1, 128)).cuda()
    selected_v = torch.gather(v, 2, repeat.unsqueeze(-1).expand(-1, -1, -1, 128)).cuda()
    # unsqueeze bsz
    new_past_key_values = [(selected_k[i].unsqueeze(0), selected_v[i].unsqueeze(0)) for i in range(selected_k.shape[0])]
    # d = time.time()
    # print(f'cal sim {b - a} cal cmp {c - b} replace{d - c} all {d - a}')
    return new_past_key_values, np.mean(radios)