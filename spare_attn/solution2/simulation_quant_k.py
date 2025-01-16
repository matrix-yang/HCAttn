import faiss
import numpy as np
import torch

def load_index(path):
    # 参数设置
    dimension = 4
    num_vectors = 1000
    k = 5

    index = faiss.IndexFlatL2(dimension)
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

    vectors = np.load(path)
    gpu_index.add(vectors)
    return gpu_index, vectors


def rebuid_norm_k(key_tensor, gpu_index, vectors):
    k = key_tensor.detach().to(torch.float).cpu().numpy()
    k_size = k.shape
    k_norms = np.linalg.norm(k, ord=2, axis=-1, keepdims=True)
    normlize_k = k / k_norms
    low_k = normlize_k.reshape(-1, 4)
    distances, indices = gpu_index.search(low_k, 1)
    rebuild_kk = vectors[indices.reshape(-1)]
    rebuild_kk = rebuild_kk.reshape(k_size)
    rebuild_kk = rebuild_kk * k_norms
    return rebuild_kk


def rebuid_no_norm_k(key_tensor, gpu_index, vectors):
    k=key_tensor.detach().to(torch.float).cpu().numpy()
    k_size = k.shape
    # k_norms=np.linalg.norm(k, ord=2, axis=-1,keepdims=True)
    # normlize_k=k/k_norms
    low_k = k.reshape(-1, 4)
    distances, indices = gpu_index.search(low_k, 1)
    rebuild_kk = vectors[indices.reshape(-1)]
    rebuild_kk = rebuild_kk.reshape(k_size)
    # rebuild_kk=rebuild_kk*k_norms
    return rebuild_kk

class Quanter():
    def __init__(self):
        p = '/nfs/hw-data/ms/FM/ydq/notebook/duo_attn/no_norm_4bits_8196.npy'
        self.gpu_index, self.vectors = load_index(p)

    def quant(self,past_key_values):
        new_past_key_values = []
        # bsz head len 128
        k = torch.cat([past_key_values[i][0] for i in range(32)])
        v = torch.cat([past_key_values[i][1] for i in range(32)])
        rbk = rebuid_no_norm_k(k, self.gpu_index, self.vectors)
        selected_k = torch.from_numpy(rbk).to(v.dtype).cuda()
        selected_v = v
        # unsqueeze bsz
        new_past_key_values = [(selected_k[i].unsqueeze(0), selected_v[i].unsqueeze(0)) for i in
                               range(selected_k.shape[0])]
        # print(f'K shape {k.shape} cal sim {b - a} cal cmp {c - b} replace{d - c} all {d - a}')
        return new_past_key_values

class NormQuanter():
    def __init__(self):
        p = '/nfs/hw-data/ms/FM/ydq/notebook/duo_attn/norm_4bits_8196.npy'
        self.gpu_index, self.vectors = load_index(p)

    def quant(self):
        new_past_key_values = []
        # bsz head len 128
        k = torch.cat([past_key_values[i][0] for i in range(32)])
        v = torch.cat([past_key_values[i][1] for i in range(32)])
        rbk = rebuid_norm_k(k, self.gpu_index, self.vectors)
        selected_k = torch.from_numpy(rbk).to(v.dtype).cuda()
        selected_v = v
        # unsqueeze bsz
        new_past_key_values = [(selected_k[i].unsqueeze(0), selected_v[i].unsqueeze(0)) for i in
                               range(selected_k.shape[0])]
        # print(f'K shape {k.shape} cal sim {b - a} cal cmp {c - b} replace{d - c} all {d - a}')
        return new_past_key_values