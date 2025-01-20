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
    k = key_tensor.to(torch.float).cpu().numpy()
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
    k=key_tensor.cpu().to(torch.float).numpy()
    k_size = k.shape
    # k_norms=np.linalg.norm(k, ord=2, axis=-1,keepdims=True)
    # normlize_k=k/k_norms
    low_k = k.reshape(-1, 4)
    print(low_k.shape)
    distances, indices = gpu_index.search(low_k, 1)
    print(indices.shape)
    rebuild_kk = vectors[indices.reshape(-1)]
    print(rebuild_kk.shape)
    rebuild_kk = rebuild_kk.reshape(k_size)

    # rebuild_kk=rebuild_kk*k_norms
    return rebuild_kk

def batch_rebuid_no_norm_k(key_tensor, gpu_index, vectors):
    k=key_tensor.cpu().to(torch.float).numpy()
    k_size = k.shape
    # k_norms=np.linalg.norm(k, ord=2, axis=-1,keepdims=True)
    # normlize_k=k/k_norms
    low_k = k.reshape(-1, 4)
    batch=32*32*32*1024
    all_indices=np.zeros(low_k.shape[0],dtype=np.int32)
    for i in range(0,low_k.shape[0],batch):
        distances, indices = gpu_index.search(low_k[i:i+batch], 1)
        all_indices[i:i+batch]=indices.reshape(-1)
    rebuild_kk = vectors[all_indices]
    rebuild_kk = rebuild_kk.reshape(k_size)

    # rebuild_kk=rebuild_kk*k_norms
    return rebuild_kk


class Quanter():
    def __init__(self,p):
        #p = '/nfs/hw-data/ms/FM/ydq/notebook/duo_attn/no_norm_4bits_8196.npy'
        self.gpu_index, self.vectors = load_index(p)

    def quant(self,past_key_values):
        #print('----------do_quant')
        new_past_key_values = []
        # bsz head len 128
        k = torch.cat([past_key_values[i][0] for i in range(32)])
        v = torch.cat([past_key_values[i][1] for i in range(32)])
        #print('k.shape',k.shape)
        #rbk = rebuid_no_norm_k(k, self.gpu_index, self.vectors)
        rbk = batch_rebuid_no_norm_k(k, self.gpu_index, self.vectors)
        del k
        torch.cuda.empty_cache()
        selected_k = torch.from_numpy(rbk).to(v.dtype).cuda()
        selected_v = v
        # unsqueeze bsz
        new_past_key_values = [(selected_k[i].unsqueeze(0), selected_v[i].unsqueeze(0)) for i in
                               range(selected_k.shape[0])]
        # print(f'K shape {k.shape} cal sim {b - a} cal cmp {c - b} replace{d - c} all {d - a}')
        #print('----------end_quant')
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