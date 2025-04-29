import faiss
import numpy as np
import torch
import threading
import concurrent
from tqdm import tqdm

def load_index(vectors, dimension):
    index = faiss.IndexFlatL2(dimension)
    res = faiss.StandardGpuResources()
    res.setTempMemory(16 * 1024 * 1024)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
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
    k = key_tensor.cpu().to(torch.float).numpy()
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


def batch_rebuid_no_norm_k(key_tensor, gpu_index, vectors, bits):
    k = key_tensor.cpu().to(torch.float).numpy()
    k_size = k.shape
    # print(k.dtype)
    # print(vectors.shape)
    # k_norms=np.linalg.norm(k, ord=2, axis=-1,keepdims=True)
    # normlize_k=k/k_norms
    low_k = k.reshape(-1, bits)
    batch = 32 * 32 * 32 * 1024
    all_indices = np.zeros(low_k.shape[0], dtype=np.int32)
    for i in range(0, low_k.shape[0], batch):
        distances, indices = gpu_index.search(low_k[i:i + batch], 1)
        all_indices[i:i + batch] = indices.reshape(-1)
    rebuild_kk = vectors[all_indices]
    rebuild_kk = rebuild_kk.reshape(k_size)

    # rebuild_kk=rebuild_kk*k_norms
    return rebuild_kk


class Quanter():
    def __init__(self, p, dims=0):
        # p = '/nfs/hw-data/ms/FM/ydq/notebook/duo_attn/no_norm_4bits_8196.npy'
        vectors = np.load(p)
        if dims == 0:
            self.dims = vectors.shape[-1]
        print(f'use quant {p} dims {self.dims}')
        # old [count,dim] new[group,count,dim]
        if len(vectors.shape)==3:
            vectors=vectors[0]
        self.gpu_index, self.vectors = load_index(vectors, self.dims)

    def quant(self, past_key_values):
        # print('----------do_quant')
        new_past_key_values = []
        # bsz head len 128
        k = torch.cat([past_key_values[i][0] for i in range(32)])
        v = torch.cat([past_key_values[i][1] for i in range(32)])
        # print('k.shape',k.shape)
        # rbk = rebuid_no_norm_k(k, self.gpu_index, self.vectors)
        rbk = batch_rebuid_no_norm_k(k, self.gpu_index, self.vectors, self.dims)
        del k
        torch.cuda.empty_cache()
        selected_k = torch.from_numpy(rbk).to(v.dtype).cuda()
        selected_v = v
        # unsqueeze bsz
        new_past_key_values = [(selected_k[i].unsqueeze(0), selected_v[i].unsqueeze(0)) for i in
                               range(selected_k.shape[0])]
        # print(f'K shape {k.shape} cal sim {b - a} cal cmp {c - b} replace{d - c} all {d - a}')
        # print('----------end_quant')
        return new_past_key_values


class KVQuanter():
    def __init__(self, p, dims=0):
        # p = '/nfs/hw-data/ms/FM/ydq/notebook/duo_attn/no_norm_4bits_8196.npy'
        vectors = np.load(p)
        if dims == 0:
            self.dims = vectors.shape[-1]
        print(f'use quant {p} dims {self.dims}')
        self.k_gpu_index, self.k_vectors = load_index(vectors[0], self.dims)
        self.v_gpu_index, self.v_vectors = load_index(vectors[1], self.dims)

    def quant(self, past_key_values):
        # print('----------do_quant')
        fs_past_key_values = []
        # past_key_values
        # layers [full,sllm] [k,v] head len 128
        for j in range(2):
            # print('layers',len(past_key_values))
            # print('f ,s ', len(past_key_values[0]))
            # print('full',past_key_values[0][0].shape)
            # print('slmm', past_key_values[0][1].shape)
            splits = [past_key_values[i][j][0].shape[0] for i in range(32)]
            dtype = past_key_values[0][j][0].dtype
            k = torch.cat([past_key_values[i][j][0] for i in range(32)])
            v = torch.cat([past_key_values[i][j][1] for i in range(32)])
            # rbk = rebuid_no_norm_k(k, self.gpu_index, self.vectors)
            rbk = batch_rebuid_no_norm_k(k, self.k_gpu_index, self.k_vectors, self.dims)
            del k
            torch.cuda.empty_cache()
            selected_k = torch.from_numpy(rbk).to(dtype).cuda().split(splits, dim=0)

            rbv = batch_rebuid_no_norm_k(v, self.v_gpu_index, self.v_vectors, self.dims)
            del v
            torch.cuda.empty_cache()
            selected_v = torch.from_numpy(rbv).to(dtype).cuda().split(splits, dim=0)
            # unsqueeze bsz
            new_past_key_values = [torch.stack([selected_k[i], selected_v[i]]) for i in range(32)]
            fs_past_key_values.append(new_past_key_values)

        pkv = []
        for i in range(32):
            pkv.append([fs_past_key_values[0][i], fs_past_key_values[1][i]])

        # print(f'K shape {k.shape} cal sim {b - a} cal cmp {c - b} replace{d - c} all {d - a}')
        # print('----------end_quant')
        return pkv


class DSQuanter():
    def __init__(self, p, dims=0):
        # p = '/nfs/hw-data/ms/FM/ydq/notebook/duo_attn/no_norm_4bits_8196.npy'
        if isinstance(p, str):
            vectors = np.load(p)
        else:
            vectors = p
        if dims == 0:
            self.dims = vectors.shape[-1]
        print(f'use quant {p} vectors shape is {vectors.shape} dims {self.dims}')
        self.gpu_index, self.vectors = load_index(vectors, self.dims)

    def quant(self, compressed_kv):
        # bsz len 576
        device = compressed_kv.device
        dtype = compressed_kv.dtype
        rbkv = batch_rebuid_no_norm_k(compressed_kv, self.gpu_index, self.vectors, self.dims)
        del compressed_kv
        torch.cuda.empty_cache()
        rb_compressed_kv = torch.from_numpy(rbkv).to(dtype).to(device)
        return rb_compressed_kv


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

class SDSQuanter():
    def __init__(self, p, dims=0):
        # p = '/nfs/hw-data/ms/FM/ydq/notebook/duo_attn/no_norm_4bits_8196.npy'
        if isinstance(p, str):
            vectors = np.load(p)
        else:
            vectors = p
        if dims == 0:
            self.dims = vectors.shape[-1]
        print(f'use quant shape is {vectors.shape} dims {self.dims}')
        self.gpu_index, self.vectors = load_index(vectors, self.dims)

    def quant(self, compressed_kv,i):
        # bsz len 576
        device = compressed_kv.device
        dtype = compressed_kv.dtype
        rbkv = batch_rebuid_no_norm_k(compressed_kv, self.gpu_index, self.vectors, self.dims)
        del compressed_kv
        torch.cuda.empty_cache()
        rb_compressed_kv = torch.from_numpy(rbkv).to(dtype).to(device)
        return rb_compressed_kv,i

class MultiDSQuanter():
    def __init__(self, p, dims=0):
        self.qs = []
        self.rebuild_kv = []
        if isinstance(p, str):
            vectors = np.load(p)
        else:
            vectors = p
        self.centroids_group_count = vectors.shape[0]
        self.dims = vectors.shape[-1]
        # print("centroids_group_count: ", self.centroids_group_count)
        # p = "/ms/FM/ydq/notebook/duo_attn/quant/cluster/setting2_16_256_8.npy" # 16,256,8
        for i in tqdm(range(self.centroids_group_count)):
            current_vectors = vectors[i, :, :]
            # print(current_vectors)
            test_quan = SDSQuanter(current_vectors, dims)
            self.qs.append(test_quan)

    def quant(self, compressed_kv):
        bsz, len, dim = compressed_kv.shape
        compressed_kv = compressed_kv.reshape(bsz, len, self.centroids_group_count, self.dims)
        results_list = [None] * self.centroids_group_count
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.centroids_group_count) as executor:
            futures = [executor.submit(self.qs[i].quant, compressed_kv[:, :, i, :],i)
                       for i in range(self.centroids_group_count)]

        for future in futures:
            rbk_cuda, i = future.result()
            results_list[i] = rbk_cuda
        del compressed_kv
        torch.cuda.empty_cache()

        compressed_kv = torch.stack(results_list, dim=2)
        return compressed_kv.reshape(bsz, len, dim)
