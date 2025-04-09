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


class SimQuanter():
    def __init__(self, vectors):
        self.dims = vectors.shape[-1]
        print(f'use quant shape is {vectors.shape} dims {self.dims}')
        self.gpu_index, self.vectors = load_index(vectors, self.dims)

    def quant(self, key_tensor, j):
        # bsz len 576
        device = key_tensor.device
        dtype = key_tensor.dtype
        k = key_tensor.cpu().to(torch.float).numpy()
        batch = 32 * 32 * 32 * 1024
        all_indices = np.zeros(k.shape[0], dtype=np.int16)
        for i in range(0, k.shape[0], batch):
            distances, indices = self.gpu_index.search(k[i:i + batch], 1)
            all_indices[i:i + batch] = indices.reshape(-1)

        rebuild_kk = self.vectors[all_indices]
        rebuild_kk = torch.from_numpy(rebuild_kk).to(dtype).to(device)
        return rebuild_kk, j


class MultiSimQuanter():
    def __init__(self, p, C_device,C_dtype, dims=0):
        self.qs = []
        self.rebuild_kv = []
        if isinstance(p, str):
            vectors = np.load(p)
        else:
            vectors = p
        self.centroids_group_count = vectors.shape[0]
        self.dims = vectors.shape[-1]
        self.vectors = torch.from_numpy(vectors).to(C_device).to(C_dtype)
        # print("centroids_group_count: ", self.centroids_group_count)
        # p = "/ms/FM/ydq/notebook/duo_attn/quant/cluster/setting2_16_256_8.npy" # 16,256,8
        for i in tqdm(range(self.centroids_group_count)):
            current_vectors = vectors[i, :, :]
            # print(current_vectors)
            test_quan = SimQuanter(current_vectors)
            self.qs.append(test_quan)

    def quant(self, compressed_k):

        bsz, head, l, dim = compressed_k.shape
        compressed_k = compressed_k.reshape(-1, self.centroids_group_count, self.dims)
        results_list = [None] * self.centroids_group_count
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.centroids_group_count) as executor:
            futures = [executor.submit(self.qs[i].quant, compressed_k[:, i, :], i)
                       for i in range(self.centroids_group_count)]

            for future in concurrent.futures.as_completed(futures):
                rbk_cuda, i = future.result()
                results_list[i] = rbk_cuda
        del compressed_k
        torch.cuda.empty_cache()
        k = torch.stack(results_list).transpose(0, 1)
        return k.reshape(bsz, head, l, dim)
