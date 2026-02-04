#import faiss_gpu
import numpy as np
import torch
import threading
import concurrent
import os
from tqdm import tqdm


class TorchQuanter():
    def __init__(self, vectors):
        self.dims = vectors.shape[-1]
        print(f'use quant shape is {vectors.shape} dims {self.dims}')
        self.C = vectors
        self.sumc = (vectors ** 2).sum(dim=1)
        self.ki_dtype=torch.uint8
    def quant(self, k, j):
        bsz=100000
        # bsz len 576
        all_len = k.size(0)
        sumc = self.sumc
        ki = torch.zeros((all_len),dtype=self.ki_dtype,device=k.device)
        for i in range(0, all_len, bsz):
            s = i * bsz
            b = k[s:s + bsz]
            sumb = (b ** 2).sum(dim=1, keepdim=True)
            dot_product = torch.matmul(b, self.C.t())
            dist = sumb + sumc - 2 * dot_product
            ki[s:s + bsz] = torch.argmin(dist, dim=-1)
        return ki, j


class MultiGroupQuanter():
    def __init__(self, p, C_device, C_dtype, dims=0):
        self.qs = []
        self.rebuild_kv = []
        if isinstance(p, str):
            vectors = np.load(p)
        else:
            vectors = p
        self.centroids_group_count = vectors.shape[0]
        self.dims = vectors.shape[-1]
        if type(vectors) == np.ndarray:
            self.vectors = torch.from_numpy(vectors).to(C_device).to(C_dtype)
        else:
            self.vectors = vectors.to(C_device).to(C_dtype)
        # print("centroids_group_count: ", self.centroids_group_count)
        # p = "/ms/FM/ydq/notebook/duo_attn/quant/cluster/setting2_16_256_8.npy" # 16,256,8
        for i in tqdm(range(self.centroids_group_count)):
            current_vectors = self.vectors[i, :, :]
            # print(current_vectors)
            #test_quan = KIQuanter(current_vectors)
            test_quan = TorchQuanter(current_vectors)
            self.qs.append(test_quan)

    def quant(self, compressed_k):
        bsz, head, l, dim = compressed_k.shape
        
        # 直接使用原始张量的视图，避免不必要的内存复制
        # 计算每个分组的输入张量
        results_list = [None] * self.centroids_group_count
        
        # 优化并行处理，使用更高效的线程池
        # 根据系统的CPU核心数动态调整线程数
        cpu_count = os.cpu_count() or 8
        max_workers = min(self.centroids_group_count, cpu_count, 16)  # 限制最大线程数，避免过度线程切换
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 为每个分组创建一个子张量，使用 reshape 确保内存连续
            reshaped_k = compressed_k.reshape(-1, dim)
            futures = [executor.submit(self.qs[i].quant, 
                                      reshaped_k[:, i*self.dims:(i+1)*self.dims], 
                                      i)
                       for i in range(self.centroids_group_count)]

            for future in concurrent.futures.as_completed(futures):
                rbk_cuda, i = future.result()
                results_list[i] = rbk_cuda
        
        # 移除不必要的内存清理
        ki = torch.stack(results_list).transpose(0, 1)
        return ki.view(bsz, head, l, self.centroids_group_count)


class FastTorchQuanter():
    def __init__(self, vectors):
        self.dims = vectors.shape[-1]
        print(f'use quant shape is {vectors.shape} dims {self.dims}')
        self.C = vectors
        self.sumc = (vectors ** 2).sum(dim=1)
        self.ki_dtype=torch.int16
    def quant(self, k, j):
        # 使用向量化操作替代循环，提高计算效率
        all_len = k.size(0)
        sumc = self.sumc
        
        # 直接对整个输入张量进行操作
        sumb = (k ** 2).sum(dim=1, keepdim=True)
        dot_product = torch.matmul(k, self.C.t())
        dist = sumb + sumc - 2 * dot_product
        ki = torch.argmin(dist, dim=-1).to(self.ki_dtype)
        
        return ki, j

class ShareQuanter():
    def __init__(self, p, C_device, C_dtype, dims=0):
        self.qs = []
        self.rebuild_kv = []
        if isinstance(p, str):
            vectors = np.load(p)
        else:
            vectors = p
        print(f"cids shape is {vectors.shape}")
        self.dims=vectors.shape[-1]
        self.group= int(128/self.dims)
        vectors = torch.from_numpy(vectors).to(C_device).to(C_dtype)
        self.vectors =vectors.repeat(self.group,1,1)
        #self.qs = TorchQuanter(vectors)
        self.qs = FastTorchQuanter(vectors)

    def quant(self, compressed_k):
        bsz, head, l, dim = compressed_k.shape
        
        # 直接使用原始张量的视图，避免不必要的内存复制
        results_list = [None] * self.group
        
        # 优化并行处理
        # 根据系统的CPU核心数动态调整线程数
        cpu_count = os.cpu_count() or 8
        max_workers = min(self.group, cpu_count, 16)  # 限制最大线程数
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 为每个分组创建一个子张量，使用 reshape 确保内存连续
            reshaped_k = compressed_k.reshape(-1, dim)
            futures = [executor.submit(self.qs.quant, 
                                      reshaped_k[:, i*self.dims:(i+1)*self.dims], 
                                      i)
                       for i in range(self.group)]

            for future in concurrent.futures.as_completed(futures):
                rbk_cuda, i = future.result()
                results_list[i] = rbk_cuda
        
        # 移除不必要的内存清理
        ki = torch.stack(results_list).transpose(0, 1)
        return ki.view(bsz, head, l, self.group)