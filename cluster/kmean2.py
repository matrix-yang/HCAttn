import sys
sys.path.append('/data1/ydq/duo-attention/')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2" 

import numpy as np
import faiss
import torch
import time

k_bag=[]
cnt=0
npy_p='/data1/ydq/notebooks/duo_attn/kv_cache_dir/'
for fn in os.listdir(npy_p):
    fp=npy_p+fn
    kv=np.load(fp)
    k=kv[:,0,0]
    k_bag.append(k)
    #print(kv.shape)
    cnt+=1
    if cnt >10:
        break
k_bag_tensor=np.concatenate(k_bag, axis=2)


data =  k_bag_tensor.reshape(-1, 128).astype('float32')
x = data / np.linalg.norm(data, axis=1, keepdims=True)
 
ncentroids = 8192
niter = 500
verbose = True
d = x.shape[1]
 
start_time = time.time()
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
kmeans.train(x)
 
train_time = time.time()
print(train_time - start_time)
 
cluster_cents = kmeans.centroids
cluster_wucha = kmeans.obj
 
D, I = kmeans.index.search(x, 1)
print(np.unique(np.array(I))) # 共有1000张数据，形状为[1000,2048]
 
search_time = time.time()
print(search_time - train_time)
 
 
# # 也可以创建一个检索器，然后搜索出离这些中心点最近的15个向量
# index = faiss.IndexFlatL2 (d)
# index.add (x)
# D, I = index.search (kmeans.centroids, 15)
