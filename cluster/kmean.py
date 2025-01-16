import sys
sys.path.append('/data1/ydq/duo-attention/')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2" 

import numpy as np
import faiss
import torch


k_bag=[]
cnt=0
for fn in os.listdir('/data1/ydq/notebooks/duo_attn/kv_cache_dir/'):
    fp='/data1/ydq/notebooks/duo_attn/kv_cache_dir/'+fn
    kv=np.load(fp)
    k=kv[:,0,0]
    k_bag.append(k)
    #print(kv.shape)
    cnt+=1
    if cnt >10:
        break
k_bag_tensor=np.concatenate(k_bag, axis=2)

# 假设你的数据是一个 NumPy 数组，每行是一个向量
data =  k_bag_tensor.reshape(-1, 128).astype('float32')
data = data / np.linalg.norm(data, axis=1, keepdims=True)

gpu_index = 0  # 假设使用第一个 GPU

# 创建一个使用 GPU 的 K-means 索引
d = data.shape[1]  # 向量维度
k = 8192  # 聚类中心的数量
clus = faiss.Clustering(d, k)

# 设置聚类参数
clus.niter = 20
clus.verbose = True
clus.spherical = False

# 创建 GPU 资源
res = faiss.StandardGpuResources()
index = faiss.GpuIndexFlatL2(res, d)

# 执行训练
clus.train(data, index)

# 检索聚类中心
centroids = faiss.vector_float_to_array(clus.centroids).reshape(k, d)

# 打印结果
print("聚类中心：", centroids)
