from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
import numpy as np

import numpy as np
import os
import sys


if __name__ == '__main__':


    dim = int(sys.argv[1])
    cids = int(sys.argv[2])
    bits=int(16 / dim)
    #cache_save_dir = '/ms/FM/ydq/notebook/duo_attn/quant/kv_cache_dir_1024K_2025/'
    cache_save_dir = '/ms/FM/ydq/notebook/duo_attn/quant/kv_cache_dir_2025/'
    cids_save_name = f'/ms/FM/ydq/notebook/duo_attn/quant/dim{dim}_equal_{bits}bits_{cids}_32K_vec2.npy'

    print(f'use cache dir is {cache_save_dir} \nsave to{cids_save_name}')


    k_bag = []
    cnt = 0
    for fn in os.listdir(cache_save_dir):
        fp = cache_save_dir + fn
        kv = np.load(fp)
        print(kv.shape)
        k = kv[:, 0]
        k = k.reshape(-1, 128)
        k_bag.append(k)
        # print(kv.shape)
        cnt += 1
        if cnt > 1:
            break

    k_bag_tensor = np.vstack(k_bag)
    # 假设 X 是您的数据矩阵
    # X = k_bag[0].reshape(-1, 128)  # 示例大数据集
    X = k_bag_tensor
    # 归一化数据到单位长度
    # X_normalized = normalize(X, norm='l2')
    X_split = X.reshape(-1, dim)
    # 使用 MiniBatchKMeans 进行聚类
    minibatch_kmeans = MiniBatchKMeans(n_clusters=cids, max_iter=200, batch_size=10000, random_state=0)
    minibatch_kmeans.fit(X_split)

    labels = minibatch_kmeans.labels_
    centroids = minibatch_kmeans.cluster_centers_

    print("Cluster labels:", labels)
    np.save(cids_save_name, centroids)