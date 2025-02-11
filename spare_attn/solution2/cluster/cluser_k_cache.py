from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
import numpy as np

import numpy as np
import os



if __name__ == '__main__':

    cache_save_path = 'Llama-2-7B-32K-Instruct_k_cache'
    cids_save_name='no_norm_8bits_8196_32K_vec3.npy'


    k_bag = []
    cnt = 0
    for fn in os.listdir(f'./{cache_save_path}/'):
        fp = './kv_cache_dir/' + fn
        kv = np.load(fp)
        print(kv.shape)
        k = kv[:, 0]
        k = k.reshape(-1, 128)
        k_bag.append(k)
        # print(kv.shape)
        cnt += 1
        if cnt > 2:
            break


    k_bag_tensor = np.vstack(k_bag)
    # 假设 X 是您的数据矩阵
    X = k_bag[0].reshape(-1, 128)  # 示例大数据集
    # 归一化数据到单位长度
    #X_normalized = normalize(X, norm='l2')
    X_split=X.reshape(-1,2)
    # 使用 MiniBatchKMeans 进行聚类
    minibatch_kmeans = MiniBatchKMeans(n_clusters=8196,max_iter=200, batch_size=10000, random_state=0)
    minibatch_kmeans.fit(X_split)

    labels = minibatch_kmeans.labels_
    centroids = minibatch_kmeans.cluster_centers_

    print("Cluster labels:", labels)
    np.save(cids_save_name, centroids)