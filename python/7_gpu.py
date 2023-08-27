"""
@ File:     7_gpu.py
@ Author:   Youngjae.you
@ Datetime: 2023-08-27
@ Desc:     gpu를 사용하여 faiss를 사용하는 예제
"""

# 1. Import Library
import numpy as np
import faiss

# 2. prepare dataset
d = 64                          # dimension
nb = 1000                       # database size
nq = 10                         # queries size

np.random.seed(1234)            # make reproducible

res = faiss.StandardGpuResources()  # use a single GPU

xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

faiss.normalize_L2(xb)        # normalize input vectors to unit length    
faiss.normalize_L2(xq)        # normalize input vectors to unit length

index = faiss.IndexFlatIP(d)   # build the index

gpu_index = faiss.index_cpu_to_gpu(res, 0, index) # make it a GPU index. 0 is the GPU id

assert gpu_index.is_trained
gpu_index.add(xb)                  # add vectors to the index

print("gpu_index.ntotal: ", gpu_index.ntotal)

# 3. search
k = 4                       # want to see 4 nearest neighbors
D, I = gpu_index.search(xq, k)  # sanity check
print("I[:5]: \n", I[:5])   # neighbors of the 5 first queries
print("D[:5]: \n", D[:5])

# 4. sanity check
D_ref, I_ref = gpu_index.search(xb[:5], k)
print("I[:5]: \n", I_ref[:5])   # neighbors of the 5 first queries
print("D[:5]: \n", D_ref[:5])
