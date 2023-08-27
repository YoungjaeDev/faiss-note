"""
@ File:     6_IndexIDMap.py
@ Author:   Youngjae.you
@ Datetime: 2023-08-26
@ Desc:     Index 파일에 저장할 때 id도 같이 저장하는 예제
"""

import numpy as np
import faiss

# 1. Import Library and Initialize GPU resources
res = faiss.StandardGpuResources()  # Initialize GPU resources

# 2. prepare dataset
vectors = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], dtype=np.float32)
ids = np.array([10, 20, 30, 10, 20, 30], dtype=np.int64)

index_base = faiss.IndexFlatL2(vectors.shape[1])

# 3. Move the index to GPU
gpu_index = faiss.index_cpu_to_gpu(res, 0, index_base) # make it a GPU index. 0 is the GPU id

# 4. Use IndexIDMap and Add to GPU index
index = faiss.IndexIDMap(gpu_index)
index.add_with_ids(vectors, ids)

D, I = index.search(np.array([[12, 13]], dtype=np.float32), 1)

print("D: ", D)
print("I: ", I)

