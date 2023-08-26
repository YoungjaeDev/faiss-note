"""
@ File:     6_IndexIDMap.py
@ Author:   Youngjae.you
@ Datetime: 2023-08-26
@ Desc:     Index 파일에 저장할 때 id도 같이 저장하는 예제
"""

# 1. Import Library
import numpy as np
import faiss

# 2. prepare dataset
vectors = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], dtype=np.float32) # 3 x 2
ids = np.array([10, 20, 30, 10, 20, 30], dtype=np.int64) # 3

index_base = faiss.IndexFlatL2(vectors.shape[1]) # 2

# 3. Use IndexIDMap
index = faiss.IndexIDMap(index_base) # 2
index.add_with_ids(vectors, ids) # 3

D, I = index.search(np.array([[12, 13]], dtype=np.float32), 1) # 1

print("D: ", D)
print("I: ", I)
