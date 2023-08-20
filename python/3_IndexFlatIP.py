"""
@ File:     3_IndexFlatIP.py
@ Author:   Youngjae.you
@ Datetime: 2023-08-20
@ Desc:     L2 거리가 아닌 Dot Product를 사용하여 검색하는 IndexFlatIP를 사용한 예제
"""

# 1. Import Library
import numpy as np
import faiss

# 2. prepare dataset
d = 64                          # dimension
nb = 1000                       # database size
nq = 10                         # queries size

np.random.seed(1234)            # make reproducible

xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

faiss.normalize_L2(xb)        # normalize input vectors to unit length    
faiss.normalize_L2(xq)        # normalize input vectors to unit length

index = faiss.IndexFlatIP(d)   # build the index
assert index.is_trained
index.add(xb)                  # add vectors to the index

print("index.ntotal: ", index.ntotal)

# 3. search
k = 4                       # want to see 4 nearest neighbors
D, I = index.search(xq, k)  # sanity check
print("I[:5]: \n", I[:5])   # neighbors of the 5 first queries
print("D[:5]: \n", D[:5])


# 4. sanity check
D_ref, I_ref = index.search(xb[:5], k)
print("I[:5]: \n", I_ref[:5])   # neighbors of the 5 first queries
print("D[:5]: \n", D_ref[:5])
