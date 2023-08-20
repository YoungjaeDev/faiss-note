"""
@ File:     4_write_and_read_index.py
@ Author:   Youngjae.you
@ Datetime: 2023-08-20
@ Desc:     Index 파일을 저장하고 불러오는 예제
"""

# 1. Import Library
import numpy as np
import faiss

# 2. prepare dataset
# 데이터 및 쿼리 벡터 준비
d = 64  # 벡터의 차원
nb = 1000  # 데이터베이스 크기

np.random.seed(1234)  # 재현성을 위한 랜덤 시드 설정

# 데이터베이스 벡터 생성
xb = np.random.random((nb, d)).astype('float32')

# faiss.normalize_L2를 사용하여 벡터 정규화
faiss.normalize_L2(xb)

# IndexFlatIP 인덱스 생성 및 데이터베이스 벡터 추가
index = faiss.IndexFlatIP(d)
index.add(xb)

# 인덱스를 파일에 저장
index_filename = "index_file.index"
faiss.write_index(index, index_filename)
print(f"Index saved to {index_filename}")

# 저장된 인덱스 파일에서 인덱스를 다시 읽어옴
index_loaded = faiss.read_index(index_filename)
print(f"Index loaded from {index_filename}")

# 쿼리 벡터 생성 및 정규화
nq = 10  # 쿼리 크기
xq = np.random.random((nq, d)).astype('float32')
faiss.normalize_L2(xq)

# 로드된 인덱스로 검색
k = 4  # 가장 가까운 4개의 이웃을 반환
D, I = index_loaded.search(xq, k)
print(I)
print(D)
