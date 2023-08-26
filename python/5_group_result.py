import numpy as np
import faiss

# 데이터 준비
d = 128  # 벡터의 차원
nb = 1000  # 데이터베이스 크기
nq = 10  # 쿼리 크기
img_per_person = 4  # 각 사람당 4장의 이미지가 있음

np.random.seed(1234)

# 데이터베이스 벡터 생성 (랜덤한 예제 데이터)
xb = np.random.random((nb, d)).astype('float32')

# Faiss 인덱스 생성 및 데이터베이스 벡터 추가
index = faiss.IndexFlatIP(d)
index.add(xb)

# 쿼리 벡터 생성 (랜덤한 예제 데이터)
xq = np.random.random((nq, d)).astype('float32')

# 검색
k = img_per_person * img_per_person  # 한 사람당 4장의 이미지이므로 4명의 사람을 찾기 위해 16개의 결과를 반환
D, I = index.search(xq, k)

# 후처리: 동일한 ID를 가진 이미지들의 유사도를 평균
grouped_results = {}
for i in range(I.shape[0]):
    for j in range(I.shape[1]):
        person_id = I[i][j] // img_per_person
        if person_id not in grouped_results:
            grouped_results[person_id] = []
        grouped_results[person_id].append(D[i][j])

avg_scores = {pid: np.mean(scores) for pid, scores in grouped_results.items()}

print(avg_scores)
