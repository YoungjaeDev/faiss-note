# Faiss Note

얼굴 인식에서 주로 매칭에서 벡터 유사도를 통해서 계산하게 되는데, 이 때 최근에 효율적으로 사용되는 [Faiss](https://github.com/facebookresearch/faiss)을 사용해보고자 한다.
Faiss는 일반적으로 고정된 차원을 가진 벡터들의 집합을 다루게 되며, 오직 float32 형식의 벡터만을 지원한다.
- 배치 처리를 통한 빠른 유사도 검색 제공

## 설치

Faiss comes with precompiled libraries for Anaconda in Python

1. python
conda 채널이 아닌 pytorch 채널을 탐색해서 설치한다

```bash
conda install faiss-cpu -c pytorch
```

or gpu
```bash
conda install faiss-gpu -c pytorch
```

faiss 1.7.4 버전에서 mkl 라이브러리 오류가 발생하는데, mkl을 재설치한다.
```bash
https://github.com/facebookresearch/faiss/issues/2890
conda install mkl=2021
```

2. C++


## 
