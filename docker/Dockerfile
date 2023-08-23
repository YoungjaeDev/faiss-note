FROM ubuntu:20.04

# 환경 변수 설정을 통해 tzdata 설치 중 대화형 질의를 방지한다
ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    git \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    swig \
    libssl-dev \
    wget 

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

RUN git clone https://github.com/facebookresearch/faiss.git /faiss

# 최신 버전의 CMake를 다운로드하고 설치합니다.
WORKDIR /tmp
RUN wget https://github.com/Kitware/CMake/releases/download/v3.23.1/cmake-3.23.1.tar.gz && \
    tar -xzvf cmake-3.23.1.tar.gz && \
    cd cmake-3.23.1 && \
    ./bootstrap && \
    make -j$(nproc) && \
    make install

RUN pip3 install numpy

WORKDIR /faiss
RUN cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON -DBUILD_SHARED_LIBS=ON -DPython_EXECUTABLE=/usr/bin/python3 -B build .

RUN make -C build -j faiss
RUN make -C build install

# Building the python bindings
RUN make -C build -j swigfaiss
WORKDIR ./build/faiss/python 
RUN python3 setup.py install
