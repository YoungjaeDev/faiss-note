/*
@ File name	: 5_gpu_load_test.cpp
@ Date		: 2023.08.28
@ Author	: You YoungJae
@ Version	: 0.1.0
@ Description	: Faiss GPU Load test example
*/

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>

#include <chrono> // for sleep
#include <thread> // for sleep

#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/StandardGpuResources.h>

int main() {
    // Read index
    faiss::Index* index = faiss::read_index("index.faiss");
    assert(index != nullptr);

    // Convert CPU index to GPU index
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexFlatIP* gpu_index = reinterpret_cast<faiss::gpu::GpuIndexFlatIP*>(faiss::gpu::index_cpu_to_gpu(&res, 0, index));
    assert(gpu_index != nullptr);
    
    // print cpu, gpu_index info
    printf("index->ntotal: %ld\n", index->ntotal);
    printf("gpu_index->ntotal: %ld\n", gpu_index->ntotal);

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;
    
    int count = 0;
    // Load test
    while(true) 
    {
        // dim=512, nq=1 random vector
        int d = 512;
        int nq = 10000;
        float* xq = new float[d * nq];
        for (int i = 0; i < nq; i++) {
            for (int j = 0; j < d; j++)
                xq[d * i + j] = distrib(rng);
            xq[d * i] += i / 1000.;
        }

        // topk = 4
        int k = 4;

        // search xq
        long* I = new long[k * nq];
        float* D = new float[k * nq];

        // time check, chrono start time 
        

        gpu_index->search(nq, xq, k, D, I);

        // print results
        if (count % 10 == 0) {  // 예: 10번에 한 번만 결과 출력
            printf("I results=\n");
            for (int i = 0; i < 4; i++) {
                printf("I[%d] = %ld\n", i, I[i]);
            }
            printf("D results=\n");
            for (int i = 0; i < 4; i++) {
                printf("D[%d] = %f\n", i, D[i]);
            }
        }
        count++;
        
        // delete
        delete[] xq;
        delete[] I;
        delete[] D;

        // sleep 33ms
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}