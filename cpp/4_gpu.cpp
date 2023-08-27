/*
@ File name	: 4_gpu.cpp
@ Date		: 2023.08.27
@ Author	: You YoungJae
@ Version	: 0.1.0
@ Description	: Faiss GPU example
*/

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>

int main() {
    int d = 64;      // dimension
    int nb = 100000; // database size
    int nq = 10000;  // nb of queries

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float* xb = new float[d * nb];
    float* xq = new float[d * nq];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);
        xq[d * i] += i / 1000.;
    }

    faiss::gpu::StandardGpuResources res;
    
    faiss::gpu::GpuIndexFlatL2 index_flat(&res, d);

    printf("is_trained = %s\n", index_flat.is_trained ? "true" : "false");
    index_flat.add(nb, xb); // add vectors to the index
    printf("ntotal = %ld\n", index_flat.ntotal);

    int k = 4;

    { // search xq
        long* I = new long[k * nq];
        float* D = new float[k * nq];

        index_flat.search(nq, xq, k, D, I);

        // print results
        printf("I (5 first results)=\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        printf("I (5 last results)=\n");
        for (int i = nq - 5; i < nq; i++) {
            for (int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        delete[] I;
        delete[] D;
    }


    return 0;
}