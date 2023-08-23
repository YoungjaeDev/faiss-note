/*
@ File name	: 2_indexFlatIP.cpp
@ Date		: 2023.08.23
@ Author	: You YoungJae
@ Version	: 0.1.0
@ Description	: Faiss indexFlatIP Example
*/

#include <cstdio>
#include <cstdlib>
#include <random>

// include faiss header
#include <faiss/IndexFlat.h>
#include <faiss/utils/distances.h>

int main() {
    int d = 64; // dimension
    int nb = 100000; // database size
    int nq = 10000; // query size

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float* xb = new float[d * nb];
    float* xq = new float[d * nq];

    for (int i=0; i<nb; i++) {
        for (int j=0; j<d; j++) {
            xb[d * i + j] = distrib(rng);
        }
        // xb[d * i] += i / 1000.;
        xb[d * i] += i;
    }

    for (int i=0; i<nq; i++) {
        for (int j=0; j<d; j++) {
            xq[d * i + j] = distrib(rng);
        }
        // xq[d * i] += i / 1000.;
        xq[d * i] += i;
    }

    // 벡터의 길이를 1로 만들려면 fvec_renorm_L2 함수를 사용해야 한다
    // fvec_norm_L2sqr 함수는 그저 벡터의 길이를 확인하기 위한 것이다 (제곱근은 포함하지 않음)

    // Normalize the vectors for xb
    for (int i = 0; i < nb; i++) {
        float norm = faiss::fvec_norm_L2sqr(xb + i * d, d);
        if (norm > 0) {
            faiss::fvec_renorm_L2(d, 1, xb + i * d);
        }
    }

    // Normalize the vectors for xq
    for (int i = 0; i < nq; i++) {
        float norm = faiss::fvec_norm_L2sqr(xq + i * d, d);
        if (norm > 0) {
            faiss::fvec_renorm_L2(d, 1, xq + i * d);
        }
    }

    // Create an index for Inner Product similarity
    faiss::IndexFlatIP index(d); // call constructor
    printf("is_trained = %s\n", index.is_trained ? "true" : "false");
    index.add(nb, xb); // add vectors to the index
    printf("ntotal = %ld\n", index.ntotal);

    int k = 4;
    
    {
        // sanity check
        faiss::idx_t* I = new faiss::idx_t[k * 5];
        float* D = new float[k * 5];

        // search xb
        index.search(5, xb, k, D, I);

        // print results
        printf("I=\n");
        for (int i=0; i<5; i++) {
            for (int j=0; j<k; j++) {
                printf("%5ld ", I[i * k + j]);
            }
            printf("\n");
        }

        printf("D=\n");
        for (int i=0; i<5; i++) {
            for (int j=0; j<k; j++) {
                printf("%7g ", D[i * k + j]);
            }
            printf("\n");
        }
    }

    {
        faiss::idx_t* I = new faiss::idx_t[k * 5];
        float* D = new float[k * 5];

        index.search(5, xq, k, D, I);

        // print results
        printf("I=\n");
        for (int i=0; i<5; i++) {
            for (int j=0; j<k; j++) {
                printf("%5ld ", I[i * k + j]);
            }
            printf("\n");
        }

        printf("D=\n");
        for (int i=0; i<5; i++) {
            for (int j=0; j<k; j++) {
                printf("%7g ", D[i * k + j]);
            }
            printf("\n");
        }
    }

    delete[] xb;
    delete[] xq;
    delete[] D;
    delete[] I;
    
    return 0;
}