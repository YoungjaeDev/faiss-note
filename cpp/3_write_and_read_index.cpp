/*
@ File: 3_write_and_read_index.cpp
@ Author: Youngjae.you
@ Datetime: 2023-08-20
@ Desc: Example of saving and loading an index file using C++
*/

#include <cstdio>
#include <random>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>

int main() {
    const float tolerance = 1e-6;  // Adjust this value based on your requirements

    // 2. prepare dataset
    int d = 64; // dimension of the vector
    int nb = 1000; // database size

    std::mt19937 rng(1234); // set random seed for reproducibility
    std::uniform_real_distribution<> distrib(0.0, 1.0);

    float* xb = new float[d * nb];
    
    // create database vector
    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++) {
            xb[d * i + j] = distrib(rng);
        }
    }

    // Normalize the vector
    for (int i = 0; i < nb; i++) {
        float norm = faiss::fvec_norm_L2sqr(xb + i * d, d);
        if (norm > 0) {
            faiss::fvec_renorm_L2(d, 1, xb + i * d);
        }
        norm = faiss::fvec_norm_L2sqr(xb + i * d, d);
        assert(std::abs(norm - 1.0) <= tolerance);
    }


    // Create IndexFlatIP index and add database vector
    faiss::IndexFlatIP index(d);
    index.add(nb, xb);

    // save the index to a file
    const char* index_filename = "index_file.index";
    faiss::write_index(&index, index_filename);
    printf("Index saved to %s\n", index_filename);

    // read the index back from the saved index file
    faiss::Index* index_loaded = faiss::read_index(index_filename);
    printf("Index loaded from %s\n", index_filename);

    // Generate and normalize query vectors
    int nq = 10; // query size
    float* xq = new float[d * nq];
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++) {
            xq[d * i + j] = distrib(rng);
        }
    }

    // Normalize the query vectors
    for (int i = 0; i < nq; i++) {
        float norm = faiss::fvec_norm_L2sqr(xq + i * d, d);
        if (norm > 0) {
            faiss::fvec_renorm_L2(d, 1, xq + i * d);
        }
    }

    // Search by loaded index
    int k = 4; // return the 4 nearest neighbors
    float* D = new float[k * nq];
    faiss::idx_t* I = new faiss::idx_t[k * nq];

    // search xq
    // parameter: nq, xq, k, D, I
    index_loaded->search(nq, xq, k, D, I);

    for (int i = 0; i < nq; i++) {
        printf("Query %d:\n", i);
        for (int j = 0; j < k; j++) {
            printf("%5ld ", I[i * k + j]);
        }
        printf("\n");
        for (int j = 0; j < k; j++) {
            printf("%7g ", D[i * k + j]);
        }
        printf("\n");
    }

    delete[] xb;
    delete[] xq;
    delete[] D;
    delete[] I;
    delete index_loaded;

    return 0;
}