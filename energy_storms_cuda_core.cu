#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "energy_storms.h"

#define BLOCK_SIZE 256

// Macro per il controllo degli errori CUDA
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/* ====================================================
 * KERNEL 1: BOMBARDAMENTO PARTICELLE
 * ==================================================== */
__global__ void bombardment_kernel(float *layer, int layer_size, int position, float energy) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= layer_size) return;

    int distance = position - k;
    if (distance < 0) distance = -distance;
    distance = distance + 1;

    float atenuacion = sqrtf((float)distance);
    float energy_k = energy / layer_size / atenuacion;

    if (energy_k >= THRESHOLD / layer_size || energy_k <= -THRESHOLD / layer_size) {
        layer[k] += energy_k;
    }
}

/* ====================================================
 * KERNEL 2: RILASSAMENTO (STENCIL 1D)
 * ==================================================== */
__global__ void relaxation_kernel(const float *layer_copy, float *layer, int layer_size) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k >= 1 && k < layer_size - 1) {
        layer[k] = (layer_copy[k - 1] + layer_copy[k] + layer_copy[k + 1]) / 3.0f;
    }
}

/* ====================================================
 * KERNEL 3: RICERCA MASSIMO LOCALE E RIDUZIONE PER BLOCCO
 * ==================================================== */
__global__ void find_max_kernel(const float *layer, int layer_size, float *d_block_max_val, int *d_block_max_pos) {
    __shared__ float s_max_val[BLOCK_SIZE];
    __shared__ int   s_max_pos[BLOCK_SIZE];

    int tid = threadIdx.x;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    float local_val = -1.0f;
    int local_pos = -1;

    if (k >= 1 && k < layer_size - 1) {
        if (layer[k] > layer[k - 1] && layer[k] > layer[k + 1]) {
            local_val = layer[k];
            local_pos = k;
        }
    }

    s_max_val[tid] = local_val;
    s_max_pos[tid] = local_pos;
    __syncthreads();

    // Riduzione ad albero binario in Shared Memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_max_val[tid + s] > s_max_val[tid]) {
                s_max_val[tid] = s_max_val[tid + s];
                s_max_pos[tid] = s_max_pos[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_block_max_val[blockIdx.x] = s_max_val[0];
        d_block_max_pos[blockIdx.x] = s_max_pos[0];
    }
}

/* ====================================================
 * FUNZIONE CORE (HOST)
 * ==================================================== */
void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    float *d_layer, *d_layer_copy;
    float *d_block_max_val;
    int   *d_block_max_pos;

    size_t layer_bytes = layer_size * sizeof(float);
    int num_blocks = (layer_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocazioni GPU con controllo errore
    CHECK_CUDA(cudaMalloc(&d_layer, layer_bytes));
    CHECK_CUDA(cudaMalloc(&d_layer_copy, layer_bytes));
    CHECK_CUDA(cudaMalloc(&d_block_max_val, num_blocks * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_block_max_pos, num_blocks * sizeof(int)));

    float *h_block_max_val = (float*)malloc(num_blocks * sizeof(float));
    int   *h_block_max_pos = (int*)malloc(num_blocks * sizeof(int));

    CHECK_CUDA(cudaMemset(d_layer, 0, layer_bytes));

    for (int i = 0; i < num_storms; i++) {

        maximum[i] = -1.0f;
        positions[i] = -1;

        // 4.1. Bombardamento
        for (int j = 0; j < storms[i].size; j++) {
            float energy = (float)storms[i].posval[j * 2 + 1] * 1000.0f;
            int position = storms[i].posval[j * 2];

            bombardment_kernel<<<num_blocks, BLOCK_SIZE>>>(d_layer, layer_size, position, energy);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        // 4.2. Rilassamento (Stencil)
        CHECK_CUDA(cudaMemcpy(d_layer_copy, d_layer, layer_bytes, cudaMemcpyDeviceToDevice));
        relaxation_kernel<<<num_blocks, BLOCK_SIZE>>>(d_layer_copy, d_layer, layer_size);
        CHECK_CUDA(cudaDeviceSynchronize());

        // 4.3. Ricerca del Massimo
        find_max_kernel<<<num_blocks, BLOCK_SIZE>>>(d_layer, layer_size, d_block_max_val, d_block_max_pos);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Trasferimento dei risultati parziali su Host
        CHECK_CUDA(cudaMemcpy(h_block_max_val, d_block_max_val, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_block_max_pos, d_block_max_pos, num_blocks * sizeof(int), cudaMemcpyDeviceToHost));

        // Riduzione finale su Host
        for (int b = 0; b < num_blocks; b++) {
            if (h_block_max_val[b] > maximum[i]) {
                maximum[i] = h_block_max_val[b];
                positions[i] = h_block_max_pos[b];
            }
        }
    }

    CHECK_CUDA(cudaFree(d_layer));
    CHECK_CUDA(cudaFree(d_layer_copy));
    CHECK_CUDA(cudaFree(d_block_max_val));
    CHECK_CUDA(cudaFree(d_block_max_pos));
    free(h_block_max_val);
    free(h_block_max_pos);
}