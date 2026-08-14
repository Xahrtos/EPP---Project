#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "energy_storms.h"

#define BLOCK_SIZE 256

/*
 * KERNEL 1: Bombardamento
 */
__global__ void bombardment_kernel(float *layer, int layer_size, int pos, float energy) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (k < layer_size) {
        int distance = pos - k;
        if (distance < 0) distance = -distance;
        distance = distance + 1;

        float atenuacion = sqrtf((float)distance);
        float energy_k = energy / layer_size / atenuacion;

        if (energy_k >= THRESHOLD / layer_size || energy_k <= -THRESHOLD / layer_size) {
            layer[k] += energy_k;
        }
    }
}

/*
 * KERNEL 2: Rilassamento (Fase 4.2.2)
 */
__global__ void relaxation_kernel(const float *layer_copy, float *layer, int layer_size) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k >= 1 && k < layer_size - 1) {
        layer[k] = (layer_copy[k - 1] + layer_copy[k] + layer_copy[k + 1]) / 3.0f;
    }
}

/*
 * KERNEL 3: Ricerca del Massimo
 */
__global__ void find_max_kernel(const float *layer, int layer_size, float *d_max_vals, int *d_max_poss) {
    __shared__ float s_max_val[BLOCK_SIZE];
    __shared__ int   s_max_pos[BLOCK_SIZE];

    int tid = threadIdx.x;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    s_max_val[tid] = -1.0f;
    s_max_pos[tid] = -1;

    if (k >= 1 && k < layer_size - 1) {
        if (layer[k] > layer[k - 1] && layer[k] > layer[k + 1]) {
            s_max_val[tid] = layer[k];
            s_max_pos[tid] = k;
        }
    }

    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_max_val[tid + stride] > s_max_val[tid]) {
                s_max_val[tid] = s_max_val[tid + stride];
                s_max_pos[tid] = s_max_pos[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_max_vals[blockIdx.x] = s_max_val[0];
        d_max_poss[blockIdx.x] = s_max_pos[0];
    }
}

/* 
 * FUNZIONE CORE HOST
 */
void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {

    float *d_layer = NULL;
    float *d_layer_copy = NULL;

    cudaMalloc((void**)&d_layer, layer_size * sizeof(float));
    cudaMalloc((void**)&d_layer_copy, layer_size * sizeof(float));

    cudaMemset(d_layer, 0, layer_size * sizeof(float));
    cudaMemset(d_layer_copy, 0, layer_size * sizeof(float));

    int num_blocks = (layer_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float *d_block_max_vals = NULL;
    int   *d_block_max_poss = NULL;

    cudaMalloc((void**)&d_block_max_vals, num_blocks * sizeof(float));
    cudaMalloc((void**)&d_block_max_poss, num_blocks * sizeof(int));

    float *h_block_max_vals = (float*) malloc(num_blocks * sizeof(float));
    int   *h_block_max_poss = (int*)   malloc(num_blocks * sizeof(int));

    for (int i = 0; i < num_storms; i++) {

        for (int j = 0; j < storms[i].size; j++) {
            float energy = (float)storms[i].posval[j * 2 + 1] * 1000.0f;
            int position = storms[i].posval[j * 2];

            bombardment_kernel<<<num_blocks, BLOCK_SIZE>>>(d_layer, layer_size, position, energy);
        }

        cudaMemcpy(d_layer_copy, d_layer, layer_size * sizeof(float), cudaMemcpyDeviceToDevice);

        relaxation_kernel<<<num_blocks, BLOCK_SIZE>>>(d_layer_copy, d_layer, layer_size);

        find_max_kernel<<<num_blocks, BLOCK_SIZE>>>(d_layer, layer_size, d_block_max_vals, d_block_max_poss);

        cudaMemcpy(h_block_max_vals, d_block_max_vals, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_block_max_poss, d_block_max_poss, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);

        float current_max_val = maximum[i];
        int   current_max_pos = positions[i];

        for (int b = 0; b < num_blocks; b++) {
            if (h_block_max_vals[b] > current_max_val) {
                current_max_val = h_block_max_vals[b];
                current_max_pos = h_block_max_poss[b];
            }
        }

        maximum[i]   = current_max_val;
        positions[i] = current_max_pos;
    }

    cudaFree(d_layer);
    cudaFree(d_layer_copy);
    cudaFree(d_block_max_vals);
    cudaFree(d_block_max_poss);

    free(h_block_max_vals);
    free(h_block_max_poss);
}