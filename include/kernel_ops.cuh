//
// Created by andres on 2/8/25.
//

#ifndef KERNEL_OPS_CUH
#define KERNEL_OPS_CUH
#define TILE_SIZE 16
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <random>

__global__ void initialize_weights(float *weights, int size, int seed, float std);
__global__ void initialize_bias(float *weights, int size, int seed, float std);
__global__ void update_weights_k(float *weights, const float *grad_weights, float learning_rate, int size);
__global__ void get_onehot(float* A, float* output, int batch);
void checkCudaError(cudaError_t error, const char* operation);
template <typename T>
__global__ void ops_sb_k(const T *A, const T *B, T *C, T a_sca, T b_sca, T cnst, int size);

template <typename T>
__global__ void fill_k(T *A, T value, int size);

template <typename T>
__global__ void ops_m_k(const T *A, const T *B, T *C, int size);
template <typename T>
__global__ void ops_d_k(const T *A, const T *B, T *C, int size);

template <typename T>
__global__ void matmul_k(const T *A, const T *B, T *C, const int M, const int N, const int K, const bool broadcast);

template <typename T>
__global__ void conv2d_k(T *input, T *filters, T *output, int batch_size, int channels, int height, int width,
                          int kernel_size, int stride, int padding,
                          int output_height, int output_width, int out_channels);
template <typename T>
__global__ void conv2dt_k(T *input, T *filters, T *output, int batch_size, int channels, int height, int width,
                          int kernel_size, int stride, int padding,
                          int output_height, int output_width, int out_channels);

template <typename T>
__global__ void transpose_k(const T *A, T* B,const int batch,const int channels, const int height, const int width);

template <typename T>
__global__ void transpose_bc_k(const T *A, T* B,const int batch,const int channels, const int height, const int width);
template <typename T>
__global__ void reverse_k(const T *A, T* B,const int batch,const int channels, const int height, const int width);

template <typename T>
__global__ void batched_mean_k(const T *A, T* B, const int batch, const int channels, const int height, const int width);

template <typename T>
__global__ void pad_tensor_k(const T *A, T* B, const int batch, const int channels, const int height, const int width, const int padding);

#endif //KERNEL_OPS_CUH
