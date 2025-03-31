//
// Created by andres on 3/8/25.
//
#include "../include/loss.cuh"

__global__ void cross_entropy_k(float* y, float* yhat, float* L, int batch_size) {
    int sample_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_idx = threadIdx.x % 32;

    if (sample_idx >= batch_size) return;
    const unsigned FULL_MASK = 0xffffffff;

    float loss_t = 0.0f;
    if ( lane_idx < NUM_CLASSES) {
        float true_v = y[sample_idx * NUM_CLASSES + lane_idx];

        float pred_v = yhat[sample_idx * NUM_CLASSES + lane_idx];

        pred_v = fmaxf(fminf(pred_v, 1.0f - 1e-7f),1e-7f);

        loss_t = -true_v * logf(pred_v);

        if (isnan(loss_t) || isinf(loss_t)) {
            loss_t = 0.0f;
            if (true_v > 0.0f) {
                loss_t = 15.0f * true_v;
            }
        }
    }
    float sum = loss_t;

    float other_sum = __shfl_down_sync(FULL_MASK, sum, 8);
    if (lane_idx < 2 && lane_idx + 8 < NUM_CLASSES) sum += other_sum;

    other_sum = __shfl_down_sync(FULL_MASK, sum, 4);
    if (lane_idx < 4) sum += other_sum;

    other_sum = __shfl_down_sync(FULL_MASK, sum, 2);
    if (lane_idx < 2) sum += other_sum;

    other_sum = __shfl_down_sync(FULL_MASK, sum, 1);
    if (lane_idx < 1) sum += other_sum;

    // First thread in warp writes the result
    if (lane_idx == 0) {
        // Final safety check
        if (sum < 0.0f || isnan(sum) || isinf(sum)) {
            sum = 10.0f;  // Safe fallback value
        }
        L[sample_idx] = sum;
    }
}

void cross_entropy_loss(Tensor<float>& y, Tensor<float>& yhat, Tensor<float>& output) {
    Tensor<float> onehot_labels = one_hot(y);
    Tensor<float> loss(y.batch_size, 1, 1, 1);
    dim3 bDim(128);
    dim3 gDim((output.batch_size * 32 + bDim.x - 1) / bDim.x);
    cross_entropy_k<<<gDim, bDim>>>(onehot_labels.d_data, yhat.d_data, loss.d_data, y.batch_size);
    cudaError_t error = cudaGetLastError();
    checkCudaError(error, "cross entropy loss");
    cudaDeviceSynchronize();
    loss.batch_mean();
    output.cpyFromDevice(loss.d_data);
}