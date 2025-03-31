//
// Created by andres on 3/10/25.
//
#include "../include/softmax.cuh"

#include "loss.cuh"

__global__ void softmax_k(float* z,float *yhat,int batch_size){
  int sample_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int lane_idx = threadIdx.x % 32;

  if (sample_idx >= batch_size) return;

  const unsigned FULL_MASK = 0xFFFFFFFF;

  float thread_val = -INFINITY;
  if (lane_idx < NUM_CLASSES) {
    thread_val = z[sample_idx * NUM_CLASSES + lane_idx];
  }

  float max_val = thread_val;

  float other = __shfl_down_sync(FULL_MASK, sample_idx, 8);
  if (lane_idx < 2 && lane_idx + 8 < NUM_CLASSES) {
    max_val = fmaxf(max_val, other);
  }
  other = __shfl_down_sync(FULL_MASK, sample_idx, 4);
  if (lane_idx < 4) {
    max_val = fmaxf(max_val, other);
  }
  other = __shfl_down_sync(FULL_MASK, sample_idx, 2);
  if (lane_idx < 2) {
    max_val = fmaxf(max_val, other);
  }
  other = __shfl_down_sync(FULL_MASK, sample_idx, 1);
  if (lane_idx < 1) {
    max_val = fmaxf(max_val, other);
  }
  max_val = __shfl_sync(FULL_MASK, max_val, 0);

  float exp_val = 0.0f;
  if (lane_idx < NUM_CLASSES) {
    exp_val = expf(thread_val - max_val);
  }

  float sum = exp_val;

  float other_sum = __shfl_down_sync(FULL_MASK, sum, 8);
  if (lane_idx < 2 && lane_idx + 8 < NUM_CLASSES) {
    sum += other_sum;
  }

  other_sum = __shfl_down_sync(FULL_MASK, sum, 4);
  if (lane_idx < 4) {
    sum += other_sum;
  }

  other_sum = __shfl_down_sync(FULL_MASK, sum, 2);
  if (lane_idx < 2) {
    sum += other_sum;
  }
  other_sum = __shfl_down_sync(FULL_MASK, sum, 1);
  if (lane_idx < 1) {
    sum += other_sum;
  }

  sum = __shfl_sync(FULL_MASK, sum, 0);

  if (lane_idx < NUM_CLASSES) {
    float result = exp_val / fmaxf(sum, 1e-7f);
    result = fmaxf(fminf(result, 1.0f - 1e-7f), 1e-7f);

    yhat[sample_idx * NUM_CLASSES + lane_idx] = result;
  }
}

Softmax::Softmax() : Layer(false) {
}
Softmax::~Softmax() {
}
Tensor<float> Softmax::forward( Tensor<float>& input) {
  Tensor<float> output(input.batch_size, 1, 1, input.width);

  dim3 bDim(128);
  dim3 gDim((output.batch_size * 32 + bDim.x - 1) / bDim.x);
  // size_t smem_size = 2 * bDim.x * sizeof(float);
  softmax_k<<<gDim,bDim>>>(input.d_data,output.d_data, input.batch_size);
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  checkCudaError(error, "backward pass softmax");
  output_fw.reset();
  output_fw.emplace(output.clone());

  return output;
}
Tensor<float> Softmax::backward( Tensor<float>& grad_output){
  Tensor<float> onehot_output = one_hot(grad_output);
  Tensor<float> grad_input = output_fw.value() - onehot_output;

  output_fw.reset();
  return grad_input;
}
