//
// Created by andres on 2/4/25.
//

#include "../include/relu.h"

__global__ void relu_forward(float* input, float* output, int size, uint8_t* bitmask) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
      if (input[i] > 0.0f){
        output[i] = input[i];
        bitmask[i] = 1;
      }
      else {
        output[i] = 0.0f;
        bitmask[i] = 0;
      }
  }
}

__global__ void relu_backward(float* grad_output, float* grad_input, int size, uint8_t* bitmask) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    grad_input[i] = (bitmask[i] == 1) ? grad_output[i] : 0.0f;
  }
}

ReLU::ReLU() : Layer(false) {
  }
ReLU::~ReLU() {
  if (bitmask.has_value()) {
    bitmask.reset();
  }
}
Tensor<float> ReLU::forward(Tensor<float>& input) {
  Tensor<float> output(input.batch_size, input.channels, input.height, input.width);
  bitmask.reset();
  bitmask.emplace(input.batch_size, input.channels, input.height, input.width);

  int size = input.batch_size * input.channels * input.height * input.width;
  int bxDim = 16;
  dim3 bDim(bxDim);
  dim3 gDim((size + bxDim - 1)/bxDim);
  relu_forward<<<gDim, bDim>>>(input.d_data, output.d_data, size, bitmask->d_data);
  cudaDeviceSynchronize();
  return output;
}

Tensor<float> ReLU::backward(Tensor<float>& grad_output) {
  Tensor<float> grad_input(grad_output.batch_size, grad_output.channels, grad_output.height, grad_output.width);
  if (!bitmask.has_value()) {
    throw std::runtime_error("ReLU::backward: bitmask not set");
  }

  int size = grad_output.batch_size * grad_output.channels * grad_output.height * grad_output.width;
  int bxDim = 16;
  dim3 bDim(bxDim);
  dim3 gDim((size + bxDim - 1)/bxDim);
  relu_backward<<<gDim, bDim>>>( grad_output.d_data, grad_input.d_data, size, bitmask->d_data);
  cudaDeviceSynchronize();


  return grad_input;
}
