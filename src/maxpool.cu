//
// Created by andres on 2/4/25.
//
#include "../include/maxpool.h"

__global__ void maxpool2d_forward(const float* input, float* output, const int batch_size,const int channels, const int input_width, const int input_height, const int output_height, const int output_width, const int kernel_size, const int stride, int* indices) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.x + threadIdx.y;
  int c = blockIdx.z % channels;
  int k = blockIdx.z / channels;
  int in_h = row * stride;
  int in_w = col * stride;
  if (in_h >= input_height || in_w >= input_width) return;
  int output_index = ((k * channels + c) * output_height + row) * output_width + col;
  float max_val = -1e10;
  int max_in = -1;
  for (int kh = 0; kh < kernel_size; kh++) {
    for (int kw = 0; kw < kernel_size; kw++) {
      if ((in_h + kh) < input_height && (in_w + kw) < input_width){
        int in = ((k * channels + c) * input_height + (in_h + kh)) * input_width + (in_w + kw);
        if (input[in] > max_val) {
          max_in = in;
          max_val = input[in];
        }
      }
    }
  }
  if (max_in >= 0 && max_in < (input_height * input_width * batch_size * channels)) {
    output[output_index] = max_val;
    indices[output_index] = max_in;
  }
  __syncthreads();
}

__global__ void maxpool2d_backward(const float* grad_output, float* grad_input, const int* indices, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    int max_in = static_cast<int>(indices[i]);
    grad_input[max_in] = grad_output[i];
  }
}

MaxPool2d::MaxPool2d(int kernel_size, int stride) : Layer(false), kernel_size(kernel_size), stride(stride) {
}

Tensor<float> MaxPool2d::forward(Tensor<float>& input) {
  int output_height = (input.height - kernel_size) / stride + 1;
  int output_width = (input.width - kernel_size) / stride + 1;
  Tensor<float> output(input.batch_size, input.channels, output_height, output_width);
  dim3 bDim(16,16);
  dim3 gDim((output_width+15)/16,(output_height+15)/16, input.batch_size*input.channels);
  if (!max_indices.has_value()) {
    max_indices.emplace(input.batch_size, input.channels, output_height, output_width);
  }
  maxpool2d_forward<<<gDim,bDim>>>(input.d_data, output.d_data, input.batch_size, input.channels, input.width, input.height, output_height, output_width, kernel_size, stride, max_indices->d_data);
  cudaDeviceSynchronize();
  return output;
}

Tensor<float> MaxPool2d::backward(Tensor<float>& grad_output) {
  int input_height = (grad_output.height - 1) * stride + kernel_size;
  int input_width = (grad_output.width - 1) * stride + kernel_size;
  int grad_output_size = grad_output.batch_size * grad_output.channels * grad_output.height * grad_output.width;
  Tensor<float> grad_input(grad_output.batch_size, grad_output.channels, input_height, input_width);
  dim3 bDim(16);
  dim3 gDim((grad_output_size + 15)/16);
  grad_input.fill(0.0);
  maxpool2d_backward<<<gDim, bDim>>>(grad_output.d_data, grad_input.d_data, max_indices->d_data, grad_output_size);
  cudaDeviceSynchronize();
  return grad_input;
}
