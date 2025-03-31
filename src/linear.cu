//
// Created by andres on 2/4/25.
//
#include "../include/linear.h"

Linear::Linear(int in_features, int out_features) : Layer(true), in_features(in_features), out_features(out_features), weights(1, 1, out_features, in_features), grad_weights(1, 1, out_features, in_features) {
  int seed = 0;
  float std = sqrtf(2.0f / static_cast<float>(in_features));
  int size = in_features * out_features;
  dim3 bDim(16);
  dim3 gDim((in_features*out_features + bDim.x - 1)/bDim.x);
  dim3 gDimB((out_features+bDim.x-1)/bDim.x);
  initialize_weights<<<gDim,bDim>>>(weights.d_data, size, seed, std);
  // initialize_bias<<<gDimB,bDim>>>(bias.d_data, in_features, out_features, seed, std);
}

Linear::~Linear() {
  if (input_fw.has_value()) {
    input_fw.reset();
  }
}

void Linear::print_weights(){
  weights.print_tensor();
}

// void Linear::print_bias(){
//   bias.print_tensor();
// }

Tensor<float> Linear::forward(Tensor<float>& input) {
  //  Y = XW.T + B
  Tensor<float> output = matmul(input, weights.T());
  cudaError_t error = cudaGetLastError();
  checkCudaError(error, "forward pass matmul");
  input_fw.reset();
  input_fw.emplace(input.clone());

  return output;
}

Tensor<float> Linear::backward(Tensor<float>& grad_output){

  Tensor<float> grad_input = matmul(grad_output,weights);
  cudaError_t error = cudaGetLastError();
  checkCudaError(error, "backward pass input matmul");
  Tensor<float> grad_output_T = grad_output.T();
  error = cudaGetLastError();
  checkCudaError(error, "backward pass output transpose");
  // std::cout << "grad_output^T b c h w : " <<grad_output_T.batch_size << " " << grad_output_T.channels << " " << grad_output_T.height << " " << grad_output_T.width << std::endl;
  // std::cout << "input copy  b c h w : " << input_fw->batch_size << " " << input_fw->channels << " " << input_fw->height << " " << input_fw->width << std::endl;
  Tensor<float> grad_weights_ = matmul(grad_output_T, input_fw.value());
  error = cudaGetLastError();
  checkCudaError(error, "backward pass weights matmul");
  // std::cout << "grad_weights_b b c h w : " << grad_weights_.batch_size << " " << grad_weights_.channels << " " << grad_weights_.height << " " << grad_weights_.width << std::endl;
  grad_weights_.batch_mean();
  error = cudaGetLastError();
  checkCudaError(error, "backward pass weight gradient batch mean");
  // std::cout << "grad_weights_b(post mean) b c h w : " << grad_weights_.batch_size << " " << grad_weights_.channels << " " << grad_weights_.height << " " << grad_weights_.width << std::endl;
  grad_weights.cpyFromDevice(grad_weights_.d_data);
  // input_fw->print_tensor();
  // grad_weights_.print_tensor();
  // std::cout << "weights b c h w : " << weights.batch_size << " " << weights.channels << " " << weights.height << " " << weights.width << std::endl;
  input_fw.reset();
  return grad_input;
}


void Linear::update(float learning_rate) {
  if (this->has_params) {
    dim3 bDim(16);
    dim3 gDim((weights.batch_size * weights.channels * weights.height * weights.width + bDim.x - 1)/bDim.x);
    update_inplace_SGD<<<gDim, bDim>>>(weights.d_data, grad_weights.d_data, learning_rate, weights.batch_size * weights.channels * weights.height * weights.width);
    cudaDeviceSynchronize();
  }
  cudaError_t error = cudaGetLastError();
  checkCudaError(error, "weight update SGD");
}

void Linear::delete_gradients() {
  if (this->has_params) {
    cudaMemset(grad_weights.d_data, 0, grad_weights.batch_size * grad_weights.channels * grad_weights.height * grad_weights.width * sizeof(float));
    cudaError_t error = cudaGetLastError();
    checkCudaError(error, "zero gradient");
  }
}