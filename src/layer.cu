//
// Created by andres on 3/2/25.
//
#include "../include/layer.cuh"

__global__ void update_inplace_SGD(float* weights, float* gradients, float learning_rate, int size){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size){
    weights[i] -= learning_rate * gradients[i];
  }
}

Layer::Layer( bool params ) : has_params(params) {
}
