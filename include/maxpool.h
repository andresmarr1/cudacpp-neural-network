//
// Created by andres on 2/4/25.
//

#ifndef MAXPOOL_H
#define MAXPOOL_H
#include "layer.cuh"
#include "tensor.cuh"
#include <cuda_runtime.h>
#include <optional>
#include <cstdio>

class MaxPool2d : public Layer {
  public:
    MaxPool2d(int kernel_size, int stride);
    Tensor<float> forward(Tensor<float>& input);
    Tensor<float> backward(Tensor<float>& grad_output);
    void print_weights() override {}
    void update(float learning_rate) override {}
    void delete_gradients() override {}
    private:
      int kernel_size, stride;
      std::optional<Tensor<int>> max_indices;
};
#endif //MAXPOOL_H
