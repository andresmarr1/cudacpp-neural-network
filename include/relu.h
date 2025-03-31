//
// Created by andres on 2/4/25.
//

#ifndef RELU_H
#define RELU_H
#include <cuda_runtime.h>
#include <optional>
#include <cstdio>
#include <cstdint>
#include "layer.cuh"
#include "tensor.cuh"

class ReLU : public Layer {
  public:
    ReLU();
    ~ReLU();
    Tensor<float> forward(Tensor<float>& input);
    Tensor<float> backward(Tensor<float>& grad_output);
    void print_weights() {}

    void update(float learning_rate) {}
    void delete_gradients() {}
  private:
    std::optional<Tensor<uint8_t>> bitmask;
};
#endif //RELU_H
