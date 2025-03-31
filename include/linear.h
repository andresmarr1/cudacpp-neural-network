//
// Created by andres on 2/4/25.
//

#ifndef LINEAR_H
#define LINEAR_H
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <random>
#include <optional>
#include "layer.cuh"
#include "tensor.cuh"
#include "kernel_ops.cuh"

class Linear : public Layer {
  public:
    Linear(int in_features, int out_features);
    Tensor<float> forward( Tensor<float>& input);
    Tensor<float> backward( Tensor<float>& grad_output);
    void print_weights();
    void update(float learning_rate);
    void delete_gradients();
    ~Linear() override;
  private:
    int in_features, out_features;
    std::optional<Tensor<float>> input_fw;
    Tensor<float> weights;
    Tensor<float> grad_weights;
    // Tensor<float> bias;
};
#endif //LINEAR_H
