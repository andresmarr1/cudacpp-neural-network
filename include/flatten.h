//
// Created by andres on 3/2/25.
//

#ifndef FLATTEN_H
#define FLATTEN_H
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <random>
#include <optional>
#include "layer.cuh"
#include "tensor.cuh"
#include "kernel_ops.cuh"

class Flatten : public Layer {
public:
    Flatten();
    Tensor<float> forward( Tensor<float>& input);
    Tensor<float> backward( Tensor<float>& grad_output);
    void print_weights() {}
    void update(float learning_rate) {}
    void delete_gradients() {}
    ~Flatten() override;
private:
    int in_width, in_height, in_channels;
};
#endif //FLATTEN_H
