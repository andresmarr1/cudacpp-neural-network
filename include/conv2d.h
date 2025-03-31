//
// Created by andres on 2/4/25.
//

#ifndef CONV2D_H
#define CONV2D_H

#include "layer.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <random>
#include "tensor.cuh"
#include <optional>

class conv2d : public Layer {
  public:
    conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding);
    Tensor<float> forward( Tensor<float>& input) ;
    Tensor<float> backward( Tensor<float>& grad_output);
    void update(float learning_rate);
    void delete_gradients();
    ~conv2d() override;
    void print_weights();
    private:
      int in_channels, out_channels, kernel_size, stride, padding;
      std::optional<Tensor<float>> input_fw;

      Tensor<float> weights;
      Tensor<float> grad_weights;
};

#endif //CONV2D_H
