//
// Created by andres on 2/28/25.
//
#include "../include/conv2d.h"


conv2d::conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding) : Layer(true),in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), stride(stride), padding(padding), weights(out_channels, in_channels, kernel_size, kernel_size), grad_weights(out_channels, in_channels, kernel_size, kernel_size) {
    int seed = 0;
    int size = out_channels * in_channels * kernel_size * kernel_size;
    float std = 2.0f / sqrtf(static_cast<float>(in_channels * kernel_size * kernel_size));
    dim3 bDim(16);
    dim3 gDim((size + bDim.x - 1)/bDim.x);
    initialize_weights<<<gDim,bDim>>>(weights.d_data, size, seed, std);
    cudaDeviceSynchronize();
}


conv2d::~conv2d() = default;

Tensor<float> conv2d::forward(Tensor<float>& input) {
    Tensor<float> output = convolution2d(input, weights, this->stride, this->padding);

    if (!input_fw.has_value()) {
        input_fw.emplace(input.clone());
    }
    return output;
}

Tensor<float> conv2d::backward( Tensor<float>& grad_output) {
    int padding_backward = ((grad_output.height - 1) * stride - input_fw.value().height + kernel_size) / 2;
    if (padding_backward > 0) {

        grad_output.pad(padding_backward);
    }
    int padding_extra;

    Tensor<float> grad_input = convolution2d_t(grad_output, this->weights.T_bc().reverse(), this->stride, this->padding);
    Tensor<float> int_temp = convolution2d_t(input_fw.value(),grad_output.T_bc(), stride, padding);
    if (padding_backward < 0) {
        padding_extra = input_fw.value().height - ((grad_output.height - 1) * stride - 2 * padding_backward + kernel_size);
        if (padding_extra > 0) {
            grad_input.pad(padding_extra);
        }
    }
    int_temp.batch_mean();
    this->grad_weights = int_temp * static_cast<float>(grad_output.batch_size);
    input_fw.reset();
    return grad_input;
}

void conv2d::print_weights() {
    weights.print_tensor();
    // Tensor<float> weights_T = this->weights.T_bc();
    // weights_T.print_tensor();
}

void conv2d::update(float learning_rate) {
    if (this->has_params) {
        dim3 bDim(16);
        dim3 gDim((weights.batch_size * weights.channels * weights.height * weights.width + bDim.x - 1)/bDim.x);
        update_inplace_SGD<<<gDim, bDim>>>(weights.d_data, grad_weights.d_data, learning_rate, weights.batch_size * weights.channels * weights.height * weights.width);
    }
}


void conv2d::delete_gradients() {
    if (this->has_params) {
        cudaMemset(grad_weights.d_data, 0.0f, grad_weights.batch_size * grad_weights.channels * grad_weights.height * sizeof(float));
    }
}