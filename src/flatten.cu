//
// Created by andres on 3/2/25.
//
#include "../include/flatten.h"

Flatten::Flatten() : Layer(false) {
}

Flatten::~Flatten() {
}

Tensor<float> Flatten::forward(Tensor<float>& input) {
    Tensor<float> output = input.clone();
    in_channels = input.channels;
    in_height = input.height;
    in_width = input.width;
    output.reshape(output.batch_size, 1, 1, in_channels * in_height * in_width);
    return output;
}

Tensor<float> Flatten::backward(Tensor<float>& grad_output){
    Tensor<float> grad_input = grad_output.clone();
    grad_input.reshape(grad_output.batch_size, in_channels, in_height, in_width);
    return grad_input;
}