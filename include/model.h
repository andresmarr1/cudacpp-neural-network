//
// Created by andres on 3/2/25.
//

#ifndef MODEL_H
#define MODEL_H

#include "layer.cuh"
#include "tensor.cuh"
#include <vector>
#include <algorithm>
#include <iostream>
#include <memory>
#include <ranges>

class Model {
  public:
    Model();
    ~Model();
    void add_layer(Layer* layer);
    void forward(Tensor<float>& input, Tensor<float>& output);
    void print_weights();
    void backward(Tensor<float>& grad_output);
    void update(float learning_rate);
    void zero_grad();

  private:
    std::vector<Layer*> layers;


};

#endif //MODEL_H
