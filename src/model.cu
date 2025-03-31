//
// Created by andres on 3/19/25.
//
#include "../include/model.h"

Model::Model() {
}
Model::~Model() {
    for (auto layer : layers) {
        delete layer;
    }
    layers.clear();
}
void Model::add_layer(Layer* layer) {
    layers.push_back(layer);
}

void Model::forward(Tensor<float>& input, Tensor<float>& output) {
    Tensor<float> current = input.clone();
    for (auto layer: layers) {
        Tensor<float> next = layer->forward(current);
        // next.print_tensor();
        current = std::move(next);
    }
    output.cpyFromDevice(current.d_data);

}

void Model::backward(Tensor<float>& grad_output) {
    Tensor<float> grad_current = grad_output.clone();

    for (auto &layer : this->layers | std::views::reverse) {
        Tensor<float> grad_next = layer->backward(grad_current);

        grad_current = std::move(grad_next);
        // grad_current.print_tensor();
    }

}
void Model::update(float learning_rate) {
    for (auto &layer : this->layers) {
        layer->update(learning_rate);
    }
}
void Model::zero_grad() {
    for (auto &layer : this->layers) {
        layer->delete_gradients();
    }
}
void Model::print_weights() {
    for (auto &layer : this->layers) {
        layer->print_weights();
    }
}