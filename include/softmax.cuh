//
// Created by andres on 3/8/25.
//

#ifndef SOFTMAX_CUH
#define SOFTMAX_CUH

#define NUM_CLASSES 10
#include "tensor.cuh"
#include "layer.cuh"
#include <optional>

__global__ void softmax_k(float* yhat, int batch);

class Softmax : public Layer {
public:
    Softmax();
    Tensor<float> forward( Tensor<float>& input);
    Tensor<float> backward( Tensor<float>& grad_output) override;
    void print_weights() override {}
    void update(float learning_rate) override {}
    void delete_gradients() override {}
    ~Softmax() override;
private:
    std::optional<Tensor<float>> output_fw;
};

#endif //SOFTMAX_CUH
