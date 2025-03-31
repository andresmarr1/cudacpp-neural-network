//
// Created by andres on 2/4/25.
//

#ifndef NNMODULE_H
#define NNMODULE_H
#include "tensor.cuh"

__global__ void update_inplace_SGD(float* weights, float* gradients, float learning_rate, int size);
class Layer {
    public:
        Layer( bool params );
        virtual ~Layer() = default;
        virtual Tensor<float> forward( Tensor<float>& input)= 0;
        virtual Tensor<float> backward( Tensor<float>& grad_output)=0;
        virtual void print_weights() = 0;
        virtual void update(float learning_rate) = 0;
        virtual void delete_gradients() = 0;
    protected:
        bool has_params;
};

#endif //NNMODULE_H
