//
// Created by andres on 3/8/25.
//

#ifndef LOSS_CUH
#define LOSS_CUH
#include "tensor.cuh"
#define NUM_CLASSES 10


// L = - sum(over all classes) y log yhat
__global__ void cross_entropy_k(float* y, float* yhat, float* L, int batch);



void cross_entropy_loss(Tensor<float>& y, Tensor<float>& yhat, Tensor<float>& output);

#endif //LOSS_CUH
