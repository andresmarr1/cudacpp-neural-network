#include <cuda_runtime.h>
#include <iostream>

#include "data_loader.h"
#include "model.h"
#include "include/tensor.cuh"
#include "include/layer.cuh"
#include "include/softmax.cuh"
#include "include/loss.cuh"
#include "include/layer.cuh"
#include "include/relu.h"
#include "include/maxpool.h"
#include "include/linear.h"
#include "include/conv2d.h"
#include "include/flatten.h"
#include "include/dataset.h"

int main() {
    MNISTDataset dataset("/home/andres/Documents/cuda_t/train-images-idx3-ubyte","/home/andres/Documents/cuda_t/train-labels-idx1-ubyte");
    int batch_size = 8;
    int epochs = 10;
    int interval = 10;
    DataLoader loader(dataset,batch_size, true, 0);
    Model model;
    float learning_rate = 0.001;
    model.add_layer(new Flatten());
    model.add_layer(new Linear(784,100));
    model.add_layer(new ReLU());
    model.add_layer(new Linear(100,100));
    model.add_layer(new ReLU());
    model.add_layer(new Linear(100,10));
    model.add_layer(new Softmax());


    Tensor<float> images(batch_size,1,28,28);
    Tensor<float> labels(batch_size,1,1,1);
    Tensor<float> output(batch_size, 1, 1, NUM_CLASSES);
    Tensor<float> loss(1,1,1,1);
    float loss_h[1] = {0.0f};
    for (int ep = 0; ep < epochs; ep++) {
        for (int i = 0; i < loader.size(); i++) {
            model.zero_grad();
            loader.next(images, labels);
            model.forward(images, output);
            cross_entropy_loss(labels, output, loss);
            model.backward(labels);
            model.update(learning_rate);
            loss.cpyToHost(loss_h);
            if (std::isnan(loss_h[0])) {
                std::cout << "nan appeared in iteration: " << i << std::endl;
            }
            if (i % (loader.size()/interval) == 0) {
                std::cout << "Epoch: " << ep << " [" << (int)(i/(loader.size()/interval)) << "/" <<interval << "]" << " \t Loss: " << loss_h[0] << std::endl;

            }
        }
    }


    return 0;
}