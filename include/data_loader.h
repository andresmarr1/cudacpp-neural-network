//
// Created by andres on 2/2/25.
//
#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <iostream>
#include <fstream>
#include <cstdint>
#include <algorithm>
#include <random>
#include <vector>
#include "dataset.h"
#include "tensor.cuh"

class DataLoader {
public:
    DataLoader(MNISTDataset& dataset, int batch_size, bool shuffle, int seed);

    void next(Tensor<float>& images, Tensor<float>& labels);

    int size();

private:
    MNISTDataset& dataset;
    std::vector<int> indices;  // Order of samples
    int image_width;
    int image_height;
    int image_channels;
    int batch_size;
    int current_batch;
    int num_samples;
    bool shuffle;
    int seed;

    void shuffle_indices();
};

#endif //DATA_LOADER_H
