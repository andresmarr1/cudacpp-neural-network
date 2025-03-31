//
// Created by andres on 2/3/25.
//
#ifndef UTILS_H
#define UTILS_H
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include "dataset.h"
void allocate_device_memory(float*& d_images, int*& d_labels, int batch_size);
void copy_batch_to_device(const std::vector<float>& batch_images, const std::vector<int>& d_labels, std::vector<float>& d_output);
void free_device_memory(float* d_images, int* d_labels);

#endif //UTILS_H
