//
// Created by andres on 2/4/25.
//
#include "utils.h"

void allocate_device_memory(float*& d_images, int*& d_labels, int batch_size){
  cudaMalloc((void**)&d_images, sizeof(float)*batch_size*IMG_SIZE);
  cudaMalloc((void**)&d_labels, sizeof(int)*batch_size);
}
void copy_batch_to_device(const std::vector<float>& batch_images,const std::vector<int>& batch_labels, const std::vector<float>& d_images, std::vector<ing>& d_labels){
  cudaMemcpy(d_images, batch_images.data(), batch_images.size()*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_labels, batch_labels.data(), d_labels.size()*sizeof(int), cudaMemcpyHostToDevice);
}
void free_device_memory(float* d_images, int* d_labels){
  cudaFree(d_images);
  cudaFree(d_labels);
}