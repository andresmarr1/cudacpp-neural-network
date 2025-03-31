//
// Created by andres on 2/27/25.
//

#ifndef TENSOR_CUH
#define TENSOR_CUH
#define TILE_SIZE 16
#define BLOCK_ROWS 8
#define NUM_CLASSES 10
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include "kernel_ops.cuh"

template <typename Tp>
class Tensor {
  public:
    int batch_size, channels, height, width;
    Tp* d_data;

    Tensor(int batch_size, int channels, int height, int width);
    Tensor() : batch_size(0), channels(0), height(0), width(0), d_data(nullptr) {};

    ~Tensor();
    Tensor(Tensor<Tp>&& other) noexcept;

    Tensor<Tp>& operator=(Tensor<Tp>&& other) noexcept {
      if (this != &other) {
        if (d_data != nullptr) {
          cudaFree(d_data);
        }
        batch_size = other.batch_size;
        channels = other.channels;
        height = other.height;
        width = other.width;
        d_data = other.d_data;

        other.d_data = nullptr;
        other.batch_size = 0;
        other.channels = 0;
        other.height = 0;
        other.width = 0;
      }
      return *this;
    }
    void cpyFromHost(const Tp* h_data);
    void cpyFromDevice(const Tp* d_data);
    void cpyToHost(Tp* h_data);
    void reshape(int new_batch_size, int new_channels, int new_height, int new_width);
    void fill(Tp value);
    void batch_mean();
    void pad(int padding);
    Tensor<Tp> clone();
    Tensor<Tp> T();
    Tensor<Tp> T_bc();
    Tensor<Tp> reverse();

    void print_tensor();
    template <typename U>
    friend Tensor<U> operator+(const Tensor<U>& A, const Tensor<U>& B);
    template <typename U>
    friend Tensor<U> operator+(const Tensor<U>& A, U B);
    template <typename U>
    friend Tensor<U> operator-(const Tensor<U>& A, const Tensor<U>& B);
    template <typename U>
    friend Tensor<U> operator-(const Tensor<U>& A, U B);
    template <typename U>
    friend Tensor<U> operator*(const Tensor<U>& A, const Tensor<U>& B);
    template <typename U>
    friend Tensor<U> operator*(const Tensor<U>& A_, U B_);
    template <typename U>
    friend Tensor<U> operator/(const Tensor<U>& A, const Tensor<U>& B);
    template <typename U>
    friend Tensor<U> operator/(const Tensor<U>& A, U B);
    template <typename U>
    friend Tensor<U> matmul(const Tensor<U>& A, const Tensor<U>& B);
    template <typename U>
    friend Tensor<U> convolution2d(const Tensor<U>& A, const Tensor<U>& filters, int stride, int padding);
    template <typename U>
    friend Tensor<U> convolution2d_t(const Tensor<U>& A, const Tensor<U>& filters, int stride, int padding);

// private:
//   void allocate();

};
Tensor<float> one_hot(Tensor<float>& labels);

template <typename Tp>
Tensor<Tp>::Tensor(int batch_size, int channels, int height, int width) : batch_size(batch_size), channels(channels), height(height), width(width) {
  cudaMalloc((void**)&d_data, batch_size * channels * height * width * sizeof(Tp));
}
template <typename Tp>
Tensor<Tp>::~Tensor() {
  if (d_data != nullptr) {
    cudaError_t error = cudaFree(d_data);
    if (error != cudaSuccess) {
      fprintf(stderr, "CUDA destructor error: %s\n", cudaGetErrorString(error));
    }
    d_data = nullptr;
  }

}
template <typename Tp>
Tensor<Tp>::Tensor(Tensor<Tp>&& other) noexcept : batch_size(other.batch_size), channels(other.channels), height(other.height), width(other.width), d_data(other.d_data) {
  other.d_data = nullptr;
  other.batch_size = 0;
  other.channels = 0;
  other.height = 0;
  other.width = 0;
}

template <typename Tp>
void Tensor<Tp>::cpyFromHost(const Tp* h_data) {
  cudaMemcpy(this->d_data, h_data, batch_size*channels*height * width * sizeof(Tp), cudaMemcpyHostToDevice);
}
template <typename Tp>
void Tensor<Tp>::cpyToHost(Tp* h_data) {
  cudaMemcpy(h_data, d_data, batch_size*channels*height * width * sizeof(Tp), cudaMemcpyDeviceToHost);
}
template <typename Tp>
void Tensor<Tp>::fill(Tp value){
  dim3 bDim(16);
  dim3 gDim ((this->batch_size*this->channels*this->height*this->width + bDim.x - 1) / bDim.x);
  fill_k<<<gDim,bDim>>>(this->d_data, value, this->batch_size*this->channels*this->height*this->width);
  cudaDeviceSynchronize();
}
template <typename Tp>
void Tensor<Tp>::reshape(int new_batch_size, int new_channels, int new_height, int new_width) {
  int old_size = batch_size * channels * height * width;
  int new_size = new_batch_size * new_channels * new_height * new_width;

  if (old_size != new_size) return;

  this->batch_size = new_batch_size;
  this->channels = new_channels;
  this->height = new_height;
  this->width = new_width;
}

template <typename Tp>
Tensor<Tp> operator+(const Tensor<Tp>& A, const Tensor<Tp>& B){
  assert(A.batch_size == B.batch_size && A.channels == B.channels && A.height == B.height && A.width == B.width && "Tensor shapes must match");

  Tensor<Tp> res(A.batch_size, A.channels, A.height, A.width);

  dim3 bDim (16);
  dim3 gDim ((A.batch_size*A.channels*A.height*A.width + bDim.x - 1) / bDim.x);
  ops_sb_k<Tp><<<gDim,bDim>>>(A.d_data, B.d_data, res.d_data,1.0, 1.0, 0.0, A.batch_size*A.channels*A.height*A.width);
  cudaDeviceSynchronize();
  return res;
}
template <typename Tp>
Tensor<Tp> operator+(const Tensor<Tp>& A, Tp B){

  Tensor<Tp> res(A.batch_size, A.channels, A.height, A.width);
  Tensor<Tp> t_temp(A.batch_size, A.channels, A.height, A.width);
  t_temp.fill(0);
  dim3 bDim (16);
  dim3 gDim ((A.batch_size*A.channels*A.height*A.width + bDim.x - 1) / bDim.x);
  ops_sb_k<Tp><<<gDim,bDim>>>(A.d_data, t_temp.d_data, res.d_data,1, 0, B, A.batch_size*A.channels*A.height*A.width);
  cudaDeviceSynchronize();
  return res;
}
template <typename Tp>
Tensor<Tp> operator-(const Tensor<Tp>& A, const Tensor<Tp>& B){
  assert(A.batch_size == B.batch_size && A.channels == B.channels && A.height == B.height && A.width == B.width && "Tensor shapes must match");

  Tensor<Tp> res(A.batch_size, A.channels, A.height, A.width);

  dim3 bDim (16);
  dim3 gDim ((A.batch_size*A.channels*A.height*A.width + bDim.x - 1) / bDim.x);
  ops_sb_k<Tp><<<gDim,bDim>>>(A.d_data, B.d_data, res.d_data,1, -1, 0, A.batch_size*A.channels*A.height*A.width);
  cudaDeviceSynchronize();
  return res;
}
template <typename Tp>
Tensor<Tp> operator-(const Tensor<Tp>& A, Tp B){
  //assert(A.batch_size == B.batch_size && A.channels == B.channels && A.height == B.height && A.width == B.width && "Tensor shapes must match");
  Tensor<Tp> res(A.batch_size, A.channels, A.height, A.width);
  Tensor<Tp> t_temp(A.batch_size, A.channels, A.height, A.width);
  t_temp.fill(0);
  dim3 bDim (16);
  dim3 gDim ((A.batch_size*A.channels*A.height*A.width + bDim.x - 1) / bDim.x);
  ops_sb_k<Tp><<<gDim,bDim>>>(A.d_data, t_temp.d_data, res.d_data,1, 0, -1*B, A.batch_size*A.channels*A.height*A.width);
  cudaDeviceSynchronize();
  return res;
}
template <typename Tp>
Tensor<Tp> operator*(const Tensor<Tp>& A, const Tensor<Tp>& B){
  assert(A.batch_size == B.batch_size && A.channels == B.channels && A.height == B.height && A.width == B.width && "Tensor shapes must match");

  Tensor<Tp> res(A.batch_size, A.channels, A.height, A.width);

  dim3 bDim (16);
  dim3 gDim ((A.batch_size*A.channels*A.height*A.width + bDim.x - 1) / bDim.x);
  ops_m_k<Tp><<<gDim, bDim>>>(A.d_data, B.d_data, res.d_data,A.batch_size*A.channels*A.height*A.width);
  cudaDeviceSynchronize();
  return res;
}
template <typename Tp>
Tensor<Tp> operator*(const Tensor<Tp>& A_, Tp B_){

  Tensor<Tp> res(A_.batch_size, A_.channels, A_.height, A_.width);
  Tensor<Tp> t_temp(A_.batch_size, A_.channels, A_.height, A_.width);
  t_temp.fill(B_);
  dim3 bDim (16);
  dim3 gDim ((A_.batch_size*A_.channels*A_.height*A_.width + bDim.x - 1) / bDim.x);

  ops_m_k<Tp><<<gDim, bDim>>>(A_.d_data, t_temp.d_data, res.d_data,A_.batch_size*A_.channels*A_.height*A_.width);
  cudaDeviceSynchronize();
  return res;
}
template <typename Tp>
Tensor<Tp> operator/(const Tensor<Tp>& A, const Tensor<Tp>& B){
  assert(A.batch_size == B.batch_size && A.channels == B.channels && A.height == B.height && A.width == B.width && "Tensor shapes must match");

  Tensor<Tp> res(A.batch_size, A.channels, A.height, A.width);

  dim3 bDim (16);
  dim3 gDim ((A.batch_size*A.channels*A.height*A.width + bDim.x - 1) / bDim.x);
  ops_d_k<Tp><<<gDim, bDim>>>(A.d_data, B.d_data, res.d_data,A.batch_size*A.channels*A.height*A.width);
  cudaDeviceSynchronize();
  return res;
}
template <typename Tp>
Tensor<Tp> operator/(const Tensor<Tp>& A, Tp B){

  Tensor<Tp> res(A.batch_size, A.channels, A.height, A.width);
  Tensor<Tp> t_temp(A.batch_size, A.channels, A.height, A.width);
  t_temp.fill(1);
  dim3 bDim (16);
  dim3 gDim ((A.batch_size*A.channels*A.height*A.width + bDim.x - 1) / bDim.x);
  ops_m_k<Tp><<<gDim, bDim>>>(A.d_data, t_temp.d_data, res.d_data, B,A.batch_size*A.channels*A.height*A.width);
  cudaDeviceSynchronize();
  return res;
}
template <typename Tp>
Tensor<Tp> matmul(const Tensor<Tp>& A, const Tensor<Tp>& B){
  bool broadcast = false;
  assert(A.channels == B.channels && A.width == B.height && "Tensor width and height must match");
  if (B.batch_size == 1) {
    broadcast = true;
  }
  //assert(broadcast == false && B.batch_size == A.batch_size && "Tensor batch sizes must match");
  Tensor<Tp> res(A.batch_size, A.channels, A.height, B.width);

  dim3 bDim(TILE_SIZE,TILE_SIZE);
  dim3 gDim((B.width + TILE_SIZE - 1) / TILE_SIZE, (A.height + TILE_SIZE - 1) / TILE_SIZE, A.batch_size);
  matmul_k<Tp><<<gDim,bDim>>>(A.d_data, B.d_data, res.d_data, A.height, B.width, A.width, broadcast);
  cudaDeviceSynchronize();
  return res;
}

template <typename Tp>
Tensor<Tp> convolution2d(const Tensor<Tp>& A, const Tensor<Tp>& filters, int stride, int padding) {
  int kernel_size = filters.width;
  int output_height = ceil((A.height - kernel_size + 2 * padding) / stride + 1);
  int output_width = ceil((A.width - kernel_size + 2 * padding ) / stride + 1);
  Tensor<Tp> res(A.batch_size, filters.batch_size, output_height, output_width);
  res.fill(0.0f);
  dim3 bDim(16,16, 4);
  dim3 gDim((output_width + bDim.x - 1)/ bDim.x, (output_width + bDim.y - 1 )/ bDim.y, (A.batch_size * filters.batch_size + bDim.z - 1)/bDim.z);
  size_t smem_size = ((bDim.x + kernel_size - 1) * (bDim.y + kernel_size - 1) + A.channels * kernel_size * kernel_size) * sizeof(Tp);
  conv2d_k<Tp><<<gDim, bDim, smem_size>>>(A.d_data, filters.d_data, res.d_data,
                                A.batch_size, A.channels, A.height,
                                    A.width, kernel_size, stride, padding,
                                          output_height, output_width, res.channels);
  cudaError_t error = cudaGetLastError();
  checkCudaError(error, "convolution2d function");
  cudaDeviceSynchronize();
  return res;
}
template <typename Tp>
Tensor<Tp> convolution2d_t(const Tensor<Tp>& A, const Tensor<Tp>& filters, int stride, int padding) {
  int kernel_size = filters.width;
  int output_height = ceil((A.height - 1) * stride - 2 * padding + kernel_size);
  int output_width = ceil((A.width - 1) * stride - 2 * padding + kernel_size);
  Tensor<Tp> res(A.batch_size, filters.batch_size, output_height, output_width);
  // grad_input_height = (grad_output_height - 1) * stride - 2 * padding + kernel_size
  dim3 bDim(16,16, 4);
  dim3 gDim((output_width + bDim.x - 1)/ bDim.x, (output_width + bDim.y - 1 )/ bDim.y, A.batch_size * filters.batch_size);
  size_t smem_size = (bDim.x + kernel_size - 1) * (bDim.x + kernel_size - 1) * sizeof(Tp);
  conv2dt_k<Tp><<<gDim, bDim, smem_size>>>(A.d_data, filters.d_data, res.d_data, A.batch_size, A.channels, A.height, A.width, kernel_size, stride, padding, output_height, output_width, res.channels);
  cudaDeviceSynchronize();
  return res;
}
template<typename Tp>
Tensor<Tp> Tensor<Tp>::clone() {
  Tensor<Tp> n_tensor(this->batch_size, this->channels, this->height, this->width);
  cudaMemcpy(n_tensor.d_data, this->d_data, this->batch_size * this->channels * this->height * this->width * sizeof(Tp), cudaMemcpyDeviceToHost);
  return n_tensor;
}

template <typename Tp>
void Tensor<Tp>::print_tensor(){
  Tp* h_tensor;
  int size_tensor = this->batch_size * this->channels * this->height * this->width;
  h_tensor = (Tp *)malloc(size_tensor * sizeof(Tp));
  cpyToHost(h_tensor);
  for (int i=0; i < size_tensor; i++){
    std::cout << h_tensor[i] << " ";
    if ((i+1)% this->width == 0) std::cout << "\n";
    if ((i+1)% (this->width * this->height) == 0) std::cout << "new channel \n";
    if ((i+1)% (this->width * this->height * this->channels) == 0) std::cout << "new batch \n";
  }
}

template <typename Tp>
Tensor<Tp> Tensor<Tp>::T(){
  Tensor<Tp> n_tensor(this->batch_size, this->channels, this->width, this->height);
  dim3 bDim(16,16);
  dim3 gDim((this->width + 15)/16,(this->height + 15)/16, this->batch_size * this->channels);
  transpose_k<Tp><<<gDim, bDim>>>(this->d_data, n_tensor.d_data, this->batch_size, this->channels, this->height, this->width);
  return n_tensor;
}

template <typename Tp>
Tensor<Tp> Tensor<Tp>::T_bc(){
  Tensor<Tp> n_tensor(this->channels, this->batch_size, this->width, this->height);
  dim3 bDim(TILE_SIZE,TILE_SIZE);
  dim3 gDim((this->width + TILE_SIZE - 1)/TILE_SIZE,(this->height+ TILE_SIZE - 1)/TILE_SIZE, this->batch_size * this->channels);
  transpose_bc_k<Tp><<<gDim, bDim>>>(this->d_data, n_tensor.d_data, this->batch_size, this->channels, this->height, this->width);
  return n_tensor;
}

template <typename Tp>
Tensor<Tp> Tensor<Tp>::reverse(){
  Tensor<Tp> n_tensor(this->batch_size, this->channels, this->width, this->height);
  dim3 bDim(16,16);
  dim3 gDim((this->width + 15)/16,(this->height + 15)/16, this->batch_size * this->channels);
  reverse_k<Tp><<<gDim, bDim>>>(this->d_data, n_tensor.d_data, this->batch_size, this->channels, this->height, this->width);
  return n_tensor;
}
template <typename Tp>
void Tensor<Tp>::batch_mean() {
  float* n_d_data;
  cudaMalloc((void**)&n_d_data, this->channels * this->height * this->width * sizeof(Tp));
  dim3 bDim(16,16,1);
  dim3 gDim((this->width + 15)/16,(this->height + 15)/16, this->channels);
  batched_mean_k<<<gDim, bDim>>>(this->d_data, n_d_data, this->batch_size, this->channels, this->height, this->width);
  cudaDeviceSynchronize();
  cudaFree(d_data);
  cudaMalloc((void**)&d_data, this->channels * this->height * this->width * sizeof(Tp));
  cudaMemcpy(d_data, n_d_data, this->channels * this->height * this->width * sizeof(Tp), cudaMemcpyDeviceToDevice);
  cudaFree(n_d_data);
  this->batch_size = 1;
}

template <typename Tp>
void Tensor<Tp>::pad(const int padding) {
  float* n_d_data;
  cudaMalloc((void**)&n_d_data, this->batch_size * this->channels * (this->height + 2 * padding) * (this->width + 2 * padding) * sizeof(Tp));
  dim3 bDim(16,16, 1);
  dim3 gDim((this->width + 15)/16,(this->height + 15)/16, this->batch_size * this->channels);
  pad_tensor_k<<<gDim, bDim>>>(d_data, n_d_data, this->batch_size, this->channels, this->height, this->width, padding);
  cudaDeviceSynchronize();
  cudaFree(d_data);
  cudaMalloc((void**)&d_data, this->batch_size * this->channels * (this->height + 2 * padding) * (this->width + 2 * padding) * sizeof(Tp));
  cudaMemcpy(d_data, n_d_data, this->batch_size * this->channels * (this->height + 2 * padding) * (this->width + 2 * padding) * sizeof(Tp), cudaMemcpyDeviceToDevice);
  cudaFree(n_d_data);
  this->height = this->height + 2 * padding;
  this->width = this->width + 2 * padding;
}
template <typename Tp>
void Tensor<Tp>::cpyFromDevice(const Tp* d_data) {
  cudaMemcpy(this->d_data, d_data, batch_size*channels*height * width * sizeof(Tp), cudaMemcpyDeviceToDevice);
}

inline Tensor<float> one_hot(Tensor<float>& labels) {
  Tensor<float> onehot_labels(labels.batch_size,1, 1, NUM_CLASSES);
  onehot_labels.fill(0.0f);
  dim3 bDim(16);
  dim3 gDim((labels.batch_size + bDim.x - 1)/ bDim.x);
  get_onehot<<<gDim, bDim>>>(labels.d_data, onehot_labels.d_data, onehot_labels.batch_size);

  cudaError_t error = cudaGetLastError();
  checkCudaError(error, "one hot labels");
  return onehot_labels;
}

#endif //TENSOR_CUH
