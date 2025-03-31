//
// Created by andres on 2/8/25.
//
#include "../include/kernel_ops.cuh"
#define TILE_SIZE 16
#define BLOCK_ROWS 8
#define NUM_CLASSES 10

__global__ void initialize_weights(float *weights, int size, int seed, float std) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size){
    curandState state;
    curand_init(seed, i, 0, &state);
    weights[i] = curand_normal(&state)*std;
  }
}

__global__ void initialize_bias(float *bias, int size, int seed, float std) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size){
    curandState state;
    curand_init(seed, i, 0, &state);
    bias[i] = (curand_uniform(&state) -0.5) * 2 * std;
  }
}

template <typename T>
__global__ void fill_k(T *A, T value, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    A[i] = value;
  }
}

template <typename T>
__global__ void ops_sb_k(const T *A, const T *B, T *C, T a_sca, T b_sca, T cnst, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size) {
    C[i] = A[i] * a_sca + B[i] * b_sca + cnst;
  }
}

template <typename T>
__global__ void ops_m_k(const T *A, const T *B, T *C, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size) {
    C[i] = A[i] * B[i];
  }
}

template <typename T>
__global__ void ops_d_k(const T *A, const T *B, T *C, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size) {
    C[i] = A[i] / (B[i]);
  }
}

template <typename T>
__global__ void matmul_k(const T *A, const T *B, T *C, const int M, const int N, const int K, const bool broadcast){
  __shared__ float tile1[TILE_SIZE][TILE_SIZE];
  __shared__ float tile2[TILE_SIZE][TILE_SIZE];

  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int z = blockIdx.z;
  float sum = 0.0f;
  int z_b = broadcast ? 0 : z * K * N;
  for (int t = 0; t < (K + TILE_SIZE - 1)/TILE_SIZE; t++){
    if (row < M && t * TILE_SIZE + threadIdx.x < K)
      tile1[threadIdx.y][threadIdx.x] = A[z * K * M + row * K + (t * TILE_SIZE + threadIdx.x)];
    else tile1[threadIdx.y][threadIdx.x] = 0.0f;

    if (col < N && t * TILE_SIZE + threadIdx.y < K)
      tile2[threadIdx.y][threadIdx.x] = B[z_b + (t * TILE_SIZE + threadIdx.y) * N + col];
    else tile2[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    for (int tk = 0; tk < TILE_SIZE; tk++) sum += tile1[threadIdx.y][tk] * tile2[tk][threadIdx.x];

    __syncthreads();
  }
  if (row < M && col < N) C[z * M * N + row * N + col] = sum;
}

template <typename T>
__global__ void conv2d_k(T *input, T *filters, T *output, int batch_size, int channels, int height, int width,
                          int kernel_size, int stride, int padding,
                          int output_height, int output_width, int out_channels) {

  extern __shared__ char smem[];
  int t_width = blockDim.x + kernel_size - 1;
  int t_height = blockDim.y + kernel_size - 1;
  T *shared_input = reinterpret_cast<T*>(smem);
  T* shared_filters = shared_input + t_width * t_height;
  int idx = threadIdx.x;
  int idy = threadIdx.y;
  int idz = threadIdx.z;

  int row_out = blockIdx.y * blockDim.y + threadIdx.y;
  int col_out = blockIdx.x * blockDim.x + threadIdx.x;

  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int batch_idx = z / out_channels;
  int filter_idx = z % out_channels;

  int filter_v = channels * kernel_size * kernel_size;
  int t_count = blockDim.x * blockDim.y * blockDim.z;
  int ipt = (filter_v + t_count - 1)/ t_count;

  int t_id = (idy * blockDim.x + idx) * blockDim.x + idz;

  for (int i = 0; i < ipt; i++) {
    int filter_idx_rm = t_id * ipt + i;
    if (filter_idx_rm < filter_v) {
      int chan_in = filter_idx_rm / (kernel_size * kernel_size);
      int rem = filter_idx_rm % (kernel_size * kernel_size);
      int ky = rem / kernel_size;
      int kx = rem % kernel_size;

      if (chan_in < channels && ky < kernel_size && kx < kernel_size) {
        int filter_os = ((filter_idx * channels + chan_in) * kernel_size + ky) * kernel_size + kx;
        shared_filters[filter_idx_rm] = filters[filter_os];
      }
    }
  }

  T output_val = 0.0;

  for (int chan_in = 0; chan_in < channels; chan_in++) {
    for (int ky = 0; ky < kernel_size; ky++) {
      for (int kx = 0; kx < kernel_size; kx++) {
        int row_in = row_out * stride + ky - padding;
        int col_in = col_out * stride + kx - padding;

        int smem_idx = (idy + ky) * t_width + (idx + kx);

        if (idx + kx < t_width && idx + ky < output_height) {
          if (row_in >= 0 && row_in < height && col_in >= 0 && col_in < width ) {
            int input_idx = ((batch_idx * channels + chan_in) * height + row_in) * width + col_in;
            shared_input[smem_idx] = input[input_idx];
          }
          else {
            shared_input[smem_idx] = 0.0;
          }
        }
      }
    }

    __syncthreads();

    for (int ky = 0; ky < kernel_size; ky++) {
      for (int kx = 0; kx < kernel_size; kx++) {
        int input_idx = (idy + ky) * t_width + (idx + kx);

        int filter_idx_sm = chan_in * kernel_size * kernel_size + ky * kernel_size + kx;

        output_val += shared_input[input_idx] * shared_filters[filter_idx_sm];
      }
    }
    __syncthreads();
  }

  if (col_out < output_width && row_out < output_height) {
    int output_idx = ((batch_idx * out_channels + filter_idx) * output_height + row_out) * output_width + col_out;
    output[output_idx] = output_val;
  }

}

template <typename T>
__global__ void conv2dt_k(T *input, T *filters, T *output, int batch_size, int channels, int height, int width,
                          int kernel_size, int stride, int padding,
                          int output_height, int output_width, int out_channels) {
  extern __shared__ char smem[];
  T *shared_filter = reinterpret_cast<T*>(smem);


  int row_in = blockIdx.y * blockDim.y + threadIdx.y;
  int col_in = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.z;
  int chan_in = threadIdx.z;

  if (col_in >= width || row_in >= height) return;

  int id_in = ((k * channels + chan_in) * height + row_in) * width + col_in;

  T value = input[id_in];


  for (int ky = 0; ky < kernel_size; ky++) {
    for (int kx = 0; kx < kernel_size; kx++) {
      int row_out = row_in * stride + ky - padding;
      int col_out = col_in * stride + kx - padding;

      if (col_out >= 0 && col_out < output_width && row_out >= 0 && row_out < output_height) {
        int id_out = ((k * out_channels + chan_in) * output_height + row_out) * output_width + col_out;
        int id_filt = ((chan_in * out_channels) * kernel_size + ky) * kernel_size + kx;
        atomicAdd(&output[id_out], value * filters[id_filt]);
      }
    }
  }
}

template <typename T>
__global__ void transpose_k(const T *A, T* B,const int batch,const int channels, const int height, const int width){
  __shared__ float tile[TILE_SIZE][TILE_SIZE+1];
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int chan = blockIdx.z % channels;
  int z = blockIdx.z / channels;


  for (int i = 0; i < TILE_SIZE; i+= BLOCK_ROWS) tile[threadIdx.y+i][threadIdx.x] = A[((z * channels + chan) * height + row + i) * width + col];
  __syncthreads();
  col = blockIdx.y * TILE_SIZE + threadIdx.x;
  row = blockIdx.x * TILE_SIZE + threadIdx.y;
  for (int i = 0; i < TILE_SIZE; i+= BLOCK_ROWS) B[((z * channels + chan) * height + row + i) * width + col] = tile[threadIdx.x][threadIdx.y + i];
}

template <typename T>
__global__ void transpose_bc_k(const T *A, T* B,const int batch,const int channels, const int height, const int width) {
  __shared__ T tile[TILE_SIZE][TILE_SIZE+1];
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int chan = blockIdx.z % channels; // channel_id
  int z = blockIdx.z / channels; // batch_id


  if (col < width && row < height && z < batch) {
    tile[threadIdx.y][threadIdx.x] = A[((z * channels + chan) * height + row) * width + col];
  }
  __syncthreads();
  if (col < width && row < height && z < batch) {
    B[((chan * batch + z) * height + row) * width + col] = tile[threadIdx.y][threadIdx.x];
  }
}

template <typename T>
__global__ void reverse_k(const T *A, T* B,const int batch,const int channels, const int height, const int width){
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int chan = blockIdx.z % channels;
  int z = blockIdx.z / channels;

  __shared__ float tile[TILE_SIZE][TILE_SIZE+1];
  for (int i = 0; i < TILE_SIZE; i+= BLOCK_ROWS) tile[threadIdx.y+i][threadIdx.x] = A[((z * channels + chan) * height + row + i) * width + col];
  __syncthreads();
  for (int i = 0; i < TILE_SIZE; i+= BLOCK_ROWS) B[((z * channels + chan) * height + height - 1 - row - i) * width + width - 1 - col] = tile[threadIdx.y+i][threadIdx.x];
}

template <typename T>
__global__ void batched_mean_k(const T *A, T* B, const int batch, const int channels, const int height, const int width) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.z * blockDim.z + threadIdx.z;

  float sum = 0.0f;
  if (col >= width || row >= height || c >= channels) return;
  for (int i=0; i < batch; i++) {
    sum += A[((i * channels + c) * height + row) * width + col];
  }

  B[(c * height + row) * width + col] = sum / batch;

}

template <typename T>
__global__ void pad_tensor_k(const T *A, T* B, const int batch, const int channels, const int height, const int width, const int padding) {
  __shared__ T tile[TILE_SIZE][TILE_SIZE+1];
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int chan = blockIdx.z % channels;
  int z = blockIdx.z / channels;

  if (col < width && row < height && z < batch) {
    tile[threadIdx.y][threadIdx.x] = A[((z * channels + chan) * height + row) * width + col];
  }
  __syncthreads();
  int output_idx = ((z * channels + chan) * (height + 2*padding) + (row+padding)) * (width + 2*padding) + (col + padding);
  if (col < width && row < height && z < batch) {
    B[output_idx] = tile[threadIdx.y][threadIdx.x];
  }
}
__global__ void get_onehot(float* labels, float* output, int batch) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < batch) {
    int pos = (int)(labels[i] + 0.5f);
    if (pos >= 0 && pos < NUM_CLASSES) {
      output[ i * NUM_CLASSES + pos] = 1.0f;
    }
    else {
      printf("Error: Label %d out of bounds (0-%d) at index %d \n", pos, NUM_CLASSES - 1, i);
    }
  }
}

void checkCudaError(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

template __global__ void matmul_k<float>(const float *A, const float *B, float *C, const int M, const int N, const int K, const bool broadcast);
template __global__ void matmul_k<int>(const int *A, const int *B, int *C, const int M, const int N, const int K, const bool broadcast);

template __global__ void transpose_k<float>(const float *A, float* B,const int batch,const int channels, const int height, const int width);
template __global__ void transpose_k<int>(const int *A, int* B,const int batch,const int channels, const int height, const int width);

template __global__ void transpose_bc_k<float>(const float *A, float* B,const int batch,const int channels, const int height, const int width);
template __global__ void transpose_bc_k<int>(const int *A, int* B,const int batch,const int channels, const int height, const int width);

template __global__ void conv2d_k<float>(float *input, float *filters, float *output, int batch_size, int channels, int height, int width,
                          int kernel_size, int stride, int padding,
                          int output_height, int output_width, int out_channels);
template __global__ void conv2d_k<int>(int *input, int *filters, int *output, int batch_size, int channels, int height, int width,
                          int kernel_size, int stride, int padding,
                          int output_height, int output_width, int out_channels);

template __global__ void conv2dt_k<float>(float *input, float *filters, float *output, int batch_size, int channels, int height, int width,
                          int kernel_size, int stride, int padding,
                          int output_height, int output_width, int out_channels);
template __global__ void conv2dt_k<int>(int *input, int *filters, int *output, int batch_size, int channels, int height, int width,
                          int kernel_size, int stride, int padding,
                          int output_height, int output_width, int out_channels);

template __global__ void reverse_k<float>(const float *A, float* B,const int batch,const int channels, const int height, const int width);
template __global__ void reverse_k<int>(const int *A, int* B,const int batch,const int channels, const int height, const int width);

template __global__ void batched_mean_k<float>(const float *A, float* B, const int batch, const int channels, const int height, const int width);
template __global__ void batched_mean_k<int>(const int *A, int* B, const int batch, const int channels, const int height, const int width);

template __global__ void ops_sb_k<float>(const float *A, const float *B, float *C, float a_sca, float b_sca, float cnst, int size);
template __global__ void ops_sb_k<int>(const int *A, const int *B, int *C, int a_sca, int b_sca, int cnst, int size);

template __global__ void ops_m_k<float>(const float *A, const float *B, float *C, int size);
template __global__ void ops_m_k<int>(const int *A, const int *B, int *C, int size);

template __global__ void ops_d_k<float>(const float *A, const float *B, float *C, int size);
template __global__ void ops_d_k<int>(const int *A, const int *B, int *C, int size);

template __global__ void fill_k<float>(float *A, float value, int size);
template __global__ void fill_k<int>(int *A, int value, int size);

template __global__ void pad_tensor_k<float>(const float *A, float* B, const int batch, const int channels, const int height, const int width, const int padding);
template __global__ void pad_tensor_k<int>(const int *A, int* B, const int batch, const int channels, const int height, const int width, const int padding);