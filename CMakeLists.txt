cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "53;61;70;72;75;80;86;87;90")

project(cuda_t LANGUAGES CUDA CXX)

include_directories(${PROJECT_SOURCE_DIR}/include)

set(SOURCES
        main.cu
        src/dataset.cu
        src/kernel_ops.cu
        src/relu.cu
        src/linear.cu
        src/maxpool.cu
        src/conv2d.cu
        src/flatten.cu
        src/softmax.cu
        src/loss.cu
        src/layer.cu
        src/data_loader.cu
        src/model.cu
)
set(HEADERS
        include/dataset.h
        include/tensor.cuh
        include/kernel_ops.cuh
        include/layer.cuh
        include/linear.h
        include/relu.h
        include/maxpool.h
        include/conv2d.h
        include/flatten.h
        include/softmax.cuh
        include/loss.cuh
        include/data_loader.h
        include/model.h
)

add_executable(cuda_t ${SOURCES} ${HEADERS})

target_compile_options(cuda_t PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:--threads=0>"
)
set_target_properties(cuda_t PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
