//
// Created by andres on 2/3/25.
//

#ifndef DATASET_H
#define DATASET_H
#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>

#define IMG_SIZE 784
#define MNIST_MEAN 0.1307f
#define MNIST_STD 0.3108f
#define NUM_CLASSES 10



class MNISTDataset {
public:
    MNISTDataset(const char* image_file, const char* label_file);

    int size();  // Get number of samples
    std::pair<std::vector<float>, int> get_item(int index);
    int read_int(std::ifstream &file);
    void load_mnist(const char* image_file,const char* label_file);
    std::tuple<int, int, int> image_shape();
private:
    int image_channels = 1, image_height = 28, image_width = 28;
    std::vector<float> images;
    std::vector<int> labels;
    int num_samples = 0;
};
#endif
