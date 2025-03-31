//
// Created by andres on 2/3/25.
//
#include <iostream>
#include <fstream>
#include <cstdint>
#include <../include/dataset.h>

int MNISTDataset::size() {
    return this->num_samples;
}

std::pair<std::vector<float>, int> MNISTDataset::get_item(int index) {
    // Return (image, label) at given index
    if (index < 0 || index >= this->num_samples) {
        throw std::out_of_range("Out of range in MNISTDataset::get_item");
    }
    std::vector<float> image(images.begin() + index * IMG_SIZE, images.begin() + (index+1) * IMG_SIZE);

    return {image, labels[index]};
}

std::tuple<int, int, int> MNISTDataset::image_shape() {
    return std::make_tuple(image_channels, image_height, image_width);
}

int MNISTDataset::read_int(std::ifstream &file) {
    unsigned char buffer[4];
    file.read(reinterpret_cast<char*>(buffer), 4);
    return (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];
}

// Function to load MNIST images and labels
void MNISTDataset::load_mnist(const char* image_path,const char* label_path) {
    std::ifstream image_file(image_path, std::ios::binary);
    std::ifstream label_file(label_path, std::ios::binary);
    std::cout << image_path << std::endl;
    std::cout << label_path << std::endl;
    if (!image_file.is_open() || !label_file.is_open()) {
        std::cerr << "Error: Unable to open MNIST data files." << std::endl;
        return;
    }

    // Read headers
    int magic_number_images = read_int(image_file);
    int magic_number_labels = read_int(label_file);
    int num_images = read_int(image_file);
    int num_labels = read_int(label_file);
    int rows = read_int(image_file);
    int cols = read_int(image_file);

    if (num_images != num_labels) {
        std::cerr << "Error: Mismatch in image and label count." << std::endl;
        return;
    }

    this->num_samples = num_images;
    int image_size = rows * cols;

    // Allocate host memory
    images.resize(num_samples * image_size);
    labels.resize(num_samples);

    // Read image data
    std::vector<unsigned char> temp_image_buffer(num_samples * image_size);
    image_file.read(reinterpret_cast<char*>(temp_image_buffer.data()), temp_image_buffer.size());

    // Normalize pixel values (0-255 -> 0-1) and substract the mean and the std of mnist dataset
    for (int i = 0; i < num_samples * image_size; ++i) {
        images[i] = ((temp_image_buffer[i] / 255.0f) - MNIST_MEAN) / MNIST_STD;
    }


    // Read label data (as integers)
    std::vector<unsigned char> temp_label_buffer(num_samples);
    label_file.read(reinterpret_cast<char*>(temp_label_buffer.data()), temp_label_buffer.size());

    for (int i = 0; i < num_samples; ++i) {
        labels[i] = static_cast<int>(temp_label_buffer[i]); // Store as integer
    }
    // Close files
    image_file.close();
    label_file.close();
}

MNISTDataset::MNISTDataset(const char* image_file, const char* label_file) {
    // Load MNIST data into `images` and `labels`
    load_mnist(image_file,label_file);
    // images.resize(num_samples * IMG_SIZE);
    // labels.resize(num_samples);
}