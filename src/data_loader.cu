#include "data_loader.h"
#include <algorithm>
#include <iostream>
#include <random>

DataLoader::DataLoader(MNISTDataset& dataset, int batch_size, bool shuffle, int seed)
    : dataset(dataset), batch_size(batch_size), current_batch(0), num_samples(dataset.size()), shuffle(shuffle),
    seed(seed) {

    std::tie(image_channels, image_height, image_width) = dataset.image_shape();
    indices.resize(num_samples);
    for (int i = 0; i < num_samples; i++) {
      indices[i] = i;
    }
    if (shuffle) {
      shuffle_indices();
    }
}

void DataLoader::shuffle_indices() {
    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);
}

int DataLoader::size() {
    return (num_samples + batch_size - 1)/batch_size;
}

void DataLoader::next(Tensor<float>& images, Tensor<float>& labels) {

    if (current_batch * batch_size >= num_samples) {
        current_batch = 0;
        shuffle_indices();
    }

    int start_idx = current_batch * batch_size;
    int end_idx = std::min(start_idx + batch_size, num_samples);
    int real_batch_size = end_idx - start_idx;
    std::vector<float> batch_images(IMG_SIZE * real_batch_size);
    std::vector<float> batch_labels(real_batch_size);
    for (int i = 0; i < real_batch_size; i++) {
        int dataset_idx = indices[start_idx + i];

        auto item = dataset.get_item(dataset_idx);

        std::copy(item.first.begin(), item.first.end(), batch_images.begin() + i * IMG_SIZE);

        batch_labels[i] = static_cast<float>(item.second);
        // if (current_batch == 0 && i < 3) {
        //     printf("Debug - Batch 0, Item %d: Label = %d (stored as %.1f)\n",
        //            i, item.second, batch_labels[i]);
        // }
    }
    current_batch++;

    images.cpyFromHost(batch_images.data());
    labels.cpyFromHost(batch_labels.data());

}