#include <thrust/device_vector.h>
#include "timer.h"
#include <iostream>
#include "cuda.h"
#include <cstdlib>
#include "kmeans.h"

template<typename T>
void fill_array(T& array, int m, int n) {
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      array[i * n + j] = (i % 2)*3 + j;
    }
  }
}

template<typename T>
void random_data(thrust::device_vector<T>& array, int m, int n) {
  thrust::host_vector<T> host_array(m*n);
  for(int i = 0; i < m * n; i++) {
    host_array[i] = (T)rand()/(T)RAND_MAX;
  }
  array = host_array;
}

void random_labels(thrust::device_vector<int>& labels, int n, int k) {
  thrust::host_vector<int> host_labels(n);
  for(int i = 0; i < n; i++) {
    host_labels[i] = rand() % k;
  }
  labels = host_labels;
}

typedef float real_t;

int main() {
  int iterations = 100;
  int n = 2000000;
  int d = 29;
  int k = 500;

  int n_gpu;
  cudaGetDeviceCount(&n_gpu);

  std::cout << n_gpu << " gpus." << std::endl;

  thrust::device_vector<real_t> *data[16];
  thrust::device_vector<int> *labels[16];
  thrust::device_vector<real_t> *centroids[16];
  thrust::device_vector<real_t> *distances[16];
  for (int q = 0; q < n_gpu; q++) {
    cudaSetDevice(q);
    data[q] = new thrust::device_vector<real_t>(n/n_gpu*d);
    labels[q] = new thrust::device_vector<int>(n/n_gpu*d);
    centroids[q] = new thrust::device_vector<real_t>(k * d);
    distances[q] = new thrust::device_vector<real_t>(n);
  }

  std::cout << "Generating random data" << std::endl;
  std::cout << "Number of points: " << n << std::endl;
  std::cout << "Number of dimensions: " << d << std::endl;
  std::cout << "Number of clusters: " << k << std::endl;
  std::cout << "Number of iterations: " << iterations << std::endl;

  for (int q = 0; q < n_gpu; q++) {
    random_data<real_t>(*data[q], n/n_gpu, d);
    random_labels(*labels[q], n/n_gpu, k);
  }
  kmeans::timer t;
  t.start();
  kmeans::kmeans<real_t>(iterations, n, d, k, data, labels, centroids, distances, n_gpu);
  float time = t.stop();
  std::cout << "  Time: " << time/1000.0 << " s" << std::endl;

  for (int q = 0; q < n_gpu; q++) {
    delete(data[q]);
    delete(labels[q]);
    delete(centroids[q]);
    delete(distances[q]);
  }
}
