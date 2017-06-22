#include <thrust/device_vector.h>
#include "kmeans.h"
#include "timer.h"
#include "util.h"
#include <iostream>

#include <cstdlib>

template<typename T>
void fill_array(T& array, int m, int n) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            array[i * n + j] = (i % 2)*3 + j;
        }
    }
}

template<typename T>
void random_data(thrust::device_vector<T>& array, int n, int d, int k) {
  thrust::host_vector<T> host_array(n*d);
  for(int i = 0; i < n; i++) {
  for(int j = 0; j < d; j++) {
    //    host_array[i] = (T)rand()/(T)RAND_MAX;
     host_array[i*d+j] = i%k;
     fprintf(stderr,"i=%d d=%d : %g\n",i,d,host_array[i*d+j]);  fflush(stderr);
    //    host_array[j*n+i] = i%k;
  }
  }
  array = host_array;
}
void random_data_orig(thrust::device_vector<double>& array, int m, int n) {
    thrust::host_vector<double> host_array(m*n);
    for(int i = 0; i < m * n; i++) {
        host_array[i] = (double)rand()/(double)RAND_MAX;
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


void tiny_test() {
    int iterations = 1;
    int n = 5;
    int d = 3;
    int k = 2;

    
    thrust::device_vector<double> data(n * d);
    thrust::device_vector<int> labels(n);
    thrust::device_vector<double> centroids(k * d);
    thrust::device_vector<double> distances(n);
    
    fill_array(data, n, d);
    std::cout << "Data: " << std::endl;
    print_array(data, n, d);

    labels[0] = 0;
    labels[1] = 0;
    labels[2] = 0;
    labels[3] = 1;
    labels[4] = 1;

    std::cout << "Labels: " << std::endl;
    print_array(labels, n, 1);
    
    int i = kmeans::kmeans(iterations, n, d, k, data, labels, centroids, distances);
    std::cout << "Performed " << i << " iterations" << std::endl;

    std::cout << "Labels: " << std::endl;
    print_array(labels, n, 1);

    std::cout << "Centroids:" << std::endl;
    print_array(centroids, k, d);

    std::cout << "Distances:" << std::endl;
    print_array(distances, n, 1);

}


void more_tiny_test() {
	double dataset[] = {
		0.5, 0.5,
		1.5, 0.5,
		1.5, 1.5,
		0.5, 1.5,
		1.1, 1.2,
		0.5, 15.5,
		1.5, 15.5,
		1.5, 16.5,
		0.5, 16.5,
		1.2, 16.1,
		15.5, 15.5,
		16.5, 15.5,
		16.5, 16.5,
		15.5, 16.5,
		15.6, 16.2,
		15.5, 0.5,
		16.5, 0.5,
		16.5, 1.5,
		15.5, 1.5,
		15.7, 1.6};
	double centers[] = {
		0.5, 0.5,
		1.5, 0.5,
		1.5, 1.5,
		0.5, 1.5};
	 
    int iterations = 3;
    int n = 20;
    int d = 2;
    int k = 4;
	
	thrust::device_vector<double> data(dataset, dataset+n*d);
    thrust::device_vector<int> labels(n);
    thrust::device_vector<double> centroids(centers, centers+k*d);
    thrust::device_vector<double> distances(n);
    
    kmeans::kmeans(iterations, n, d, k, data, labels, centroids, distances, false);

    std::cout << "Labels: " << std::endl;
    print_array(labels, n, 1);

    std::cout << "Centroids:" << std::endl;
    print_array(centroids, k, d);

}



int main() {
    std::cout << "Input a character to choose a test:" << std::endl;
    std::cout << "Tiny test: t" << std::endl;
    std::cout << "More tiny test: m" << std::endl;
    std::cout << "Huge test: h: " << std::endl;
    char c;
    std::cin >> c;
    switch (c) {
    case 't':
        tiny_test();
        exit(0);
    case 'm':
        more_tiny_test();
        exit(0);
    case 'h':
        break;
    default:
        std::cout << "Choice not understood, running huge test" << std::endl;
    }
    int iterations = 50;
    int n = 3;
    int d = 3;
    int k = n;

    thrust::device_vector<double> data(n * d);
    thrust::device_vector<int> labels(n);
    thrust::device_vector<double> centroids(k * d);
    thrust::device_vector<double> distances(n);
    
    std::cout << "Generating random data" << std::endl;
    std::cout << "Number of points: " << n << std::endl;
    std::cout << "Number of dimensions: " << d << std::endl;
    std::cout << "Number of clusters: " << k << std::endl;
    std::cout << "Number of iterations: " << iterations << std::endl;
    
    random_data(data, n, d, k);
    random_labels(labels, n, k);
    kmeans::timer t;
    t.start();
    kmeans::kmeans(iterations, n, d, k, data, labels, centroids, distances);
    float time = t.stop();
    std::cout << "  Time: " << time/1000.0 << " s" << std::endl;

      // debug
  int printcenters=1;
  if(printcenters){
    fprintf(stderr,"centers\n"); fflush(stderr);
    thrust::host_vector<double> *ctr = new thrust::host_vector<double>(centroids);
    for(unsigned int ii=0;ii<k;ii++){
      fprintf(stderr,"ii=%d of k=%d ",ii,k);
      for(unsigned int jj=0;jj<d;jj++){
        fprintf(stderr,"%g ",(*ctr)[d*ii+jj]);
      }
      fprintf(stderr,"\n");
      fflush(stderr);
    }
  }
  int printlabels=1;
  if(printlabels){
    fprintf(stderr,"labels\n"); fflush(stderr);
    thrust::host_vector<int> *lbl = new thrust::host_vector<int>(labels);
    for(unsigned int ii=0;ii<n;ii++){
      fprintf(stderr,"ii=%d of n=%d ",ii,n);
      fprintf(stderr,"%d\n",(*lbl)[ii]);
      fflush(stderr);
    }
  }
  
}
