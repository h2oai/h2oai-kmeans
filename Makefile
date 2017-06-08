CUDA_ARCH ?= sm_60
CUB_HOME ?= ./cub

test: test.cu labels.o timer.o
	nvcc -arch=$(CUDA_ARCH) -Xptxas -v -I$(CUB_HOME) -o test test.cu labels.o timer.o -lcublas

labels.o: labels.cu labels.h
	nvcc -arch=$(CUDA_ARCH) -Xptxas -v -I$(CUB_HOME) -c -o labels.o labels.cu

timer.o:
	nvcc -arch=$(CUDA_ARCH) -c -o timer.o timer.cu

clean:
	rm -f *.o test
