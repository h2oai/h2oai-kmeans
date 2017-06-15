CUDA_ARCH ?= sm_52
CUB_HOME ?= ./cub

test: test.cu labels.o timer.o kmeans.h
	nvcc -arch=$(CUDA_ARCH) -Xptxas -v -I$(CUB_HOME) -o test test.cu labels.o timer.o -lcublas -lineinfo

labels.o: labels.cu labels.h
	nvcc -arch=$(CUDA_ARCH) -Xptxas -v -I$(CUB_HOME) -c -o labels.o labels.cu -lineinfo

timer.o:
	nvcc -arch=$(CUDA_ARCH) -c -o timer.o timer.cu -lineinfo

clean:
	rm -f *.o test
