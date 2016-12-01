all: hsexec

hsexec: kernels.o main.cpp
	g++ -fopenmp -std=c++11 -Wall -o hsexec -L/usr/local/cuda/lib64 -lcuda -lcudart main.cpp  kernels.o 	

kernels.o: kernels.cu
	nvcc -c  --ptxas-options=-v -Xcompiler -fopenmp -O3 -std=c++11  -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -arch=compute_20 kernels.cu  

clean:
	rm -rf *o hsexec


