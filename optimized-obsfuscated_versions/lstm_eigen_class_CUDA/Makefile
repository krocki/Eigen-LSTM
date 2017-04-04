#
#  Author: Kamil Rocki <kmrocki@us.ibm.com>
#  Created on: 02/23/2016
#

OS := $(shell uname)

GPU_MODE=0
CUDA_PRECISE_MATH=0
CC=g++
NVCC=nvcc
LINKER=g++

USE_BLAS=1
INCLUDES=-I.

CFLAGS=-O3 -std=c++11

ifeq ($(OS),Linux)
GPU_MODE=1
else
GPU_MODE=0
endif

ifeq ($(GPU_MODE),1)
CC=$(NVCC)
LINKER=$(NVCC)
NVCCFLAGS=-Xcompiler $(CFLAGS) -m64 -ccbin=g++ --gpu-architecture=sm_52 -D__STRICT_ANSI__ -D_MWAITXINTRIN_H_INCLUDED -lcuda -lcudart
LFLAGS=-L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
CFLAGS := -D__GPU__ $(CFLAGS)
ifeq ($(CUDA_PRECISE_MATH),1)
	NVCCFLAGS := -DPRECISE_MATH $(NVCCFLAGS)
	CFLAGS := -DPRECISE_MATH $(CFLAGS)
else
	NVCCFLAGS := --use_fast_math $(NVCCFLAGS)
endif
else
CFLAGS := -DPRECISE_MATH $(CFLAGS)
CC=g++
LINKER=g++
endif



ifeq ($(OS),Linux)
	CFLAGS := $(CFLAGS)
	INCLUDES := -I/usr/include/eigen3 $(INCLUDES) 
else
	#OSX
	INCLUDES := -I/usr/local/include/eigen3 $(INCLUDES)
endif

ifeq ($(USE_BLAS),1)

	ifeq ($(OS),Linux)
		INCLUDES := -I/opt/OpenBLAS/include $(INCLUDES)
		LFLAGS := -lopenblas -L/opt/OpenBLAS/lib $(LFLAGS)
	else
		#OSX
		INCLUDES := -I/usr/local/opt/openblas/include $(INCLUDES)
		LFLAGS := -lopenblas -L/usr/local/opt/openblas/lib $(LFLAGS)
	endif

	CFLAGS := -DUSE_BLAS $(CFLAGS)

endif

all:
	$(NVCC) ./cu_kernels.cu $(INCLUDES) $(NVCCFLAGS) -c cu_kernels.o
	$(NVCC) ./cu_matrix.cu $(INCLUDES) $(NVCCFLAGS) -c cu_matrix.o
	$(NVCC) -c ./lstm.cc $(INCLUDES) $(CFLAGS) -o lstm.o
	$(LINKER) $(LFLAGS) lstm.o cu_kernels.o cu_matrix.o -o lstm

cpu:
	$(CC) $(LFLAGS) $(INCLUDES) $(CFLAGS) lstm.cc -o lstm
