#
#  Author: Kamil Rocki <kmrocki@us.ibm.com>
#  Created on: 02/15/2016
#
# C++/CUDA

OS := $(shell uname)

CUDA_PRECISE_MATH=1
CC=g++
NVCC=nvcc
INCLUDES=-I.
LFLAGS= -lcudart -lcublas -lcurand
NVCCFLAGS=-m64 --gpu-architecture=sm_52 --resource-usage -O3 --disable-warnings --std=c++11 -D__STRICT_ANSI__ -D_MWAITXINTRIN_H_INCLUDED

ifeq ($(CUDA_PRECISE_MATH), 1)
	NVCCFLAGS := -DPRECISE_MATH $(NVCCFLAGS) --prec-sqrt=true --prec-div=true --ftz=false
else
	NVCCFLAGS := --use_fast_math $(NVCCFLAGS)
endif

#--verbose
CFLAGS=-Ofast -std=c++11

ifeq ($(OS),Linux)
	CFLAGS := $(CFLAGS)
	INCLUDES := -I/usr/include/eigen3 $(INCLUDES) 
else
	#OSX
	INCLUDES := -I/usr/local/include/eigen3 $(INCLUDES)
endif

all:
	$(NVCC) ./cuda_kernels.cu $(INCLUDES) $(NVCCFLAGS) -c cuda_kernels.o
	$(NVCC) ./lstm.cu $(INCLUDES) $(NVCCFLAGS) $(LFLAGS) -o lstm cuda_kernels.o
	#$(NVCC) ./lstm_noeigen.cu $(INCLUDES) $(NVCCFLAGS) $(LFLAGS) -o lstm_noeigen cuda_kernels.o

