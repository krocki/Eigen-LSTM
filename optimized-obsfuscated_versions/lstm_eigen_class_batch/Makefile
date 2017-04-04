#
#  Author: Kamil Rocki <kmrocki@us.ibm.com>
#  Created on: 02/09/2016
#

OS := $(shell uname)

CC=g++
USE_BLAS=1
INCLUDES=-I.
LFLAGS=
CFLAGS=-Ofast -std=c++11

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
	#$(CC) ./lstm_segment.cc $(INCLUDES) $(CFLAGS) $(LFLAGS) -o lstm_segment
	$(CC) ./lstm.cc $(INCLUDES) $(CFLAGS) $(LFLAGS) -o lstm
	

