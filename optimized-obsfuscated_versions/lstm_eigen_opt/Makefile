#
#  Author: Kamil Rocki <kmrocki@us.ibm.com>
#  Created on: 02/09/2016
#

#CC=g++-5.2.0
CC=g++
#ubuntu
#INCLUDES=-I/usr/include/eigen3 -I.
INCLUDES=-I/usr/local/include/eigen3 -I.
LFLAGS=
#ubuntu
#-mavx2 -march=native -ffast-math
CFLAGS= -O3 -std=c++11

all:
	$(CC) ./lstm.cc $(INCLUDES) $(CFLAGS) $(LFLAGS) -o lstm
