#ifndef _CU_KERNELS_H_
#define _CU_KERNELS_H_

#include <datatype.h>

//only NVCC
#if  defined(__CUDACC__)
#define NUM_THREADS 1024

__global__ void kernel_matrix_colwise_vector_add(dtype* c, dtype* a, dtype* vec, size_t n, size_t rows);
__global__ void kernel_matrix_rowwise_vector_divide(dtype* c, dtype* a, dtype* vec, size_t n, size_t cols);
__global__ void kernel_elementwise_logistic_block(dtype* c, size_t c0, size_t cols_c, size_t n, size_t stride_c);
__global__ void kernel_elementwise_tanh_block(dtype* c, size_t c0, size_t cols_c, size_t n, size_t stride_c);
__global__ void kernel_elementwise_tanh_prime_block(dtype* c, size_t c0, size_t cols_c, size_t n, size_t stride_c);
__global__ void kernel_elementwise_logistic_prime_block(dtype* c, size_t c0, size_t cols_c, size_t n, size_t stride_c);
__global__ void kernel_elementwise_mult_block(dtype* c, size_t c0, size_t cols_c,
		dtype* a, size_t a0, size_t cols_a,
		dtype* b, size_t b0, size_t cols_b,
		size_t n, size_t stride_c, size_t stride_a, size_t stride_b);

__global__ void kernel_elementwise_exp_block(dtype* c, size_t c0, size_t cols_c, size_t n, size_t stride_c);
__global__ void kernel_elementwise_mult(dtype* c, dtype* a, dtype* b, size_t n, dtype beta);

__global__ void kernel_elementwise_neglog(dtype* c, dtype* a, size_t n);
__global__ void kernel_elementwise_add(dtype* c, dtype* a, dtype* b, size_t n);
__global__ void kernel_elementwise_sub(dtype* c, dtype* a, dtype* b, size_t n);
__global__ void kernel_elementwise_sub_element(dtype* c, int idx, size_t n);

__global__ void kernel_cuda_elementwise_adagrad(double learning_rate, dtype* p, dtype* d, dtype* m, size_t n);

#endif
#endif