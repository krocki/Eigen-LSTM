#ifndef _CU_KERNELS_
#define _CU_KERNELS_

#define dtype float
//max shared mem size for vectors
#define MAX_VEC_LENGTH 4096

//matrix add vector, c = a + vec
__global__ void kernel_colwise_vector_add(dtype* c, dtype* a, dtype* vec, size_t n, size_t rows);
__global__ void kernel_colwise_vector_add_shared(dtype* c, dtype* a, dtype* vec, size_t n, size_t rows);
//matrix div vector, c = a / vec
__global__ void kernel_rowwise_vector_div(dtype* c, dtype* a, dtype* vec, size_t n, size_t rows);
__global__ void kernel_rowwise_vector_div_shared(dtype* c, dtype* a, dtype* vec, size_t n, size_t rows);

__global__ void kernel_elementwise_logistic_block(dtype* c, size_t c0, size_t cols_c, size_t n, size_t stride_c);
__global__ void kernel_elementwise_tanh_block(dtype* c, size_t c0, size_t cols_c, size_t n, size_t stride_c);
__global__ void kernel_elementwise_logistic_prime_block(dtype* c, size_t c0, size_t cols_c, size_t n, size_t stride_c);
__global__ void kernel_elementwise_tanh_prime_block(dtype* c, size_t c0, size_t cols_c, size_t n, size_t stride_c);
__global__ void kernel_elementwise_neglog2_block(dtype* c, size_t c0, size_t cols_c, size_t n, size_t stride_c);
__global__ void kernel_elementwise_exp_block(dtype* c, size_t c0, size_t cols_c, size_t n, size_t stride_c);
__global__ void kernel_elementwise_mult_block(dtype* c, size_t c0, size_t cols_c, dtype* a, size_t a0, size_t cols_a, dtype* b, size_t b0, size_t cols_b, size_t n, size_t stride_c, size_t stride_a, size_t stride_b);
__global__ void kernel_elementwise_div_block(dtype* c, size_t c0, size_t cols_c, dtype* a, size_t a0, size_t cols_a, dtype* b, size_t b0, size_t cols_b, size_t n, size_t stride_c, size_t stride_a, size_t stride_b);
__global__ void kernel_elementwise_div(dtype* c, dtype* a, dtype* b, size_t n);
__global__ void kernel_elementwise_mult(dtype* c, dtype* a, dtype* b, size_t n);
__global__ void kernel_elementwise_mult_block_noadd(dtype* c, size_t c0, size_t cols_c, dtype* a, size_t a0, size_t cols_a, dtype* b, size_t b0, size_t cols_b, size_t n, size_t stride_c, size_t stride_a, size_t stride_b);
__global__ void kernel_elementwise_sub_block(dtype* c, size_t c0, size_t cols_c, dtype* a, size_t a0, size_t cols_a, dtype* b, size_t b0, size_t cols_b, size_t n, size_t stride_c, size_t stride_a, size_t stride_b);
__global__ void kernel_elementwise_add_block(dtype* c, size_t c0, size_t cols_c, dtype* a, size_t a0, size_t cols_a, dtype* b, size_t b0, size_t cols_b, size_t n, size_t stride_c, size_t stride_a, size_t stride_b);
__global__ void kernel_elementwise_square_eps_block(dtype* c, size_t c0, size_t cols_c, size_t n, size_t stride_c);
__global__ void kernel_elementwise_square_eps(dtype* c, size_t n);
__global__ void kernel_encode(unsigned int* data, dtype* d_targets, dtype* d_codes, unsigned int* positions, size_t elements, size_t M);
__global__ void kernel_advance_positions(unsigned int* positions, size_t elements, unsigned int length);
__global__ void kernel_shift(dtype* m, size_t n);
__global__ void kernel_colwise_sum(dtype* sum, dtype* m, size_t n, size_t M, size_t B);
__global__ void kernel_rowwise_sum(dtype* sum, dtype* m, size_t n, size_t M, size_t B);
#endif