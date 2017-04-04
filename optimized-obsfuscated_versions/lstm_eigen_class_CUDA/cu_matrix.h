
/*process only if we are using NVCC*/

#if defined(__GPU__) || defined(__CUDACC__)

#ifndef _CU_MATRIX_H_
#define _CU_MATRIX_H_

#include <cu_kernels.h>

#include <cublas_v2.h>
#include <curand.h>
#include <matrix.h>

typedef struct {

	dtype* 		data;
	size_t 		rows;
	size_t 		cols;

} cuda_matrix;

void cuda_zero_matrix(cuda_matrix* m);
void cuda_alloc_matrix(cuda_matrix* m);
void cuda_alloc_matrix(cuda_matrix* m, size_t rows, size_t cols);
void cuda_free_matrix(cuda_matrix* m);

void cuda_rand_matrix(cuda_matrix* m);
void cuda_randn_matrix(cuda_matrix* m, dtype mean, dtype stddev);

void init_curand( void );
void init_cublas( void );
void teardown_cublas( void );

void cuda_copy_host_to_device(Matrix& m, cuda_matrix* dst);
void cuda_copy_device_to_device(cuda_matrix* src, cuda_matrix* dst);
void cuda_copy_device_to_host(cuda_matrix* src, Matrix& m);
void print_cuda_matrix(cuda_matrix* m);
void compare_matrices(const char* message, cuda_matrix* m, Matrix& n);

//arithmetic
void cuda_matmul(cuda_matrix* c, const cuda_matrix* a, const cuda_matrix* b, dtype beta = (dtype)1, bool aT = false, bool bT = false);
void cuda_matrix_add_vector(cuda_matrix* c, cuda_matrix* a, const cuda_matrix* v);
void cuda_matrix_divide_vector(cuda_matrix* c, cuda_matrix* a, const cuda_matrix* v);

void cuda_elementwise_add(cuda_matrix* c, cuda_matrix* a, cuda_matrix* b);
void cuda_elementwise_sub(cuda_matrix* c, cuda_matrix* a, cuda_matrix* b);
void cuda_elementwise_sub_scalar(cuda_matrix* c, cuda_matrix* a, dtype b);

void cuda_elementwise_logistic(cuda_matrix* c);
void cuda_elementwise_tanh(cuda_matrix* c);
void cuda_elementwise_tanh_prime(cuda_matrix* c);
void cuda_elementwise_mult(cuda_matrix* c, cuda_matrix* a, cuda_matrix* b, dtype beta);
void cuda_elementwise_exp(cuda_matrix* c);

//block versions
void cuda_elementwise_logistic_block(cuda_matrix* c, size_t c0, size_t c1);
void cuda_elementwise_tanh_block(cuda_matrix* c, size_t c0, size_t c1);
void cuda_elementwise_tanh_prime(cuda_matrix* c, size_t c0, size_t c1);
void cuda_elementwise_logistic_prime(cuda_matrix* c, size_t c0, size_t c1);
void cuda_elementwise_tanh_prime_block(cuda_matrix* c, size_t c0, size_t c1);
void cuda_elementwise_logistic_prime_block(cuda_matrix* c, size_t c0, size_t c1);
void cuda_elementwise_mult_block(cuda_matrix* c, size_t c0, size_t c1,
								 cuda_matrix* a, size_t a0, size_t a1,
								 cuda_matrix* b, size_t b0, size_t b1,
								 dtype beta = (dtype)0);

void cuda_elementwise_adagrad(double learning_rate, cuda_matrix* p, cuda_matrix* d,  cuda_matrix* m);


dtype cuda_matrix_sum(cuda_matrix* m);
void cuda_elementwise_neglog(cuda_matrix* c, cuda_matrix* a);
int cuda_max_matrix(cuda_matrix* c);
int cuda_max_array(dtype* array, size_t array_size);
void cuda_elementwise_sub_element(cuda_matrix* c, int idx);

#endif

#endif