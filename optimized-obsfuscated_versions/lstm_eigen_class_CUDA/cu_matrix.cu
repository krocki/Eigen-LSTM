#include <cu_matrix.h>
#include <iostream>
#include <iomanip>

curandGenerator_t prng;
cublasHandle_t handle;

void cuda_zero_matrix(cuda_matrix* m) {

	cudaMemset(m->data, '\0', m->rows * m->cols * sizeof(dtype));

}

void cuda_alloc_matrix(cuda_matrix* m) {

	if (cudaMalloc((void**) & (m->data), m->rows * m->cols * sizeof(dtype)) != cudaSuccess) {

		std::cout << "alloc_cuda_matrix: cudaMalloc failed!" << std::endl;

	} else {

		cuda_zero_matrix(m);
	}

}

void cuda_alloc_matrix(cuda_matrix* m, size_t rows, size_t cols) {

	m->rows = rows;
	m->cols = cols;
	cuda_alloc_matrix(m);

}

void cuda_free_matrix(cuda_matrix* m) {

	cudaFree(m->data);

}

void cuda_rand_matrix(cuda_matrix* m) {

#ifdef PRECISE_MATH
	curandGenerateUniformDouble(prng, m->data, m->rows * m->cols);
#else
	curandGenerateUniform(prng, m->data, m->rows * m->cols);
#endif

}

void cuda_randn_matrix(cuda_matrix* m, dtype mean, dtype stddev) {

#ifdef PRECISE_MATH
	curandGenerateNormalDouble(prng, m->data, m->rows * m->cols, mean, stddev);
#else
	curandGenerateNormal(prng, m->data, m->rows * m->cols, mean, stddev);
#endif
}

void init_curand( void ) {

	curandCreateGenerator( &prng, CURAND_RNG_PSEUDO_DEFAULT );
	curandSetPseudoRandomGeneratorSeed( prng, ( unsigned long long ) clock() );

}

void init_cublas( void ) {

	if ( cublasCreate( &handle ) != CUBLAS_STATUS_SUCCESS ) {

		std::cout << "!!!! CUBLAS initialization error" << std::endl;
	}

}

void teardown_cublas( void ) {

	if ( cublasDestroy( handle ) != CUBLAS_STATUS_SUCCESS ) {
		std::cout << "!!!! CUBLAS shutdown error" << std::endl;
	}

}
void cuda_copy_device_to_host(cuda_matrix* src, Matrix& m) {

	if (cublasGetVector(m.rows() * m.cols(), sizeof(dtype), src->data, 1, m.data(), 1) != CUBLAS_STATUS_SUCCESS) {

		std::cout << "!!!! cublasGetVector error" << std::endl;

	}

}

void cuda_copy_host_to_device(Matrix& m, cuda_matrix* dst) {

	if (cublasSetVector(m.rows() * m.cols(), sizeof(dtype), m.data(), 1, dst->data, 1) != CUBLAS_STATUS_SUCCESS) {

		std::cout << "!!!! cublasSetVector error" << std::endl;

	}

}

void cuda_copy_device_to_device(cuda_matrix* src, cuda_matrix* dst) {

	if (cudaMemcpy(dst->data, src->data, src->rows * src->cols * sizeof(dtype), cudaMemcpyDeviceToDevice) != cudaSuccess) {

		std::cout << "!!!! cudacudaMemcpy error" << std::endl;
	}
}

void print_cuda_matrix(cuda_matrix* m) {

	Matrix temp_cpu_matrix(m->rows, m->cols);
	cuda_copy_device_to_host(m, temp_cpu_matrix);

	std::cout << temp_cpu_matrix << std::endl;

}

void compare_matrices(const char* message, cuda_matrix* m, Matrix& n) {

	Matrix temp_cpu_matrix(m->rows, m->cols);
	Matrix diff(m->rows, m->cols);


	cuda_copy_device_to_host(m, temp_cpu_matrix);
	diff = temp_cpu_matrix - n;

	std::cout << message << ": max error: " << std::setprecision(12) << diff.cwiseAbs().maxCoeff() <<
			  ", mean error: " << diff.cwiseAbs().sum() / float(diff.rows() * diff.cols()) << std::endl;
}

void cuda_matmul(cuda_matrix* c, const cuda_matrix* a, const cuda_matrix* b, dtype beta, bool aT, bool bT) {

	dtype alpha = (dtype)1;

	size_t M = c->rows;
	size_t N = c->cols;
	size_t K = aT ? b->rows : a->cols;

	size_t lda = aT ? K : M;
	size_t ldb = bT ? N : K;
	size_t ldc = M;

	const cublasOperation_t transA = aT ? CUBLAS_OP_T : CUBLAS_OP_N;
	const cublasOperation_t transB = bT ? CUBLAS_OP_T : CUBLAS_OP_N;

#ifdef PRECISE_MATH

	if (cublasDgemm(handle,
					transA, transB,
					M, N, K,
					&alpha,
					(dtype*)a->data, lda,
					(dtype*)b->data, ldb,
					&beta,
					(dtype*)c->data, ldc)

			!= CUBLAS_STATUS_SUCCESS) {

		std::cout << "!!!! cublas_dgemm error" << std::endl;
	}

#else

	if (cublasSgemm(handle,
					transA, transB,
					M, N, K,
					&alpha,
					(dtype*)a->data, lda,
					(dtype*)b->data, ldb,
					&beta,
					(dtype*)c->data, ldc)

			!= CUBLAS_STATUS_SUCCESS) {

		std::cout << "!!!! cublas_sgemm error" << std::endl;
	}
#endif
}

dtype cuda_matrix_sum(cuda_matrix* m) {

	dtype _sum = (dtype)0;

#ifdef PRECISE_MATH
	cublasDasum(handle, m->rows * m->cols, m->data, 1, &_sum);
#else
	cublasSasum(handle, m->rows * m->cols, m->data, 1, &_sum);
#endif
	return _sum;
}

//matrix add vector, c = a + v
void cuda_matrix_add_vector(cuda_matrix* c, cuda_matrix* a, const cuda_matrix* v) {

	size_t num_blocks = (c->rows * c->cols) / NUM_THREADS + 1;
	kernel_matrix_colwise_vector_add <<< num_blocks, NUM_THREADS>>> (c->data, a->data, v->data, c->rows * c->cols, c->rows);

}

void cuda_matrix_divide_vector(cuda_matrix* c, cuda_matrix* a, const cuda_matrix* v) {

	size_t num_blocks = (c->rows * c->cols) / NUM_THREADS + 1;
	kernel_matrix_rowwise_vector_divide <<< num_blocks, NUM_THREADS>>> (c->data, a->data, v->data, c->rows * c->cols, c->rows);

}

void cuda_elementwise_neglog(cuda_matrix* c, cuda_matrix* a) {

	size_t elements = c->rows * c->cols;

	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_neglog <<< num_blocks, NUM_THREADS>>> (c->data, a->data, elements);

}
void cuda_elementwise_add(cuda_matrix* c, cuda_matrix* a, cuda_matrix* b) {

	size_t elements = c->rows * c->cols;

	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_add <<< num_blocks, NUM_THREADS>>> (c->data, a->data, b->data, elements);


}

void cuda_elementwise_sub(cuda_matrix* c, cuda_matrix* a, cuda_matrix* b) {

	size_t elements = c->rows * c->cols;

	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_sub <<< num_blocks, NUM_THREADS>>> (c->data, a->data, b->data, elements);

}

void cuda_elementwise_sub_element(cuda_matrix* c, int idx) {

	size_t elements = c->rows * c->cols;

	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_sub_element <<< num_blocks, NUM_THREADS>>> (c->data, idx, elements);

}

void cuda_elementwise_logistic(cuda_matrix* c) {

	size_t c0 = 0;
	size_t c1 = c->rows;

	size_t elements = (c1 - c0) * c->cols;
	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_logistic_block <<< num_blocks, NUM_THREADS>>> (c->data, c0, (c1 - c0), elements, c->rows - (c1 - c0));

}

void cuda_elementwise_tanh(cuda_matrix* c) {

	size_t c0 = 0;
	size_t c1 = c->rows;

	size_t elements = (c1 - c0) * c->cols;
	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_tanh_block <<< num_blocks, NUM_THREADS>>> (c->data, c0, (c1 - c0), elements, c->rows - (c1 - c0));

}

void cuda_elementwise_tanh_prime(cuda_matrix* c) {

	size_t c0 = 0;
	size_t c1 = c->rows;

	size_t elements = (c1 - c0) * c->cols;
	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_tanh_prime_block <<< num_blocks, NUM_THREADS>>> (c->data, c0, (c1 - c0), elements, c->rows - (c1 - c0));

}

void cuda_elementwise_logistic_prime(cuda_matrix* c) {

	size_t c0 = 0;
	size_t c1 = c->rows;

	size_t elements = (c1 - c0) * c->cols;
	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_logistic_prime_block <<< num_blocks, NUM_THREADS>>> (c->data, c0, (c1 - c0), elements, c->rows - (c1 - c0));

}

void cuda_elementwise_mult(cuda_matrix* c, cuda_matrix* a, cuda_matrix* b, dtype beta) {

	size_t elements = c->rows * c->cols;

	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_mult <<< num_blocks, NUM_THREADS>>> (c->data, a->data, b->data, elements, beta);

}

void cuda_elementwise_exp(cuda_matrix* c) {

	size_t c0 = 0;
	size_t c1 = c->rows;

	size_t elements = (c1 - c0) * c->cols;
	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_exp_block <<< num_blocks, NUM_THREADS>>> (c->data, c0, (c1 - c0), elements, c->rows - (c1 - c0));

}

void cuda_elementwise_logistic_block(cuda_matrix* c, size_t c0, size_t c1) {

	size_t elements = (c1 - c0) * c->cols;
	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_logistic_block <<< num_blocks, NUM_THREADS>>> (c->data, c0, (c1 - c0), elements, c->rows - (c1 - c0));

}

void cuda_elementwise_tanh_block(cuda_matrix* c, size_t c0, size_t c1) {

	size_t elements = (c1 - c0) * c->cols;
	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_tanh_block <<< num_blocks, NUM_THREADS>>> (c->data, c0, (c1 - c0), elements, c->rows - (c1 - c0));

}

void cuda_elementwise_logistic_prime_block(cuda_matrix* c, size_t c0, size_t c1) {

	size_t elements = (c1 - c0) * c->cols;
	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_logistic_prime_block <<< num_blocks, NUM_THREADS>>> (c->data, c0, (c1 - c0), elements, c->rows - (c1 - c0));

}

void cuda_elementwise_tanh_prime_block(cuda_matrix* c, size_t c0, size_t c1) {

	size_t elements = (c1 - c0) * c->cols;
	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_tanh_prime_block <<< num_blocks, NUM_THREADS>>> (c->data, c0, (c1 - c0), elements, c->rows - (c1 - c0));

}

void cuda_elementwise_mult_block(cuda_matrix* c, size_t c0, size_t c1, cuda_matrix* a, size_t a0, size_t a1, cuda_matrix* b, size_t b0, size_t b1, dtype beta) {

	size_t elements = (c1 - c0) * c->cols;

	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_mult_block <<< num_blocks, NUM_THREADS>>> (c->data, c0, (c1 - c0), a->data, a0, (a1 - a0), b->data, b0, (b1 - b0), elements, c->rows - (c1 - c0), a->rows - (a1 - a0), b->rows - (b1 - b0));

}

void cuda_elementwise_adagrad(double learning_rate, cuda_matrix* c, cuda_matrix* a,  cuda_matrix* b) {

	size_t elements = c->rows * c->cols;

	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_cuda_elementwise_adagrad <<< num_blocks, NUM_THREADS>>> (learning_rate, c->data, a->data,  b->data, elements);



}

int cuda_max_matrix(cuda_matrix* c) {

	return cuda_max_array(c->data, c->rows * c->cols);

}

int cuda_max_array(dtype* array, size_t array_size) {

	int max_idx;

#ifdef PRECISE_MATH
	cublasIdamax(handle, array_size, array, 1, &max_idx);
#else
	cublasIsamax(handle, array_size, array, 1, &max_idx);
#endif

	return max_idx;

}
