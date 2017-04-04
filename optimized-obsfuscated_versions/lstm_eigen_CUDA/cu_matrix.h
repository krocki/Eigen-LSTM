#ifndef __CU_MATRIX_H__
#define __CU_MATRIX_H__

//FP32 - for sgemmEx
#define dtype float
#define CUBLAS_DATA_TYPE CUBLAS_DATA_FLOAT

#include <iostream>

#include <Eigen/Dense>
#include <cublas_v2.h>
#include <curand.h>

#include <cuda_kernels.h>

#define NUM_THREADS 1024

curandGenerator_t prng;
cublasHandle_t handle;

//unfortunately, still need to have this
typedef struct {

	dtype* 		data;
	size_t 		rows;
	size_t 		cols;
	bool 		transposed;

} cuda_matrix;

void cuda_zero_matrix(cuda_matrix* m) {

	cudaMemset(m->data, '\0', m->rows * m->cols * sizeof(dtype));
	m->transposed = false;

}

void cuda_alloc_matrix(cuda_matrix* m) {

	if (cudaMalloc((void**) & (m->data), m->rows * m->cols * sizeof(dtype)) != cudaSuccess) {

		std::cout << "alloc_cuda_matrix: cudaMalloc failed!" << std::endl;

	} else {

		cuda_zero_matrix(m);
	}

}

void cuda_alloc_matrix(cuda_matrix* m, Eigen::MatrixXf e) {

	m->rows = e.rows();
	m->cols = e.cols();
	m->transposed = false;
	cuda_alloc_matrix(m);

}

void cuda_alloc_matrix(cuda_matrix* m, size_t rows, size_t cols) {

	m->rows = rows;
	m->cols = cols;
	m->transposed = false;
	cuda_alloc_matrix(m);

}

void cuda_free_matrix(cuda_matrix* m) {

	cudaFree(m->data);

}

void cuda_copy_host_to_device(Eigen::MatrixXf& m, cuda_matrix* dst) {

	if (cublasSetVector(m.rows() * m.cols(), sizeof(dtype), m.data(), 1, dst->data, 1) != CUBLAS_STATUS_SUCCESS) {

		std::cout << "!!!! cublasSetVector error" << std::endl;

	}

}

void cuda_copy_host_to_device(Eigen::VectorXf& m, cuda_matrix* dst) {

	if (cublasSetVector(m.rows() * m.cols(), sizeof(dtype), m.data(), 1, dst->data, 1) != CUBLAS_STATUS_SUCCESS) {

		std::cout << "!!!! cublasSetVector error" << std::endl;

	}

}

void cuda_copy_device_to_host(cuda_matrix* src, Eigen::MatrixXf& m) {

	if (cublasGetVector(m.rows() * m.cols(), sizeof(dtype), src->data, 1, m.data(), 1) != CUBLAS_STATUS_SUCCESS) {

		std::cout << "!!!! cublasGetVector error" << std::endl;

	}

}

void cuda_copy_device_to_host(cuda_matrix* src, Eigen::VectorXf& m) {

	if (cublasGetVector(m.rows() * m.cols(), sizeof(dtype), src->data, 1, m.data(), 1) != CUBLAS_STATUS_SUCCESS) {

		std::cout << "!!!! cublasGetVector error" << std::endl;

	}

}

void cuda_copy_device_to_device(cuda_matrix* src, cuda_matrix* dst) {

	if (cudaMemcpy(dst->data, src->data, src->rows * src->cols * sizeof(dtype), cudaMemcpyDeviceToDevice) != cudaSuccess) {

		std::cout << "!!!! cudacudaMemcpy error" << std::endl;
	}
}

void cuda_rand_matrix(cuda_matrix* m) {

	curandGenerateUniform(prng, m->data, m->rows * m->cols);

}

void cuda_randn_matrix(cuda_matrix* m, float mean, float stddev) {

	curandGenerateNormal(prng, m->data, m->rows * m->cols, mean, stddev);
}

void cuda_matmul(cuda_matrix* c, cuda_matrix* a, cuda_matrix* b, bool aT = false, bool bT = false) {

	float alpha = 1.0f;
	float beta = 1.0f;

	size_t M = c->rows;
	size_t N = c->cols;
	size_t K = aT ? b->rows : a->cols;

	size_t lda = aT ? K : M;
	size_t ldb = bT ? N : K;
	size_t ldc = M;

	const cublasOperation_t transA = aT ? CUBLAS_OP_T : CUBLAS_OP_N;
	const cublasOperation_t transB = bT ? CUBLAS_OP_T : CUBLAS_OP_N;

	if (cublasSgemmEx(handle,
					  transA, transB,
					  M, N, K,
					  &alpha,
					  a->data, CUBLAS_DATA_TYPE, lda,
					  b->data, CUBLAS_DATA_TYPE, ldb,
					  &beta,
					  c->data, CUBLAS_DATA_TYPE, ldc)

			!= CUBLAS_STATUS_SUCCESS) {

		std::cout << "!!!! cublasSgemm error" << std::endl;
	}

}
//matrix add vector, c = a + vec
void cuda_gmav(cuda_matrix* c, cuda_matrix* a, cuda_matrix* vec) {

	size_t num_blocks = (c->rows * c->cols) / NUM_THREADS + 1;

	if (c->rows > MAX_VEC_LENGTH) {

		std::cout << "!!!! cuda_gmav: c->rows > MAX_VEC_LENGTH " << std::endl;

	}

	kernel_colwise_vector_add <<< num_blocks, NUM_THREADS>>> (c->data, a->data, vec->data, c->rows * c->cols, c->rows);

}

//matrix div vector, c = a / vec
void cuda_gmdv(cuda_matrix* c, cuda_matrix* a, cuda_matrix* vec) {

	size_t num_blocks = (c->rows * c->cols) / NUM_THREADS + 1;

	if (c->rows > MAX_VEC_LENGTH) {

		std::cout << "!!!! cuda_gmav: c->rows > MAX_VEC_LENGTH " << std::endl;

	}

	kernel_rowwise_vector_div <<< num_blocks, NUM_THREADS>>> (c->data, a->data, vec->data, c->rows * c->cols, c->cols);

}

void cuda_elementwise_logistic_block(cuda_matrix* c, size_t c0, size_t c1) {

	if ((c1 - c0) < 1) {

		std::cout << "!!!! cuda_elementwise_logistic: num_rows <= 0 " << std::endl;

	}

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

void cuda_elementwise_sub(cuda_matrix* c, cuda_matrix* a, cuda_matrix* b) {

	size_t c0 = 0;
	size_t c1 = c->rows;

	size_t elements = (c1 - c0) * c->cols;
	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_sub_block <<< num_blocks, NUM_THREADS>>> (c->data, c0, (c1 - c0), a->data, c0, (c1 - c0), b->data, c0, (c1 - c0), elements, c->rows - (c1 - c0), c->rows - (c1 - c0), c->rows - (c1 - c0));

}

void cuda_elementwise_add(cuda_matrix* c, cuda_matrix* a, cuda_matrix* b) {

	size_t c0 = 0;
	size_t c1 = c->rows;

	size_t elements = (c1 - c0) * c->cols;
	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_add_block <<< num_blocks, NUM_THREADS>>> (c->data, c0, (c1 - c0), a->data, c0, (c1 - c0), b->data, c0, (c1 - c0), elements, c->rows - (c1 - c0), c->rows - (c1 - c0), c->rows - (c1 - c0));

}

void cuda_elementwise_neglog2(cuda_matrix* c) {

	size_t c0 = 0;
	size_t c1 = c->rows;

	size_t elements = (c1 - c0) * c->cols;
	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_neglog2_block <<< num_blocks, NUM_THREADS>>> (c->data, c0, (c1 - c0), elements, c->rows - (c1 - c0));

}

void cuda_elementwise_exp(cuda_matrix* c) {

	size_t c0 = 0;
	size_t c1 = c->rows;

	size_t elements = (c1 - c0) * c->cols;
	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_exp_block <<< num_blocks, NUM_THREADS>>> (c->data, c0, (c1 - c0), elements, c->rows - (c1 - c0));

}

void cuda_elementwise_tanh_block(cuda_matrix* c, size_t c0, size_t c1) {

	if ((c1 - c0) < 1) {

		std::cout << "!!!! cuda_elementwise_tanh: num_rows <= 0 " << std::endl;

	}

	size_t elements = (c1 - c0) * c->cols;
	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_tanh_block <<< num_blocks, NUM_THREADS>>> (c->data, c0, (c1 - c0), elements, c->rows - (c1 - c0));

}

void cuda_elementwise_logistic_prime_block(cuda_matrix* c, size_t c0, size_t c1) {

	if ((c1 - c0) < 1) {

		std::cout << "!!!! cuda_elementwise_logistic_prime_block: num_rows <= 0 " << std::endl;

	}

	size_t elements = (c1 - c0) * c->cols;
	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_logistic_prime_block <<< num_blocks, NUM_THREADS>>> (c->data, c0, (c1 - c0), elements, c->rows - (c1 - c0));

}

void cuda_elementwise_tanh_prime_block(cuda_matrix* c, size_t c0, size_t c1) {

	if ((c1 - c0) < 1) {

		std::cout << "!!!! cuda_elementwise_tanh_prime_block: num_rows <= 0 " << std::endl;

	}

	size_t elements = (c1 - c0) * c->cols;
	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_tanh_prime_block <<< num_blocks, NUM_THREADS>>> (c->data, c0, (c1 - c0), elements, c->rows - (c1 - c0));

}

void cuda_elementwise_sqrt_eps(cuda_matrix* c) {

	size_t elements = c->rows * c->cols;
	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_square_eps <<< num_blocks, NUM_THREADS>>> (c->data, elements);

}

void cuda_elementwise_tanh_prime(cuda_matrix* c) {

	size_t c0 = 0;
	size_t c1 = c->rows;

	if ((c1 - c0) < 1) {

		std::cout << "!!!! cuda_elementwise_tanh_prime: num_rows <= 0 " << std::endl;

	}

	size_t elements = (c1 - c0) * c->cols;
	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_tanh_prime_block <<< num_blocks, NUM_THREADS>>> (c->data, c0, (c1 - c0), elements, c->rows - (c1 - c0));

}

void cuda_elementwise_mult_block(cuda_matrix* c, size_t c0, size_t c1, cuda_matrix* a, size_t a0, size_t a1, cuda_matrix* b, size_t b0, size_t b1) {

	size_t elements = (c1 - c0) * c->cols;

	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_mult_block <<< num_blocks, NUM_THREADS>>> (c->data, c0, (c1 - c0), a->data, a0, (a1 - a0), b->data, b0, (b1 - b0), elements, c->rows - (c1 - c0), a->rows - (a1 - a0), b->rows - (b1 - b0));

}

void cuda_elementwise_mult(cuda_matrix* c, cuda_matrix* a, cuda_matrix* b) {

	size_t elements = c->rows * c->cols;

	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_mult <<< num_blocks, NUM_THREADS>>> (c->data, a->data, b->data, elements);

}

void cuda_elementwise_div(cuda_matrix* c, cuda_matrix* a, cuda_matrix* b) {

	size_t elements = c->rows * c->cols;

	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_div <<< num_blocks, NUM_THREADS>>> (c->data, a->data, b->data, elements);

}

void cuda_elementwise_mul_scalar(cuda_matrix* m, const float scalar) {

	cublasSscal(handle, m->rows * m->cols, &scalar, m->data, 1);

}

void cuda_elementwise_mult_noadd(cuda_matrix* c, cuda_matrix* a, cuda_matrix* b) {

	size_t c0 = 0;
	size_t c1 = c->rows;
	size_t elements = (c1 - c0) * c->cols;

	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_elementwise_mult_block_noadd <<< num_blocks, NUM_THREADS>>> (c->data, c0, (c1 - c0), a->data, c0, (c1 - c0), b->data, c0, (c1 - c0), elements, c->rows - (c1 - c0), c->rows - (c1 - c0), c->rows - (c1 - c0));

}

dtype cuda_matrix_sum(cuda_matrix* m) {

	dtype _sum = (dtype)0;

	cublasSasum(handle, m->rows * m->cols, m->data, 1, &_sum);

	return _sum;
}

dtype cuda_colwise_sum(cuda_matrix* n, cuda_matrix* m) {

	size_t M = m->rows;
	size_t B = m->cols;
	size_t elements = M * B;
	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_colwise_sum <<< num_blocks, NUM_THREADS>>> (n->data, m->data, elements, M, B);

}

dtype cuda_rowwise_sum(cuda_matrix* n, cuda_matrix* m) {

	size_t M = m->rows;
	size_t B = m->cols;
	size_t elements = M * B;
	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_rowwise_sum <<< num_blocks, NUM_THREADS>>> (n->data, m->data, elements, M, B);

}

void cuda_ascii_to_codes(unsigned int* d_data, cuda_matrix* d_targets, cuda_matrix* d_codes, unsigned int* positions) {

	size_t M = d_targets->rows;
	size_t B = d_targets->cols;
	size_t elements = M * B;
	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_encode <<< num_blocks, NUM_THREADS>>> (d_data, d_targets->data, d_codes->data, positions, elements, M);

}

void cuda_advance_positions(unsigned int* positions, size_t B, unsigned int length) {

	size_t num_blocks = B / NUM_THREADS + 1;

	kernel_advance_positions <<< num_blocks, NUM_THREADS>>> (positions, B, length);

}

void cuda_shift_matrix(cuda_matrix* m) {

	size_t elements = m->rows * m->cols;
	size_t num_blocks = elements / NUM_THREADS + 1;

	kernel_shift <<< num_blocks, NUM_THREADS>>> (m->data, elements);

}

//compare matrices src and m, diff should be as small as possible
void cuda_check_matrix_error(const char* message, cuda_matrix* src, Eigen::MatrixXf& m) {

	Eigen::MatrixXf cpu_copy = Eigen::MatrixXf(src->rows, src->cols);
	Eigen::MatrixXf diff = Eigen::MatrixXf(src->rows, src->cols);

	cuda_copy_device_to_host(src, cpu_copy);

	// std::cout << m << std::endl << std::endl;
	// std::cout << cpu_copy << std::endl;

	diff = cpu_copy - m;

	// std::cout << "diff matrix" << std::endl << std::endl;

	// std::cout << diff << std::endl;

	std::cout << message << ": max error: " << diff.cwiseAbs().maxCoeff() <<
			  ", mean error: " << diff.cwiseAbs().sum() / float(diff.rows() * diff.cols()) <<
			  ", mean error norm: " << diff.cwiseAbs().sum() / float(diff.rows() * diff.cols()) /
			  (m.cwiseAbs().sum() / float(m.rows() * m.cols())) << std::endl;

}

//compare matrices src and m, diff should be as small as possible
void cuda_check_matrix_error(const char* message, cuda_matrix* src, Eigen::VectorXf& m) {

	Eigen::VectorXf cpu_copy = Eigen::VectorXf(src->rows, src->cols);
	Eigen::VectorXf diff = Eigen::VectorXf(src->rows, src->cols);

	cuda_copy_device_to_host(src, cpu_copy);

	//std::cout << m << std::endl << std::endl;
	//std::cout << cpu_copy << std::endl;

	diff = cpu_copy - m;

	//std::cout << "diff matrix" << std::endl << std::endl;

	//std::cout << diff << std::endl;

	std::cout << message << ": max error: " << diff.cwiseAbs().maxCoeff() <<
			  ", mean error: " << diff.cwiseAbs().sum() / float(diff.rows() * diff.cols()) << std::endl;

}

//compare matrices src and m, diff should be as small as possible
void check_matrix_error(const char* message, Eigen::MatrixXf& n, Eigen::MatrixXf& m) {

	std::cout << m << std::endl << std::endl;
	std::cout << n << std::endl << std::endl;

	Eigen::MatrixXf diff = n - m;

	std::cout << "diff matrix" << std::endl << std::endl;

	std::cout << diff << std::endl;

	std::cout << message << ": max error: " << diff.cwiseAbs().maxCoeff() <<
			  ", mean error: " << diff.cwiseAbs().sum() / float(diff.rows() * diff.cols()) << std::endl;

}

void check_gradient_error(const char* message, Eigen::VectorXf& n, Eigen::VectorXf& m) {

	Eigen::VectorXf diff = n - m;
	Eigen::VectorXf sum = n + m;

	Eigen::VectorXf error;
	error = diff.cwiseAbs().array() / sum.cwiseAbs().array();

	std::cout << message << ": max error: " << error.maxCoeff() <<
			  ", mean error: " << error.sum() / float(error.rows() * error.cols()) << std::endl;

}

void check_matrix_error(const char* message, Eigen::VectorXf& n, Eigen::VectorXf& m) {

	std::cout << m << std::endl << std::endl;
	std::cout << n << std::endl << std::endl;

	Eigen::VectorXf diff = n - m;

	std::cout << "diff matrix" << std::endl << std::endl;

	std::cout << diff << std::endl;

	std::cout << message << ": max error: " << diff.cwiseAbs().maxCoeff() <<
			  ", mean error: " << diff.cwiseAbs().sum() / float(diff.rows() * diff.cols()) << std::endl;

}

void cuda_check_matrix(const char* message, cuda_matrix* src, Eigen::MatrixXf& m) {

	Eigen::MatrixXf cpu_copy = Eigen::MatrixXf(src->rows, src->cols);
	Eigen::MatrixXf diff = Eigen::MatrixXf(src->rows, src->cols);

	cuda_copy_device_to_host(src, cpu_copy);

	std::cout << m << std::endl << std::endl << std::endl;
	std::cout << cpu_copy << std::endl << std::endl;

}

#endif