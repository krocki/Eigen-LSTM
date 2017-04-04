#include <cu_kernels.h>

__global__ void kernel_matrix_colwise_vector_add(dtype* c, dtype* a, dtype* vec, size_t n, size_t rows) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < n) {

		c[tid] = a[tid] + vec[tid % rows];

	}

}

__global__ void kernel_matrix_rowwise_vector_divide(dtype* c, dtype* a, dtype* vec, size_t n, size_t rows) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < n) {

		c[tid] = a[tid] / vec[tid / rows];

	}

}

__global__ void kernel_elementwise_logistic_block(dtype* c, size_t c0, size_t cols_c, size_t n, size_t stride_c) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int cid = c0 + tid + (tid / cols_c) * stride_c;

	if (tid < n) {
#ifdef PRECISE_MATH
		c[cid] = 1 / ((dtype)1 + exp(-c[cid]));
#else
		c[cid] = __frcp_rn((dtype)1 + __expf(-c[cid]));
#endif
	}

}

__global__ void kernel_elementwise_tanh_block(dtype* c, size_t c0, size_t cols_c, size_t n, size_t stride_c) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int cid = c0 + tid + (tid / cols_c) * stride_c;

	if (tid < n) {
#ifdef PRECISE_MATH
		c[cid] = tanh(c[cid]);
#else
		c[cid] = tanhf(c[cid]);
#endif
	}

}

__global__ void kernel_elementwise_tanh_prime_block(dtype* c, size_t c0, size_t cols_c, size_t n, size_t stride_c) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int cid = c0 + tid + (tid / cols_c) * stride_c;

	if (tid < n) {

#ifdef PRECISE_MATH
		dtype x = c[cid];
		c[cid] = (dtype)1.0 - x * x;
#else

		c[cid] = __fmaf_rn(-c[cid], c[cid], 1.0f);
#endif
	}

}

__global__ void kernel_elementwise_logistic_prime_block(dtype* c, size_t c0, size_t cols_c, size_t n, size_t stride_c) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int cid = c0 + tid + (tid / cols_c) * stride_c;

	if (tid < n) {

		dtype x = c[cid];
		c[cid] = x * ((dtype)1 - x);

	}

}

__global__ void kernel_elementwise_mult_block(dtype* c, size_t c0, size_t cols_c, dtype* a, size_t a0, size_t cols_a, dtype* b, size_t b0, size_t cols_b, size_t n, size_t stride_c, size_t stride_a, size_t stride_b) {


	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	int cid = c0 + tid + (tid / cols_c) * stride_c;
	int aid = a0 + tid + (tid / cols_a) * stride_a;
	int bid = b0 + tid + (tid / cols_b) * stride_b;

	if (tid < n) {

#ifdef PRECISE_MATH
		c[cid] += a[aid] * b[bid];
#else
		c[cid] = __fmaf_rn(a[aid], b[bid], c[cid]);
#endif
	}
}

__global__ void kernel_elementwise_mult_block(dtype* c, size_t c0, size_t cols_c,
		dtype* a, size_t a0, size_t cols_a,
		dtype* b, size_t b0, size_t cols_b,
		size_t n, dtype beta,
		size_t stride_c, size_t stride_a, size_t stride_b) {


	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	int cid = c0 + tid + (tid / cols_c) * stride_c;
	int aid = a0 + tid + (tid / cols_a) * stride_a;
	int bid = b0 + tid + (tid / cols_b) * stride_b;

	if (tid < n) {

		c[cid] = beta * c[cid] + a[aid] * b[bid];

	}
}

__global__ void kernel_elementwise_exp_block(dtype* c, size_t c0, size_t cols_c, size_t n, size_t stride_c) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int cid = c0 + tid + (tid / cols_c) * stride_c;

	if (tid < n) {
#ifdef PRECISE_MATH
		c[cid] = exp(c[cid]);
#else
		c[cid] = __expf(c[cid]);
#endif
	}

}

__global__ void kernel_elementwise_mult(dtype* c, dtype* a, dtype* b, size_t n, dtype beta) {


	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < n) {

		c[tid] = c[tid] * beta + a[tid] * b[tid];

	}
}

__global__ void kernel_elementwise_add(dtype* c, dtype* a, dtype* b, size_t n) {


	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < n) {

		c[tid] = a[tid] + b[tid];

	}

}

__global__ void kernel_elementwise_sub(dtype* c, dtype* a, dtype* b, size_t n) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < n) {

		c[tid] = a[tid] - b[tid];

	}

}

__global__ void kernel_elementwise_sub_element(dtype* c, int idx, size_t n) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	dtype val = c[idx];

	if (tid < n) {

		c[tid] = c[tid] - val;

	}

}

__global__ void kernel_cuda_elementwise_adagrad(double learning_rate, dtype* p, dtype* d, dtype* m, size_t n) {


	int tid = blockDim.x * blockIdx.x + threadIdx.x;


	if (tid < n) {

#ifdef PRECISE_MATH
		p[tid] = p[tid] - learning_rate * d[tid] / (sqrt(m[tid] + 1e-10));
#else
		p[tid] = p[tid] - (dtype)learning_rate * d[tid] * (__frsqrt_rn(m[tid] + 1e-10));
#endif
	}


}

__global__ void kernel_elementwise_neglog(dtype* c, dtype* a, size_t n) {


	int tid = blockDim.x * blockIdx.x + threadIdx.x;


	if (tid < n) {
#ifdef PRECISE_MATH
		c[tid] = -log2(a[tid]);
#else
		c[tid] = -__log2f(a[tid]);
#endif
	}

}