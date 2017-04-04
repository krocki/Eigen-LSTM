#include <cuda_kernels.h>

__global__ void kernel_colwise_vector_add(dtype* c, dtype* a, dtype* vec, size_t n, size_t rows) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < n) {

		c[tid] = a[tid] + vec[tid % rows];

	}

}

__global__ void kernel_colwise_vector_add_shared(dtype* c, dtype* a, dtype* vec, size_t n, size_t rows) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	__shared__ float v_shared[MAX_VEC_LENGTH];

	//copy global to shared
	for (int j = 0; j < MAX_VEC_LENGTH; j += blockDim.x)
		v_shared[threadIdx.x + j] = vec[threadIdx.x + j];

	__syncthreads();

	if (tid < n) {

		c[tid] = a[tid] + v_shared[tid % rows];

	}

}

__global__ void kernel_encode(unsigned int* data, dtype* d_targets, dtype* d_codes, unsigned int* positions, size_t n, size_t M) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	int pos = tid / M;
	int ascii = data[positions[pos]];

	if (tid < n) {

		d_targets[tid] = d_codes[ascii * M + tid % M];

	}

}

__global__ void kernel_advance_positions(unsigned int* positions, size_t n, unsigned int length) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < n) {

		positions[tid] = (positions[tid] + 1) % length;

	}

}

__global__ void kernel_shift(float* m, size_t n) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < n && tid > 0) {

		m[tid - 1] = m[tid];

	}
}

__global__ void kernel_rowwise_vector_div(dtype* c, dtype* a, dtype* vec, size_t n, size_t cols) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < n) {

		c[tid] = a[tid] / vec[tid % cols];

	}

}

__global__ void kernel_rowwise_vector_div_shared(dtype* c, dtype* a, dtype* vec, size_t n, size_t cols) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	__shared__ float v_shared[MAX_VEC_LENGTH];

	//copy global to shared
	for (int j = 0; j < MAX_VEC_LENGTH; j += blockDim.x)
		v_shared[threadIdx.x + j] = vec[threadIdx.x + j];

	__syncthreads();

	if (tid < n) {

		c[tid] = a[tid] / v_shared[tid % cols];

	}

}

__global__ void kernel_elementwise_logistic_block(dtype* c, size_t c0, size_t cols_c, size_t n, size_t stride_c) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int cid = c0 + tid + (tid / cols_c) * stride_c;

	if (tid < n) {

		// 1/1+exp(-x)
// #ifdef PRECISE_MATH
		c[cid] = 1 / ((dtype)1 + exp(-c[cid]));
// #else
// 		c[cid] = __frcp_rn((dtype)1 + __expf(-c[cid]));
// #endif
	}

}

//DEBUG CODE
__global__ void kernel_colwise_sum(dtype* sum, dtype* m, size_t n, size_t M, size_t B) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < n) {

		atomicAdd(&sum[tid / M], m[tid]);

	}

}

__global__ void kernel_rowwise_sum(dtype* sum, dtype* m, size_t n, size_t M, size_t B) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < n) {

		atomicAdd(&sum[tid / B], m[tid]);

	}

}

__global__ void kernel_elementwise_tanh_block(dtype* c, size_t c0, size_t cols_c, size_t n, size_t stride_c) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int cid = c0 + tid + (tid / cols_c) * stride_c;

	if (tid < n) {

// #ifdef PRECISE_MATH
		c[cid] = tanh(c[cid]);
// #else
// 		c[cid] = tanhf(c[cid]);
// #endif
	}

}

__global__ void kernel_elementwise_tanh_prime_block(dtype* c, size_t c0, size_t cols_c, size_t n, size_t stride_c) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int cid = c0 + tid + (tid / cols_c) * stride_c;

	if (tid < n) {

		float x = c[cid];
		c[cid] = 1.0f - x * x;

	}

}

__global__ void kernel_elementwise_logistic_prime_block(dtype* c, size_t c0, size_t cols_c, size_t n, size_t stride_c) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int cid = c0 + tid + (tid / cols_c) * stride_c;

	if (tid < n) {

		float x = c[cid];
		c[cid] = x * (1.0f - x);

	}

}


__global__ void kernel_elementwise_neglog2_block(dtype* c, size_t c0, size_t cols_c, size_t n, size_t stride_c) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int cid = c0 + tid + (tid / cols_c) * stride_c;

	if (tid < n) {

// #ifdef PRECISE_MATH
		c[cid] = -logf(c[cid]);
// #else
// 		c[cid] = -log2f(c[cid]);
// #endif
	}

}

__global__ void kernel_elementwise_exp_block(dtype* c, size_t c0, size_t cols_c, size_t n, size_t stride_c) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int cid = c0 + tid + (tid / cols_c) * stride_c;

	if (tid < n) {

// #ifdef PRECISE_MATH
		c[cid] = expf(c[cid]);
// #else
// 		c[cid] = __expf(c[cid]);
// #endif
	}

}

#define eps 1e-4

__global__ void kernel_elementwise_square_eps_block(dtype* c, size_t c0, size_t cols_c, size_t n, size_t stride_c) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int cid = c0 + tid + (tid / cols_c) * stride_c;

	if (tid < n) {

// #ifdef PRECISE_MATH
		c[cid] = sqrtf(c[cid] + eps);
// #else
// 		c[cid] = sqrtf(c[cid] + eps);
// #endif
	}

}

__global__ void kernel_elementwise_square_eps(dtype* c, size_t n) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < n) {

// #ifdef PRECISE_MATH
		c[tid] = sqrtf(c[tid] + eps);
// #else
// 		c[tid] = sqrtf(c[tid] + eps);
// #endif
	}

}

__global__ void kernel_elementwise_div(dtype* c, dtype* a, dtype* b, size_t n) {


	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < n) {

		c[tid] += a[tid] / b[tid];

	}
}

__global__ void kernel_elementwise_mult(dtype* c, dtype* a, dtype* b, size_t n) {


	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < n) {

		c[tid] += a[tid] * b[tid];

	}
}

__global__ void kernel_elementwise_div_block(dtype* c, size_t c0, size_t cols_c, dtype* a, size_t a0, size_t cols_a, dtype* b, size_t b0, size_t cols_b, size_t n, size_t stride_c, size_t stride_a, size_t stride_b) {


	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	int cid = c0 + tid + (tid / cols_c) * stride_c;
	int aid = a0 + tid + (tid / cols_a) * stride_a;
	int bid = b0 + tid + (tid / cols_b) * stride_b;

	if (tid < n) {

		c[cid] += a[aid] / b[bid];

	}
}

__global__ void kernel_elementwise_mult_block(dtype* c, size_t c0, size_t cols_c, dtype* a, size_t a0, size_t cols_a, dtype* b, size_t b0, size_t cols_b, size_t n, size_t stride_c, size_t stride_a, size_t stride_b) {


	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	int cid = c0 + tid + (tid / cols_c) * stride_c;
	int aid = a0 + tid + (tid / cols_a) * stride_a;
	int bid = b0 + tid + (tid / cols_b) * stride_b;

	if (tid < n) {

		c[cid] += a[aid] * b[bid];

	}
}

__global__ void kernel_elementwise_mult_block_noadd(dtype* c, size_t c0, size_t cols_c, dtype* a, size_t a0, size_t cols_a, dtype* b, size_t b0, size_t cols_b, size_t n, size_t stride_c, size_t stride_a, size_t stride_b) {


	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	int cid = c0 + tid + (tid / cols_c) * stride_c;
	int aid = a0 + tid + (tid / cols_a) * stride_a;
	int bid = b0 + tid + (tid / cols_b) * stride_b;

	if (tid < n) {

		c[cid] = a[aid] * b[bid];

	}
}

__global__ void kernel_elementwise_sub_block(dtype* c, size_t c0, size_t cols_c, dtype* a, size_t a0, size_t cols_a, dtype* b, size_t b0, size_t cols_b, size_t n, size_t stride_c, size_t stride_a, size_t stride_b) {


	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	int cid = c0 + tid + (tid / cols_c) * stride_c;
	int aid = a0 + tid + (tid / cols_a) * stride_a;
	int bid = b0 + tid + (tid / cols_b) * stride_b;

	if (tid < n) {

		c[cid] = a[aid] - b[bid];

	}
}

__global__ void kernel_elementwise_add_block(dtype* c, size_t c0, size_t cols_c, dtype* a, size_t a0, size_t cols_a, dtype* b, size_t b0, size_t cols_b, size_t n, size_t stride_c, size_t stride_a, size_t stride_b) {


	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	int cid = c0 + tid + (tid / cols_c) * stride_c;
	int aid = a0 + tid + (tid / cols_a) * stride_a;
	int bid = b0 + tid + (tid / cols_b) * stride_b;

	if (tid < n) {

		c[cid] = a[aid] + b[bid];

	}
}