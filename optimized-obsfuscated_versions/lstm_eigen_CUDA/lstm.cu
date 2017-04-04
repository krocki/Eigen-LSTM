//
//  lstm.cu
//
//  LSTM code - Eigen/CUDA implementation
//
//  Author: Kamil Rocki <kmrocki@us.ibm.com>
//  Created on: 02/15/2016
//  TODO: fp16 support
//

#include <iostream>
#include <iomanip>
#include <random>
#include <timer.h>

#include <Eigen/Dense>

//CUDA
#include <cublas_v2.h>
#include <curand.h>

#include <cu_matrix.h>

void init_curand( void );
void init_cublas( void );
void test_cublas (void);
void destroy_cublas( void );

//shouldn't Matrix and Vector have the same base class, TODO
void rand(Eigen::MatrixXf& m, float range_min, float range_max);
void rand(Eigen::VectorXf& m, float range_min, float range_max);
void randn(Eigen::MatrixXf& m, float mean, float stddev);
void randn(Eigen::VectorXf& m, float mean, float stddev);
void randi(Eigen::MatrixXf& m, int range_min, int range_max);
void randi(Eigen::VectorXf& m, int range_min, int range_max);

void CUBLAS_mmul( Eigen::MatrixXf& __restrict c, Eigen::MatrixXf& __restrict a,
				  Eigen::MatrixXf& __restrict b, bool aT = false, bool bT = false);


typedef struct {

	Eigen::MatrixXf dy;
	Eigen::MatrixXf dWhy;
	Eigen::VectorXf dby;
	Eigen::VectorXf eby;
	Eigen::MatrixXf dh;
	Eigen::MatrixXf dc;
	Eigen::MatrixXf dg;
	Eigen::MatrixXf dU;
	Eigen::MatrixXf dW;
	Eigen::MatrixXf db;
	Eigen::MatrixXf dhnext;
	Eigen::MatrixXf dcnext;

} gradients;

gradients check_grads(cuda_matrix* d_g, cuda_matrix* d_x, cuda_matrix d_W, cuda_matrix d_U, cuda_matrix d_b,
					  cuda_matrix* d_c, cuda_matrix* d_h, cuda_matrix* d_y,
					  cuda_matrix d_Why, cuda_matrix* d_probs, cuda_matrix d_sums,
					  cuda_matrix d_by, cuda_matrix d_probs_ones, cuda_matrix* d_neglogprobs,
					  cuda_matrix* d_neglogprobs_out, cuda_matrix* d_targets,
					  Eigen::MatrixXf* g, Eigen::MatrixXf* x, Eigen::MatrixXf* h, Eigen::MatrixXf W, Eigen::MatrixXf U,
					  Eigen::MatrixXf* c, Eigen::VectorXf b, Eigen::MatrixXf* y,
					  Eigen::MatrixXf Why, Eigen::MatrixXf* probs, Eigen::MatrixXf sums,
					  Eigen::VectorXf by, Eigen::MatrixXf* neglogprobs,
					  Eigen::MatrixXf* targets,
					  size_t S, size_t M, size_t N, size_t B);

double eigen_lstm_forward(	Eigen::MatrixXf* g, Eigen::MatrixXf* x, Eigen::MatrixXf* h, Eigen::MatrixXf W, Eigen::MatrixXf U,
							Eigen::MatrixXf* c, Eigen::VectorXf b, Eigen::MatrixXf* y,
							Eigen::MatrixXf Why, Eigen::MatrixXf* probs, Eigen::MatrixXf sums,
							Eigen::VectorXf by, Eigen::MatrixXf* neglogprobs,
							Eigen::MatrixXf* targets,
							const size_t S, const size_t N, const size_t B);

double cuda_lstm_forward(	cuda_matrix* d_g, cuda_matrix* d_x, cuda_matrix d_W, cuda_matrix d_U, cuda_matrix d_b,
							cuda_matrix* d_c, cuda_matrix* d_h, cuda_matrix* d_y,
							cuda_matrix d_Why, cuda_matrix* d_probs, cuda_matrix d_sums,
							cuda_matrix d_by, cuda_matrix d_probs_ones, cuda_matrix* d_neglogprobs,
							cuda_matrix* d_neglogprobs_out, cuda_matrix* d_targets,
							size_t S, size_t N);

#define eps 1e-4
//y = sqrt(x + eps)
inline float sqrt_eps(const float x) {
	return sqrtf(x + eps);
}

//f'(x) = f(x)(1-f(x))
inline float logistic_prime(const float x) {
	return x * (1.0f - x);
}

//f'(x) = 1-(f(x))^2
inline float tanh_prime(const float x) {
	return 1.0f - x * x;
}

//f(x) = sigm(x)
inline float logistic(const float x) {
	return 1.0f / (1.0f +::expf(-x));
}

Eigen::MatrixXi rawread(const char* filename);

// hidden size
const size_t N = 16;
// vocab size (# of distinct observable events)
const size_t M = 256;
// sequence length for learning
const size_t S = 5;
// number of concurrent sequence streams (batch size)
const size_t B = 1;

void run_lstm_test(void) {


	//max epochs
	size_t epochs = 10;

	float learning_rate = 1e-1;

	// read text
	Eigen::MatrixXi data = rawread("enwik4.txt");

	size_t length = data.rows();

	if (length == 0)
		exit(-1);

	// some approximation on the number of FlOPs
	double flops_per_iteration = (S - 1) * (
									 //forward
									 (N * M * B * 2) + (N * 4 * N * B) + (N * 4 * B * 2) + //g[t].array() = (W * x[t] + U * h[t - 1]).array() + b.replicate(1, B).array();
									 (5 * N * 4 * B) + //nolinearities
									 (6 * N * B) + //c(t) + h(t)
									 (M * N * B * 2) + // y[t].array()
									 (8 * N * B) + //probs[t]
									 //backward
									 (N * B) +
									 (M * B * N * 3) +
									 (N * B * 6) +
									 (N * M * B * 4)  + // dh = Why.transpose() * dy[t] + dhnext;
									 (N * B * 8) +
									 (N * 4 * B * M * 3) + // dU += dg * h[t - 1].transpose();
									 (N * 4 * B * N * 3) + // dW += dg * x[t].transpose();
									 (N * 4 * B) +
									 (N * 4 * N * B * 2) + //dhnext = U.transpose() * dg;
									 (N * B) //dcnext.array() = dc.array() * g[t].block<N, B>(2 * N, 0).array();
								 ) +
								 8 * (M * N + M + N * 4 * N + N * 4 * M + N * 4); //adapt

	double flops_per_epoch = flops_per_iteration * (length - S);

	//CPU mem alloc
	Eigen::MatrixXf W(N * 4, M); 									// x -> h matrix (i, o, f, c connections bundled)
	Eigen::MatrixXf U(N * 4, N); 									// x -> h matrix (i, o, f, c connections bundled)
	Eigen::MatrixXf Why(M, N);										// output linear layer
	Eigen::VectorXf b = Eigen::VectorXf::Zero(N * 4); 				// biases to gates
	Eigen::VectorXf by = Eigen::VectorXf::Zero(M);					// biases to outputs

	Eigen::MatrixXf probs_ones = Eigen::MatrixXf::Ones(1, M);
	Eigen::MatrixXf batch_ones = Eigen::MatrixXf::Ones(1, B);

	Eigen::MatrixXf g[S]; 				// gates
	Eigen::MatrixXf x[S];				//inputs
	Eigen::MatrixXf targets[S];			//output targets
	Eigen::MatrixXf h[S];				//hidden states
	Eigen::MatrixXf c[S];				//memory cells
	Eigen::MatrixXf y[S];				//outputs
	Eigen::MatrixXf probs[S];			//normalized outputs, probabilities
	Eigen::MatrixXf dy[S];				//dE/dy
	Eigen::MatrixXf neglogprobs[S];

	for (size_t t = 0; t < S; t++) {

		g[t] = Eigen::MatrixXf::Zero(N * 4, B);
		x[t] = Eigen::MatrixXf::Zero(M, B);
		targets[t] = Eigen::MatrixXf::Zero(M, B);
		h[t] = Eigen::MatrixXf::Zero(N, B);
		c[t] = Eigen::MatrixXf::Zero(N, B);
		y[t] = Eigen::MatrixXf::Zero(M, B);
		probs[t] = Eigen::MatrixXf::Zero(M, B);
		dy[t] = Eigen::MatrixXf::Zero(M, B);
		neglogprobs[t] = Eigen::MatrixXf::Zero(M, B);
	}

	// gradients
	Eigen::MatrixXf dWhy = Eigen::MatrixXf::Zero(M, N);
	Eigen::VectorXf dby = Eigen::VectorXf::Zero(M);
	Eigen::MatrixXf dh = Eigen::MatrixXf::Zero(N, B);
	Eigen::MatrixXf dc = Eigen::MatrixXf::Zero(N, B);
	Eigen::MatrixXf dg = Eigen::MatrixXf::Zero(N * 4, B);
	Eigen::MatrixXf dU = Eigen::MatrixXf::Zero(N * 4, N);
	Eigen::MatrixXf dW = Eigen::MatrixXf::Zero(N * 4, M);
	Eigen::VectorXf db = Eigen::VectorXf::Zero(N * 4);
	Eigen::MatrixXf dhnext = Eigen::MatrixXf::Zero(N, B);
	Eigen::MatrixXf dcnext = Eigen::MatrixXf::Zero(N, B);

	// storage for adagrad update
	Eigen::MatrixXf mWhy = Eigen::MatrixXf::Zero(M, N);
	Eigen::MatrixXf mby = Eigen::VectorXf::Zero(M);
	Eigen::MatrixXf mU = Eigen::MatrixXf::Zero(N * 4, N);
	Eigen::MatrixXf mW = Eigen::MatrixXf::Zero(N * 4, M);
	Eigen::MatrixXf mb = Eigen::VectorXf::Zero(N * 4);

	//temp storage for sums of probs
	Eigen::MatrixXf sums(1, B);

	//encoder
	Eigen::MatrixXf codes = Eigen::MatrixXf::Identity(M, M);

	unsigned int positions[B];

	//GPU mem alloc
	cuda_matrix d_W, d_U, d_b,
				d_g[S], d_x[S], d_h[S], d_targets[S],
				d_c[S], d_y[S], d_dy[S], d_probs[S],
				d_codes, d_Why, d_by, d_sums,
				d_probs_ones, d_neglogprobs[S], d_neglogprobs_out[S],
				d_dWhy, d_dby, d_dh, d_dc, d_dg,
				d_dc_prime, d_dg_prime,
				d_dU, d_dW, d_db, d_dhnext, d_dcnext, d_batch_ones,
				d_mWhy, d_mby, d_mU, d_mW, d_mb,
				d_mWhy_sqrt, d_mby_sqrt, d_mU_sqrt, d_mW_sqrt, d_mb_sqrt;

	unsigned int* d_positions;
	unsigned int* d_data;
	cudaMalloc((void**) & (d_positions), B * sizeof(unsigned int));
	cudaMalloc((void**) & (d_data), length * sizeof(unsigned int));

	// initial positions
	// initial positions in text for every sequence stream

	for (size_t b = 0; b < B; b++) {

		positions[b] = rand() % (length - S);		//rand ints [S, length]
		std::cout << b << ", " << positions[b] << std::endl;

	}

	cublasSetVector(B, sizeof(unsigned int), positions, 1, d_positions, 1);
	cublasSetVector(length, sizeof(unsigned int), data.data(), 1, d_data, 1);


	//cuda_check_matrix_error("positions", d_positions, positions);
	//cuda_check_matrix_error("data", &d_data, data);

	for (size_t t = 0; t < S; t++) {

		cuda_alloc_matrix(&d_g[t], g[t]);
		cuda_alloc_matrix(&d_x[t], x[t]);
		cuda_alloc_matrix(&d_h[t], h[t]);
		cuda_alloc_matrix(&d_c[t], c[t]);
		cuda_alloc_matrix(&d_y[t], y[t]);
		cuda_alloc_matrix(&d_targets[t], targets[t]);
		cuda_alloc_matrix(&d_dy[t], dy[t]);
		cuda_alloc_matrix(&d_probs[t], probs[t]);
		cuda_alloc_matrix(&d_neglogprobs[t], y[t]);
		cuda_alloc_matrix(&d_neglogprobs_out[t], y[t]);

	}

	cuda_alloc_matrix(&d_W, W);
	cuda_alloc_matrix(&d_U, U);
	cuda_alloc_matrix(&d_Why, Why);
	cuda_alloc_matrix(&d_b, b);
	cuda_alloc_matrix(&d_by, by);

	cuda_alloc_matrix(&d_sums, sums);
	cuda_alloc_matrix(&d_probs_ones, probs_ones);
	cuda_alloc_matrix(&d_batch_ones, batch_ones);

	cuda_alloc_matrix(&d_codes, codes);

	//gradients

	cuda_alloc_matrix(&d_dWhy, dWhy);
	cuda_alloc_matrix(&d_dby, dby);
	cuda_alloc_matrix(&d_dh, dh);
	cuda_alloc_matrix(&d_dc, dc);
	cuda_alloc_matrix(&d_dc_prime, dc);
	cuda_alloc_matrix(&d_dg_prime, dg);
	cuda_alloc_matrix(&d_dg, dg);
	cuda_alloc_matrix(&d_dU, dU);
	cuda_alloc_matrix(&d_dW, dW);
	cuda_alloc_matrix(&d_db, db);
	cuda_alloc_matrix(&d_dhnext, dhnext);
	cuda_alloc_matrix(&d_dcnext, dcnext);

	//adadrad
	cuda_alloc_matrix(&d_mWhy, mWhy);
	cuda_alloc_matrix(&d_mby, mby);
	cuda_alloc_matrix(&d_mU, mU);
	cuda_alloc_matrix(&d_mW, mW);
	cuda_alloc_matrix(&d_mb, mb);

	cuda_alloc_matrix(&d_mWhy_sqrt, mWhy);
	cuda_alloc_matrix(&d_mby_sqrt, mby);
	cuda_alloc_matrix(&d_mU_sqrt, mU);
	cuda_alloc_matrix(&d_mW_sqrt, mW);
	cuda_alloc_matrix(&d_mb_sqrt, mb);

	// init matrices
	randn(W, 0, 0.1);
	randn(U, 0, 0.1);
	randn(Why, 0, 0.1);

	cuda_copy_host_to_device(W, &d_W);
	cuda_copy_host_to_device(U, &d_U);
	cuda_copy_host_to_device(Why, &d_Why);
	cuda_copy_host_to_device(b, &d_b);
	cuda_copy_host_to_device(by, &d_by);
	cuda_copy_host_to_device(probs_ones, &d_probs_ones);
	cuda_copy_host_to_device(batch_ones, &d_batch_ones);
	cuda_copy_host_to_device(codes, &d_codes);

	double loss;
	double loss_cuda;
	Timer epoch_timer;
	Timer flops_timer;

	double epoch_loss, epoch_loss_cuda;

	for (size_t e = 0; e < epochs; e++) {

		epoch_loss = 0.0;

		epoch_timer.start();
		flops_timer.start();

		//reset h and c at the beginning of each text
		randn(h[S - 1], 1, 0.0);
		randn(c[S - 1], 1, 0.0);
		cuda_randn_matrix(&d_h[S - 1], 1, 0.0);
		cuda_randn_matrix(&d_c[S - 1], 1, 0.0);

		//go over the text
		for (size_t i = 0; i < length; i++) {

			if (i > 0) {

				//carry over
				for (size_t b = 0; b < B; b++) {

					//h(t-1) = h(t)
					//c(t-1) = c(t)
					for (size_t s = 1; s < S; s++) {
						//eigen
						x[s - 1].col(b) = x[s].col(b);
						h[s - 1].col(b) = h[s].col(b);
						c[s - 1].col(b) = c[s].col(b);
						targets[s - 1].col(b) = targets[s].col(b);

					}

					//x becomes last t
					x[S - 1].col(b) = targets[S - 2].col(b);


				}

				for (size_t s = 1; s < S; s++) {
					//cuda
					cuda_copy_device_to_device(&d_x[s], &d_x[s - 1]);
					cuda_copy_device_to_device(&d_h[s], &d_h[s - 1]);
					cuda_copy_device_to_device(&d_c[s], &d_c[s - 1]);
					cuda_copy_device_to_device(&d_targets[s], &d_targets[s - 1]);
				}

				cuda_copy_device_to_device(&d_targets[S - 2], &d_x[S - 1]);

			}

			//read in next symbols into targets[t] and update position

			//eigen
			for (size_t b = 0; b < B; b++) {

				size_t event = ((int*)data.data())[positions[b]];
				targets[S - 1].col(b) = codes.col(event);
				positions[b]++;
				positions[b] = positions[b] % length;

			}

			//cuda
			cuda_ascii_to_codes(d_data, &d_targets[S - 1], &d_codes, d_positions);
			cuda_advance_positions(d_positions, B, length);

			//cuda_check_matrix_error("d_targets", &d_targets, targets);
			//cuda_check_matrix_error("d_x", &d_x, x);

			loss = 0.0;
			loss_cuda = 0.0;

			std::cout  << std::endl << std::endl;

			gradients numerical_gradients =
				check_grads(d_g, d_x, d_W, d_U, d_b,
							d_c, d_h, d_y,
							d_Why, d_probs, d_sums,
							d_by, d_probs_ones, d_neglogprobs,
							d_neglogprobs_out, d_targets,
							g, x, h, W,  U,
							c, b, y,
							Why, probs, sums,
							by, neglogprobs,
							targets,
							S, M, N, B);


			if (i >= S) { //wait until there is enough context}

				for (size_t t = 1; t < S; t++) { // compute activations for sequence

					// forward
					//1. g[t] = 0; memset
					//2. g[t] += W * x[t]; sgemm
					//3. g += U  * h(t-1)


					//Eigen
					g[t] = (W * x[t] + U * h[t - 1]);


					//CUDA
					cuda_zero_matrix(&d_g[t]);
					cuda_matmul(&d_g[t], &d_W, &d_x[t]);
					cuda_matmul(&d_g[t], &d_U, &d_h[t - 1]);

					//4. g += b;
					g[t].array() = g[t].array().colwise() + b.array();
					cuda_gmav(&d_g[t], &d_g[t], &d_b);

					//5. g(ifo) = sigmoid(g(ifo)) // custom
					g[t].block<3 * N, B>(0, 0).noalias() = g[t].block<3 * N, B>(0, 0).unaryExpr((float (*)(const float))logistic);
					cuda_elementwise_logistic_block(&d_g[t], 0, 3 * N);

					//6. g(c) = tanh(c) // custom
					g[t].block<N, B>(3 * N, 0).noalias() = g[t].block<N, B>(3 * N, 0).unaryExpr(std::ptr_fun(::tanhf));
					cuda_elementwise_tanh_block(&d_g[t], 3 * N, 4 * N);

					//7. c(t) = i(t) * cc(t) + f(t) * c(t-1)
					c[t] = g[t].block<N, B>(0, 0).array() * g[t].block<N, B>(3 * N, 0).array();

					cuda_zero_matrix(&d_c[t]);
					cuda_elementwise_mult_block(&d_c[t], 0, N, &d_g[t], 0, N, &d_g[t], 3 * N, 4 * N);
					cuda_elementwise_mult_block(&d_c[t], 0, N, &d_g[t], 2 * N, 3 * N, &d_c[t - 1], 0, N);
					c[t] = c[t].array() + g[t].block<N, B>(2 * N, 0).array() * c[t - 1].array();

					//8. c(t) = tanh(c(t)) // custom
					c[t].noalias() = c[t].unaryExpr(std::ptr_fun(::tanhf));
					cuda_elementwise_tanh(&d_c[t]);

					//9. h(t) = o(t) .* c(t) //elementwise mul, custom
					h[t] = g[t].block<N, B>(N, 0).array() * c[t].array();
					cuda_zero_matrix(&d_h[t]);
					cuda_elementwise_mult_block(&d_h[t], 0, N, &d_g[t], N, 2 * N, &d_c[t], 0, N);

					//10. y = 0;
					cuda_zero_matrix(&d_y[t]);
					//11. y += Why * h(t) sgemm
					y[t].array() = (Why * h[t]).array().colwise() + by.array();
					cuda_matmul(&d_y[t], &d_Why, &d_h[t]);
					//12. y += by
					cuda_gmav(&d_y[t], &d_y[t], &d_by);

					//13. probs = exp(probs) // custom elementwise
					probs[t] = y[t].unaryExpr(std::ptr_fun(::expf));
					cuda_copy_device_to_device(&d_y[t], &d_probs[t]);
					cuda_elementwise_exp(&d_probs[t]);

					//14. sums = probs[t].colwise().sum();
					sums = probs[t].colwise().sum();
					//http://stackoverflow.com/questions/17862078/reduce-matrix-rows-with-cuda
					// d_sums = ones(1 x M) x y;
					cuda_zero_matrix(&d_sums);
					cuda_matmul(&d_sums, &d_probs_ones, &d_probs[t]);
					//cuda_colwise_sum(&d_sums, &d_probs[t]);

					//15. probs = probs ./ repmat(sums)
					probs[t].noalias() = probs[t].cwiseQuotient( sums.replicate(probs[t].rows(), 1 ));
					cuda_gmdv(&d_probs[t], &d_probs[t], &d_sums);

					//16. surprisals = -log2(probs(:, t)) .* target(:, t);
					neglogprobs[t].array() = (-probs[t].unaryExpr(std::ptr_fun(::logf))).array() * targets[t].array();
					cuda_copy_device_to_device(&d_probs[t], &d_neglogprobs[t]);
					cuda_elementwise_neglog2(&d_neglogprobs[t]);
					cuda_zero_matrix(&d_neglogprobs_out[t]);
					cuda_elementwise_mult(&d_neglogprobs_out[t], &d_targets[t], &d_neglogprobs[t]);

					loss += (double)neglogprobs[t].sum() / (double)B;
					loss_cuda += (double)cuda_matrix_sum(&d_neglogprobs_out[t]) / (double)B;

				}

				// end of forward pass

				// for (size_t t = S - 1; t < S; t++) {

				// 	std::cout << t << std::endl;
				// 	cuda_check_matrix_error("x[t]", &d_x[t], x[t]);
				// 	cuda_check_matrix_error("h[t-1]", &d_h[t - 1], h[t - 1]);
				// 	cuda_check_matrix_error("c[t-1]", &d_c[t - 1], c[t - 1]);
				// 	cuda_check_matrix_error("g[t]", &d_g[t], g[t]);
				// 	cuda_check_matrix_error("c[t]", &d_c[t], c[t]);
				// 	cuda_check_matrix_error("h[t]", &d_h[t], h[t]);
				// 	cuda_check_matrix_error("probs[t]", &d_probs[t], probs[t]);
				// 	cuda_check_matrix_error("y[t]", &d_y[t], y[t]);

				// }

				// cuda_check_matrix_error("b", &d_b, b);
				// cuda_check_matrix_error("U", &d_U, U);
				// cuda_check_matrix_error("W", &d_W, W);

				// cuda_check_matrix_error("by", &d_by, by);
				// cuda_check_matrix_error("sums", &d_sums, sums);


				//std::cout << "end of forward pass, loss eigen: " << loss / S << ", loss cuda: " <<  loss_cuda / S << std::endl;

				epoch_loss += loss / S;
				epoch_loss_cuda += loss_cuda / S;

				//	backward pass:
				//	1. 	reset gradients
				// 	EIGEN
				for (size_t t = 0; t < S; t++) {
					dy[t] = Eigen::MatrixXf::Zero(M, B);
				}

				dWhy = Eigen::MatrixXf::Zero(M, N);
				dby = Eigen::VectorXf::Zero(M);
				dh = Eigen::MatrixXf::Zero(N, B);
				dc = Eigen::MatrixXf::Zero(N, B);
				dg = Eigen::MatrixXf::Zero(N * 4, B);
				dU = Eigen::MatrixXf::Zero(N * 4, N);
				dW = Eigen::MatrixXf::Zero(N * 4, M);
				db = Eigen::VectorXf::Zero(N * 4);
				dhnext = Eigen::MatrixXf::Zero(N, B);
				dcnext = Eigen::MatrixXf::Zero(N, B);

				//CUDA
				for (size_t t = 0; t < S; t++) {
					cuda_zero_matrix(&d_dy[t]);
				}

				cuda_zero_matrix(&d_dWhy);
				cuda_zero_matrix(&d_dby);
				cuda_zero_matrix(&d_dh);
				cuda_zero_matrix(&d_dc);
				cuda_zero_matrix(&d_dc_prime);
				cuda_zero_matrix(&d_dg_prime);
				cuda_zero_matrix(&d_dg);
				cuda_zero_matrix(&d_dU);
				cuda_zero_matrix(&d_dW);
				cuda_zero_matrix(&d_db);
				cuda_zero_matrix(&d_dhnext);
				cuda_zero_matrix(&d_dcnext);

				for (size_t t = S - 1; t > 0; t--) {

					//2. 	dy = probs - target
					dy[t] = probs[t].array() - targets[t].array();
					cuda_elementwise_sub(&d_dy[t], &d_probs[t], &d_targets[t]);

					//3. 	dWhy += dy * h.transpose();
					dWhy.noalias() += dy[t] * h[t].transpose();
					cuda_matmul(&d_dWhy, &d_dy[t], &d_h[t], false, true);

					//4. 	dby += dy.rowwise().sum();
					dby.noalias() += dy[t].rowwise().sum();
					cuda_matmul(&d_dby, &d_dy[t], &d_batch_ones);

					//5. 	dh = Why.transpose() * dy + dhnext;
					dh = Why.transpose() * dy[t] + dhnext;
					cuda_zero_matrix(&d_dh);
					cuda_matmul(&d_dh, &d_Why, &d_dy[t], true, false);
					cuda_elementwise_add(&d_dh, &d_dh, &d_dhnext);

					//6. 	dc = dh.array() * g[t].block<N, B>(N, 0).array() + dcnext.array();
					dc = dh.array() * g[t].block<N, B>(N, 0).array() + dcnext.array();

					cuda_zero_matrix(&d_dc);
					cuda_elementwise_mult_block(&d_dc, 0, N, &d_dh, 0, N, &d_g[t], N, 2 * N);
					cuda_elementwise_add(&d_dc, &d_dc, &d_dcnext);

					//7. 	dc.array() *= c[t].unaryExpr(std::ptr_fun(tanh_prime)).array();
					dc.array() *= c[t].unaryExpr(std::ptr_fun(tanh_prime)).array();
					cuda_copy_device_to_device(&d_c[t], &d_dc_prime);
					cuda_elementwise_tanh_prime(&d_dc_prime);
					cuda_elementwise_mult_noadd(&d_dc, &d_dc, &d_dc_prime);

					cuda_zero_matrix(&d_dg);
					//8. 	dg.block<N, B>(N, 0).array() = dh.array() * c[t].array(); 							// do
					dg.block<N, B>(N, 0).array() = dh.array() * c[t].array();
					cuda_elementwise_mult_block(&d_dg, N, 2 * N, &d_dh, 0, N, &d_c[t], 0, N);

					//9. 	dg.block<N, B>(0, 0).array() = dc.array() * g[t].block<N, B>(3 * N, 0).array(); 	// di
					dg.block<N, B>(0, 0).array() = dc.array() * g[t].block<N, B>(3 * N, 0).array();
					cuda_elementwise_mult_block(&d_dg, 0, N, &d_dc, 0, N, &d_g[t], 3 * N, 4 * N);

					//10. 	dg.block<N, B>(2 * N, 0).array() = dc.array() * c[t - 1].array(); 					// df
					dg.block<N, B>(2 * N, 0).array() = dc.array() * c[t - 1].array();
					cuda_elementwise_mult_block(&d_dg, 2 * N, 3 * N, &d_dc, 0, N, &d_c[t - 1], 0, N);

					//11. 	dg.block<N, B>(3 * N, 0).array() = dc.array() * g[t].block<N, B>(0, 0).array();		// du
					dg.block<N, B>(3 * N, 0).array() = dc.array() * g[t].block<N, B>(0, 0).array();
					cuda_elementwise_mult_block(&d_dg, 3 * N, 4 * N, &d_dc, 0, N, &d_g[t], 0, N);

					//12.	dg.block<3 * N, B>(0, 0).array() = dg.block<3 * N, B>(0, 0).array() * g[t].block<3 * N, B>(0, 0).unaryExpr(std::ptr_fun(logistic_prime)).array();
					dg.block<3 * N, B>(0, 0).array() =
						dg.block<3 * N, B>(0, 0).array() * g[t].block<3 * N, B>(0, 0).unaryExpr(std::ptr_fun(logistic_prime)).array();

					//13.	dg.block<N, B>(3 * N, 0).array() = dg.block<N, B>(3 * N, 0).array() * g[t].block< N, B>(3 * N, 0).unaryExpr(std::ptr_fun(tanh_prime)).array();
					dg.block<N, B>(3 * N, 0).array()
						= dg.block<N, B>(3 * N, 0).array() * g[t].block< N, B>(3 * N, 0).unaryExpr(std::ptr_fun(tanh_prime)).array();

					cuda_copy_device_to_device(&d_g[t], &d_dg_prime);
					cuda_elementwise_tanh_prime_block(&d_dg_prime, 3 * N, 4 * N);
					cuda_elementwise_logistic_prime_block(&d_dg_prime, 0, 3 * N);
					cuda_elementwise_mult_noadd(&d_dg, &d_dg, &d_dg_prime);

					//14.	dU += dg * h[t - 1].transpose();
					//15.	dW += dg * x[t].transpose();
					dU += dg * h[t - 1].transpose();
					dW += dg * x[t].transpose();
					db += dg.rowwise().sum();

					cuda_matmul(&d_dU, &d_dg, &d_h[t - 1], false, true);
					cuda_matmul(&d_dW, &d_dg, &d_x[t], false, true);
					cuda_matmul(&d_db, &d_dg, &d_batch_ones, false, true);
					//cuda_rowwise_sum(&d_db, &d_dg);

					//16.	dhnext = U.transpose() * dg;
					dhnext = U.transpose() * dg;
					cuda_zero_matrix(&d_dhnext);
					cuda_matmul(&d_dhnext, &d_U, &d_dg, true, false);

					//17.	dcnext.array() = dc.array() * g[t].block<N, B>(2 * N, 0).array();
					dcnext.array() = dc.array() * g[t].block<N, B>(2 * N, 0).array();
					cuda_zero_matrix(&d_dcnext);
					cuda_elementwise_mult_block(&d_dcnext, 0, N, &d_dc, 0, N, &d_g[t], 2 * N, 3 * N);

				}

				// backward pass end
				// cuda_check_matrix_error("dy[1]", &d_dy[1], dy[1]);
				// cuda_check_matrix_error("dWhy", &d_dWhy, dWhy);
				// cuda_check_matrix_error("dby", &d_dby, dby);
				// cuda_check_matrix_error("dh", &d_dh, dh);
				// cuda_check_matrix_error("dc", &d_dc, dc);
				// cuda_check_matrix_error("dg", &d_dg, dg);
				// cuda_check_matrix_error("dU", &d_dU, dU);
				// cuda_check_matrix_error("dW", &d_dW, dW);
				// cuda_check_matrix_error("db", &d_db, db);
				// cuda_check_matrix_error("dhnext", &d_dhnext, dhnext);
				// cuda_check_matrix_error("dcnext", &d_dcnext, dcnext);

				cuda_check_matrix_error("numerical dby", &d_dby, numerical_gradients.dby);
				check_matrix_error("numerical eby1", numerical_gradients.eby, dby);
				check_gradient_error("numerical eby2", numerical_gradients.eby, dby);
				cuda_check_matrix_error("numerical dWhy", &d_dWhy, numerical_gradients.dWhy);

				//	adagrad memory
				mWhy.noalias() += dWhy.cwiseProduct(dWhy);
				mby.noalias() += dby.cwiseProduct(dby);
				mU.noalias() += dU.cwiseProduct(dU);
				mW.noalias() += dW.cwiseProduct(dW);
				mb.noalias() += db.cwiseProduct(db);

				cuda_elementwise_mult(&d_mWhy, &d_dWhy, &d_dWhy);
				cuda_elementwise_mult(&d_mby, &d_dby, &d_dby);
				cuda_elementwise_mult(&d_mU, &d_dU, &d_dU);
				cuda_elementwise_mult(&d_mW, &d_dW, &d_dW);
				cuda_elementwise_mult(&d_mb, &d_db, &d_db);

				//	change weights
				//	EIGEN
				Why.noalias() -= learning_rate * dWhy.cwiseQuotient(mWhy.unaryExpr(std::ptr_fun(sqrt_eps)));
				by.noalias() -= learning_rate * dby.cwiseQuotient(mby.unaryExpr(std::ptr_fun(sqrt_eps)));
				U.noalias() -= learning_rate * dU.cwiseQuotient(mU.unaryExpr(std::ptr_fun(sqrt_eps)));
				W.noalias() -= learning_rate * dW.cwiseQuotient(mW.unaryExpr(std::ptr_fun(sqrt_eps)));
				b.noalias() -= learning_rate * db.cwiseQuotient(mb.unaryExpr(std::ptr_fun(sqrt_eps)));

				// //	CUDA
				cuda_copy_device_to_device(&d_mWhy, &d_mWhy_sqrt);
				cuda_copy_device_to_device(&d_mby, &d_mby_sqrt);
				cuda_copy_device_to_device(&d_mU, &d_mU_sqrt);
				cuda_copy_device_to_device(&d_mW, &d_mW_sqrt);
				cuda_copy_device_to_device(&d_mb, &d_mb_sqrt);

				cuda_elementwise_sqrt_eps(&d_mWhy_sqrt);
				cuda_elementwise_sqrt_eps(&d_mby_sqrt);
				cuda_elementwise_sqrt_eps(&d_mU_sqrt);
				cuda_elementwise_sqrt_eps(&d_mW_sqrt);
				cuda_elementwise_sqrt_eps(&d_mb_sqrt);

				//cuda_check_matrix_error("d_mU_sqrt", &d_mU_sqrt, mU.unaryExpr(std::ptr_fun(sqrt_eps)));
				//cuda_check_matrix_error("d_mW_sqrt", &d_mW_sqrt, mW.unaryExpr(std::ptr_fun(sqrt_eps)));

				cuda_elementwise_mul_scalar(&d_dWhy, -learning_rate);
				cuda_elementwise_mul_scalar(&d_dby, -learning_rate);
				cuda_elementwise_mul_scalar(&d_dU, -learning_rate);
				cuda_elementwise_mul_scalar(&d_dW, -learning_rate);
				cuda_elementwise_mul_scalar(&d_db, -learning_rate);

				cuda_elementwise_div(&d_Why, &d_dWhy, &d_mWhy_sqrt);
				cuda_elementwise_div(&d_by, &d_dby, &d_mby_sqrt);
				cuda_elementwise_div(&d_U, &d_dU, &d_mU_sqrt);
				cuda_elementwise_div(&d_W, &d_dW, &d_mW_sqrt);
				cuda_elementwise_div(&d_b, &d_db, &d_mb_sqrt);
				//cuda_check_matrix_error("1/d_mU_sqrt", &d_mU_sqrt, dU.cwiseQuotient(mU.unaryExpr(std::ptr_fun(sqrt_eps))));
				//cuda_check_matrix_error("1/d_mW_sqrt", &d_mW_sqrt, dW.cwiseQuotient(mW.unaryExpr(std::ptr_fun(sqrt_eps))));


				//end of updates
			} //if (i >= S)
		} // for (size_t i = S; i < length; i++)

		double epoch_time = epoch_timer.end();

		std::cout  << std::endl << std::setfill('=') << std::setw(80) << "=" << std::endl <<
				   "Epoch " << e + 1 << "/" << epochs <<
				   std::fixed << std::setprecision(3) <<
				   ", t = " << epoch_time << " s" << ", est GFLOP/s = " <<
				   (flops_per_epoch / powf(2, 30)) / epoch_time <<
				   ", avg loss = " << epoch_loss / (S * length) <<
				   ", avg loss cuda = " << epoch_loss_cuda / (S * length) <<
				   " bits/char" << std::endl;

	} // for (size_t e = 0; e < epochs; e++)
	cuda_free_matrix(&d_W);
	cuda_free_matrix(&d_U);
	cuda_free_matrix(&d_Why);
	cuda_free_matrix(&d_b);
	cuda_free_matrix(&d_by);
	cuda_free_matrix(&d_sums);
	cuda_free_matrix(&d_probs_ones);
	cuda_free_matrix(&d_batch_ones);
	cuda_free_matrix(&d_codes);

	for (size_t t = 0; t < S; t++) {

		cuda_free_matrix(&d_g[t]);
		cuda_free_matrix(&d_x[t]);
		cuda_free_matrix(&d_h[t]);
		cuda_free_matrix(&d_c[t]);
		cuda_free_matrix(&d_y[t]);
		cuda_free_matrix(&d_targets[t]);
		cuda_free_matrix(&d_dy[t]);
		cuda_free_matrix(&d_probs[t]);
		cuda_free_matrix(&d_neglogprobs[t]);
		cuda_free_matrix(&d_neglogprobs_out[t]);
	}

	//gradients
	cuda_free_matrix(&d_dWhy);
	cuda_free_matrix(&d_dby);
	cuda_free_matrix(&d_dh);
	cuda_free_matrix(&d_dc);
	cuda_free_matrix(&d_dc_prime);
	cuda_free_matrix(&d_dg_prime);
	cuda_free_matrix(&d_dg);
	cuda_free_matrix(&d_dU);
	cuda_free_matrix(&d_dW);
	cuda_free_matrix(&d_db);
	cuda_free_matrix(&d_dhnext);
	cuda_free_matrix(&d_dcnext);

	//adagrad
	cuda_free_matrix(&d_mWhy);
	cuda_free_matrix(&d_mby);
	cuda_free_matrix(&d_mU);
	cuda_free_matrix(&d_mW);
	cuda_free_matrix(&d_mb);
	cuda_free_matrix(&d_mWhy_sqrt);
	cuda_free_matrix(&d_mby_sqrt);
	cuda_free_matrix(&d_mU_sqrt);
	cuda_free_matrix(&d_mW_sqrt);
	cuda_free_matrix(&d_mb_sqrt);

	cudaFree(&d_positions);
	cudaFree(&d_data);
}

int main() {

	size_t dev = 0;

	std::cout << "Using CUDA device " << dev << std::endl;

	cudaSetDevice( dev );

	init_curand();
	init_cublas();


	//test_cublas();

	run_lstm_test();

	destroy_cublas();

	return 0;
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

void destroy_cublas( void ) {

	if ( cublasDestroy( handle ) != CUBLAS_STATUS_SUCCESS ) {
		std::cout << "!!!! cuBlas shutdown error" << std::endl;
	}

}

void test_cublas (void) {

	const size_t M = 4096;
	const size_t N = 1024;
	const size_t K = 2048;

	Timer timer;

	std::cout << "Running SGEMM tests..." << std::endl;

	Eigen::MatrixXf A(M, N);
	Eigen::MatrixXf B(N, K);
	Eigen::MatrixXf C(M, K);
	Eigen::MatrixXf gC(M, K);
	Eigen::MatrixXf c_diff(M, K);

	randn(A, 0, 0.5);
	randn(B, 0, 0.5);
	randn(C, 0, 0.5);

	timer.start();
	//Eigen
	C = A * B;

	double eigenTime = timer.end();

	timer.start();
	//cuBLAS
	CUBLAS_mmul(gC, A, B);

	double cudaTime = timer.end();

	c_diff = C - gC;


	std::cout << "max error: " << c_diff.cwiseAbs().maxCoeff() <<
			  ", mean error: " << c_diff.cwiseAbs().sum() / float(M * K) << std::endl;
	std::cout << "Eigen Time = " << eigenTime << "s , CUDA Time = " << cudaTime << " s" << std::endl;
}

void rand(Eigen::MatrixXf& m, float range_min, float range_max) {

	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<> dis(range_min, range_max);

	for (int i = 0; i < m.rows(); i++) {
		for (int j = 0; j < m.cols(); j++) {
			m.coeffRef(i, j) = dis(mt);
		}
	}

}

void randn(Eigen::MatrixXf& m, float mean, float stddev) {

	std::random_device rd;
	std::mt19937 mt(rd());
	std::normal_distribution<> randn(mean, stddev);

	for (int i = 0; i < m.rows(); i++) {
		for (int j = 0; j < m.cols(); j++) {
			m.coeffRef(i, j) = randn(mt);
		}
	}

}

void rand(Eigen::VectorXf& m, float range_min, float range_max) {

	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<> dis(range_min, range_max);

	for (int i = 0; i < m.rows(); i++) {
		for (int j = 0; j < m.cols(); j++) {
			m.coeffRef(i, j) = dis(mt);
		}
	}

}

void randi(Eigen::MatrixXf& m, int range_min, int range_max) {

	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<> dis(range_min, range_max);

	for (int i = 0; i < m.rows(); i++) {
		for (int j = 0; j < m.cols(); j++) {
			m.coeffRef(i, j) = (float)dis(mt);
		}
	}

}


void randi(Eigen::VectorXf& m, int range_min, int range_max) {

	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<> dis(range_min, range_max);

	for (int i = 0; i < m.rows(); i++) {
		for (int j = 0; j < m.cols(); j++) {
			m.coeffRef(i, j) = (float)dis(mt);
		}
	}

}

void randn(Eigen::VectorXf& m, float mean, float stddev) {

	std::random_device rd;
	std::mt19937 mt(rd());
	std::normal_distribution<> randn(mean, stddev);

	for (int i = 0; i < m.rows(); i++) {
		for (int j = 0; j < m.cols(); j++) {
			m.coeffRef(i, j) = randn(mt);
		}
	}

}

//CuBLAS involving matrix copy host-dev-host
void CUBLAS_mmul( Eigen::MatrixXf& __restrict c, Eigen::MatrixXf& __restrict a,
				  Eigen::MatrixXf& __restrict b, bool aT, bool bT ) {

	float* d_A = 0;
	float* d_B = 0;
	float* d_C = 0;

	float alpha = 1.0f;
	float beta = 0.0f;

	size_t M = c.rows();
	size_t N = c.cols();
	size_t K = aT ? a.rows() : a.cols();

	size_t lda = aT ? K : M;
	size_t ldb = bT ? N : K;
	size_t ldc = M;

	const cublasOperation_t transA = aT ? CUBLAS_OP_T : CUBLAS_OP_N;
	const cublasOperation_t transB = bT ? CUBLAS_OP_T : CUBLAS_OP_N;

	cublasStatus_t status;

	// alloc GPU mem
	cudaMalloc((void**)&d_A, a.rows() * a.cols() * sizeof(a.data()[0]));
	cudaMalloc((void**)&d_B, b.rows() * b.cols() * sizeof(b.data()[0]));
	cudaMalloc((void**)&d_C, c.rows() * c.cols() * sizeof(c.data()[0]));

	// copy A and B to GPU mem
	status = cublasSetVector(a.rows() * a.cols(), sizeof(a.data()[0]), a.data(), 1, d_A, 1);
	status = cublasSetVector(b.rows() * b.cols(), sizeof(b.data()[0]), b.data(), 1, d_B, 1);

	if (status != CUBLAS_STATUS_SUCCESS) {
		std::cout << "!!!! cublasSetVector error" << std::endl;
	}

	// C = alpha * A * B + beta * C
	status = cublasSgemmEx(handle,
						   transA, transB,
						   M, N, K,
						   &alpha,
						   d_A, CUBLAS_DATA_TYPE, lda,
						   d_B, CUBLAS_DATA_TYPE, ldb,
						   &beta,
						   d_C, CUBLAS_DATA_TYPE, ldc);

	if (status != CUBLAS_STATUS_SUCCESS) {
		std::cout << "!!!! cublasSgemm error" << std::endl;
	}

	// copy C to CPU mem
	status = cublasGetVector(c.rows() * c.cols(), sizeof(c.data()[0]), d_C, 1, c.data(), 1);

	if (status != CUBLAS_STATUS_SUCCESS) {
		std::cout << "!!!! cublasGetVector" << std::endl;
	}

	//free GPU mem
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

}

Eigen::MatrixXi rawread(const char* filename) {

	Eigen::MatrixXi m(0, 0);

	if (FILE* fp = fopen(filename, "rb")) {

		std::vector<unsigned char> v;
		char buf[1024];

		while (size_t len = fread(buf, 1, sizeof(buf), fp))
			v.insert(v.end(), buf, buf + len);

		fclose(fp);

		if (v.size() > 0) {

			std::cout << "Read " << v.size() << " bytes (" << filename << ")" << std::endl;
			m.resize(v.size(), 1);

			// TODO: probably there is a better way to map std::vector to Eigen::MatrixXi
			for (int i = 0; i < v.size(); i++) {

				((int*)m.data())[i] = (int)v[i];

			}

		} else {

			std::cout << "Empty file! (" << filename << ")" << std::endl;

		}

	} else {

		std::cout << "fopen error: (" << filename << ")" << std::endl;
	}

	return m;
}

//check gradients


gradients check_grads(cuda_matrix* d_g, cuda_matrix* d_x, cuda_matrix d_W, cuda_matrix d_U, cuda_matrix d_b,
					  cuda_matrix* d_c, cuda_matrix* d_h, cuda_matrix* d_y,
					  cuda_matrix d_Why, cuda_matrix* d_probs, cuda_matrix d_sums,
					  cuda_matrix d_by, cuda_matrix d_probs_ones, cuda_matrix* d_neglogprobs,
					  cuda_matrix* d_neglogprobs_out, cuda_matrix* d_targets,
					  Eigen::MatrixXf* g, Eigen::MatrixXf* x, Eigen::MatrixXf* h, Eigen::MatrixXf W, Eigen::MatrixXf U,
					  Eigen::MatrixXf* c, Eigen::VectorXf b, Eigen::MatrixXf* y,
					  Eigen::MatrixXf Why, Eigen::MatrixXf* probs, Eigen::MatrixXf sums,
					  Eigen::VectorXf by, Eigen::MatrixXf* neglogprobs,
					  Eigen::MatrixXf* targets,
					  size_t S, size_t M, size_t N, size_t B) {


	gradients numerical_gradients;

	double epsilon = 1e-4;

	cuda_matrix d_tWhy;
	cuda_matrix d_tby;
	cuda_alloc_matrix(&d_tWhy, M, N);
	cuda_alloc_matrix(&d_tby, M, N);
	numerical_gradients.dWhy = Eigen::MatrixXf::Zero(M, N);
	numerical_gradients.dby = Eigen::VectorXf::Zero(M);
	numerical_gradients.eby = Eigen::VectorXf::Zero(M);
	Eigen::MatrixXf tWhy = Eigen::MatrixXf::Zero(M, N);
	Eigen::VectorXf tby = Eigen::VectorXf::Zero(M);

	double pre, post, pre_eigen, post_eigen;
	double numerical_gradient;

	for (size_t m = 0; m < M; m++) {

		//by
		//eigen
		tby = by;
		tby(m) -= epsilon;
		pre_eigen =  eigen_lstm_forward(g, x, h, W, U,
										c, b, y, Why, probs, sums,
										tby, neglogprobs,
										targets,
										S, N, B);

		tby(m) += 2 * epsilon;

		post_eigen =  eigen_lstm_forward(g, x, h, W, U,
										 c, b, y, Why, probs, sums,
										 tby, neglogprobs,
										 targets,
										 S, N, B);

		numerical_gradient = (post_eigen - pre_eigen) / (2 * epsilon);
		numerical_gradients.eby(m) = numerical_gradient;




		//cuda
		cuda_copy_device_to_host(&d_by, tby);
		tby(m) -= epsilon;
		cuda_copy_host_to_device(tby, &d_tby);

		pre = cuda_lstm_forward(d_g, d_x, d_W, d_U, d_b, d_c, d_h, d_y, d_tWhy, d_probs,
								d_sums, d_by, d_probs_ones, d_neglogprobs, d_neglogprobs_out, d_targets, S,  N);

		tby(m) += 2 * epsilon;
		cuda_copy_host_to_device(tby, &d_tby);
		post = cuda_lstm_forward(d_g, d_x, d_W, d_U, d_b, d_c, d_h, d_y, d_tWhy, d_probs,
								 d_sums, d_by, d_probs_ones, d_neglogprobs, d_neglogprobs_out, d_targets, S,  N);
		numerical_gradient = (post - pre) / (2 * epsilon);
		numerical_gradients.dby(m) = numerical_gradient;


		for (size_t n = 0; n < N; n++) {

			cuda_copy_device_to_host(&d_Why, tWhy);
			//std::cout << tWhy(m, n) << std::endl;
			tWhy(m, n) -= epsilon;
			//std::cout << tWhy(m, n) << std::endl;
			cuda_copy_host_to_device(tWhy, &d_tWhy);

			//cudaThreadSynchronize();
			pre = cuda_lstm_forward(d_g, d_x, d_W, d_U, d_b, d_c, d_h, d_y, d_tWhy, d_probs,
									d_sums, d_by, d_probs_ones, d_neglogprobs, d_neglogprobs_out, d_targets, S,  N);

			tWhy(m, n) +=  2 * epsilon;
			//std::cout << tWhy(m, n) << std::endl;
			cuda_copy_host_to_device(tWhy, &d_tWhy);
			//cudaThreadSynchronize();
			post = cuda_lstm_forward(d_g, d_x, d_W, d_U, d_b, d_c, d_h, d_y, d_tWhy, d_probs,
									 d_sums, d_by, d_probs_ones, d_neglogprobs, d_neglogprobs_out, d_targets, S,  N);


			//std::cout << m << ", " << n << " = " << pre << " vs " << post << " --- " << (post - pre) / (epsilon * 2) << std::endl;
			numerical_gradient = (post - pre) / (2 * epsilon);
			numerical_gradients.dWhy(m, n) = numerical_gradient;

		}
	}

	//std::cout << numerical_gradients.dWhy << std::endl;
	cuda_free_matrix(&d_tWhy);
	cuda_free_matrix(&d_tby);

	return numerical_gradients;
}

double cuda_lstm_forward(	cuda_matrix* d_g, cuda_matrix* d_x, cuda_matrix d_W, cuda_matrix d_U, cuda_matrix d_b,
							cuda_matrix* d_c, cuda_matrix* d_h, cuda_matrix* d_y,
							cuda_matrix d_Why, cuda_matrix* d_probs, cuda_matrix d_sums,
							cuda_matrix d_by, cuda_matrix d_probs_ones, cuda_matrix* d_neglogprobs,
							cuda_matrix* d_neglogprobs_out, cuda_matrix* d_targets,
							size_t S, size_t N) {

	double loss = 0.0;

	for (size_t t = 1; t < S; t++) { // compute activations for sequence

		//CUDA
		cuda_zero_matrix(&d_g[t]);
		cuda_matmul(&d_g[t], &d_W, &d_x[t]);
		cuda_matmul(&d_g[t], &d_U, &d_h[t - 1]);
		cuda_gmav(&d_g[t], &d_g[t], &d_b);
		cuda_elementwise_logistic_block(&d_g[t], 0, 3 * N);
		cuda_elementwise_tanh_block(&d_g[t], 3 * N, 4 * N);
		cuda_zero_matrix(&d_c[t]);
		cuda_elementwise_mult_block(&d_c[t], 0, N, &d_g[t], 0, N, &d_g[t], 3 * N, 4 * N);
		cuda_elementwise_mult_block(&d_c[t], 0, N, &d_g[t], 2 * N, 3 * N, &d_c[t - 1], 0, N);
		cuda_elementwise_tanh(&d_c[t]);
		cuda_zero_matrix(&d_h[t]);
		cuda_elementwise_mult_block(&d_h[t], 0, N, &d_g[t], N, 2 * N, &d_c[t], 0, N);
		cuda_zero_matrix(&d_y[t]);
		cuda_matmul(&d_y[t], &d_Why, &d_h[t]);
		cuda_gmav(&d_y[t], &d_y[t], &d_by);

		cuda_copy_device_to_device(&d_y[t], &d_probs[t]);
		cuda_elementwise_exp(&d_probs[t]);

		cuda_zero_matrix(&d_sums);
		//cuda_matmul(&d_sums, &d_probs_ones, &d_probs[t]);
		cuda_colwise_sum(&d_sums, &d_probs[t]);
		cuda_gmdv(&d_probs[t], &d_probs[t], &d_sums);

		cuda_copy_device_to_device(&d_probs[t], &d_neglogprobs[t]);
		cuda_elementwise_neglog2(&d_neglogprobs[t]);
		cuda_zero_matrix(&d_neglogprobs_out[t]);
		cuda_elementwise_mult(&d_neglogprobs_out[t], &d_targets[t], &d_neglogprobs[t]);

		loss += (double)cuda_matrix_sum(&d_neglogprobs_out[t]);

	}

	return loss;
}

double eigen_lstm_forward(	Eigen::MatrixXf* g, Eigen::MatrixXf* x, Eigen::MatrixXf* h, Eigen::MatrixXf W, Eigen::MatrixXf U,
							Eigen::MatrixXf* c, Eigen::VectorXf b, Eigen::MatrixXf* y,
							Eigen::MatrixXf Why, Eigen::MatrixXf* probs, Eigen::MatrixXf sums,
							Eigen::VectorXf by, Eigen::MatrixXf* neglogprobs,
							Eigen::MatrixXf* targets,
							const size_t _S, const size_t _N, const size_t _B) {

	double loss = 0.0;


	for (size_t t = 1; t < S; t++) { // compute activations for sequence

		g[t] = (W * x[t] + U * h[t - 1]);
		g[t].array() = g[t].array().colwise() + b.array();
		g[t].block<3 * N, B>(0, 0).noalias() = g[t].block<3 * N, B>(0, 0).unaryExpr((float (*)(const float))logistic);
		g[t].block<N, B>(3 * N, 0).noalias() = g[t].block<N, B>(3 * N, 0).unaryExpr(std::ptr_fun(::tanhf));
		c[t] = g[t].block<N, B>(0, 0).array() * g[t].block<N, B>(3 * N, 0).array();
		c[t] = c[t].array() + g[t].block<N, B>(2 * N, 0).array() * c[t - 1].array();
		c[t].noalias() = c[t].unaryExpr(std::ptr_fun(::tanhf));
		h[t] = g[t].block<N, B>(N, 0).array() * c[t].array();
		y[t].array() = (Why * h[t]).array().colwise() + by.array();
		probs[t] = y[t].unaryExpr(std::ptr_fun(::expf));
		sums = probs[t].colwise().sum();
		probs[t].noalias() = probs[t].cwiseQuotient( sums.replicate(probs[t].rows(), 1 ));
		neglogprobs[t].array() = (-probs[t].unaryExpr(std::ptr_fun(::logf))).array() * targets[t].array();

		loss += (double)neglogprobs[t].sum() / (double)B;


	}

	return loss;
}