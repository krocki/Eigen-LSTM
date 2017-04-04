//
//  cu_lstm.h
//
//  CUDA LSTM code
//
//  Author: Kamil Rocki <kmrocki@us.ibm.com>
//  Created on: 02/19/2016
//

/*process only if we are using NVCC*/

#if defined(__GPU__) || defined(__CUDACC__)

#ifndef _CU_LSTM_H_
#define _CU_LSTM_H_

#include <cu_matrix.h>
#include <Eigen/Dense>

class cuParameters {

	public:

		cuParameters(size_t _M, size_t _N) : M(_M), N(_N) {

			cuda_alloc_matrix(&W, 		N * 4, 		M);
			cuda_alloc_matrix(&U, 		N * 4, 		N);
			cuda_alloc_matrix(&b, 		N * 4, 		1);
			cuda_alloc_matrix(&Why, 	M, 			N);
			cuda_alloc_matrix(&by, 		M, 			1);
			reset();
		}

		~cuParameters() {

			cuda_free_matrix(&W);
			cuda_free_matrix(&U);
			cuda_free_matrix(&b);
			cuda_free_matrix(&Why);
			cuda_free_matrix(&by);

		}

		void reset() {

			// init matrices
			cuda_randn_matrix(&W, 0, 0.01); 								//normal distr
			cuda_randn_matrix(&U, 0, 0.01); 								//normal distr
			cuda_randn_matrix(&Why, 0, 0.01); 								//normal distr

			cuda_zero_matrix(&b);
			cuda_zero_matrix(&by);

		}

		void zero() {

			// init matrices
			cuda_zero_matrix(&W);
			cuda_zero_matrix(&U);
			cuda_zero_matrix(&Why);
			cuda_zero_matrix(&b);
			cuda_zero_matrix(&by);

		}

		const size_t M;
		const size_t N;

		cuda_matrix W; 		// x -> h matrix (i, o, f, c connections bundled)
		cuda_matrix U;		// h -> h matrix (i, o, f, c connections bundled)
		cuda_matrix b; 		// biases to gates
		cuda_matrix Why; 	// h -> y
		cuda_matrix by;		// y biases

};

template <size_t S>
class cuLSTM {

	public:

		cuLSTM(size_t _M, size_t _N, size_t _B) : M(_M), N(_N), B(_B) {

			for (size_t t = 0; t < S; t++) {

				cuda_alloc_matrix(&h[t], 			N, 		B);
				cuda_alloc_matrix(&c[t], 			N, 		B);
				cuda_alloc_matrix(&g[t], 			N * 4, 	B);
				cuda_alloc_matrix(&y[t], 			M, 		B);
				cuda_alloc_matrix(&probs[t], 		M, 		B);
				cuda_alloc_matrix(&neglogprobs[t], 	M, 		B);
				cuda_alloc_matrix(&x[t], 			M, 		B);
				cuda_alloc_matrix(&target[t], 		M, 		B);
				//derivatives temp
				cuda_alloc_matrix(&dy[t], 			M, 		B);

			}

			//derivatives temp
			cuda_alloc_matrix(&dh, 			N, 		B);
			cuda_alloc_matrix(&dc, 			N, 		B);
			cuda_alloc_matrix(&dhnext, 		N, 		B);
			cuda_alloc_matrix(&dcnext, 		N, 		B);
			cuda_alloc_matrix(&dcprime, 	N, 		B);
			cuda_alloc_matrix(&dg, 			N * 4, 	B);
			cuda_alloc_matrix(&dgprime, 	N * 4, 	B);

			cuda_alloc_matrix(&sums, 1, B);
			cuda_alloc_matrix(&M_ones, 1, M);
			cuda_alloc_matrix(&B_ones, 1, B);

			reset();
		}

		~cuLSTM() {

			for (size_t t = 0; t < S; t++) {

				cuda_free_matrix(&h[t]);
				cuda_free_matrix(&c[t]);
				cuda_free_matrix(&g[t]);
				cuda_free_matrix(&y[t]);
				cuda_free_matrix(&probs[t]);
				cuda_free_matrix(&x[t]);
				cuda_free_matrix(&target[t]);

				cuda_free_matrix(&dy[t]);
				cuda_free_matrix(&neglogprobs[t]);

			}

			cuda_free_matrix(&sums);
			cuda_free_matrix(&M_ones);
			cuda_free_matrix(&B_ones);

			cuda_free_matrix(&dh);
			cuda_free_matrix(&dc);
			cuda_free_matrix(&dhnext);
			cuda_free_matrix(&dcnext);
			cuda_free_matrix(&dcprime);
			cuda_free_matrix(&dg);
			cuda_free_matrix(&dgprime);
		}

		void reset() {

			for (size_t t = 0; t < S; t++) {

				cuda_zero_matrix(&h[t]);
				cuda_zero_matrix(&c[t]);
				cuda_zero_matrix(&g[t]);
				cuda_zero_matrix(&y[t]);
				cuda_zero_matrix(&probs[t]);
				cuda_zero_matrix(&neglogprobs[t]);
				cuda_zero_matrix(&x[t]);
				cuda_zero_matrix(&target[t]);
			}

		}

		void forward(const cuParameters& p) {

			// forward:
			for (size_t t = 1; t < S; t++) { // compute activations for sequence

				cuda_zero_matrix(&g[t]);
				cuda_matmul(&g[t], &p.W, &x[t], 0);
				cuda_matmul(&g[t], &p.U, &h[t - 1], 1);
				cuda_matrix_add_vector(&g[t], &g[t], &p.b);
				cuda_elementwise_logistic_block(&g[t], 0, 3 * N);
				cuda_elementwise_tanh_block(&g[t], 3 * N, 4 * N);

				cuda_zero_matrix(&c[t]);
				cuda_elementwise_mult_block(&c[t], 0, N, &g[t], 0, N, &g[t], 3 * N, 4 * N);
				cuda_elementwise_mult_block(&c[t], 0, N, &g[t], 2 * N, 3 * N, &c[t - 1], 0, N);
				cuda_elementwise_tanh(&c[t]);

				cuda_zero_matrix(&h[t]);
				cuda_elementwise_mult_block(&h[t], 0, N, &g[t], 	N, 2 * N, 	&c[t], 		0, 			N);

				cuda_matmul(&y[t], &p.Why, &h[t], 0);
				cuda_matrix_add_vector(&y[t], &y[t], &p.by);
				cuda_copy_device_to_device(&y[t], &probs[t]);

				//TODO:
				//probs[t] -= max(probs[t]);;
				// int idx = cuda_max_matrix(&probs[t]);
				// //sub max(probs[t])
				// cuda_elementwise_sub_element(&probs[t], idx);

				cuda_elementwise_exp(&probs[t]);

				cuda_zero_matrix(&sums);
				cuda_matmul(&sums, &M_ones, &probs[t]);

				cuda_matrix_divide_vector(&probs[t], &probs[t], &sums);

			}

		}

		double calculate_loss() {

			double loss = 0.0;

			for (size_t t = S - 1; t < S; t++) {
				cuda_elementwise_neglog(&neglogprobs[t], &probs[t]);
				cuda_elementwise_mult(&neglogprobs[t], &target[t], &neglogprobs[t], 0);
				if (!std::isnan(loss))
					loss += cuda_matrix_sum(&neglogprobs[t]) / B;
			}

			return loss;
		}
		void backward(cuParameters& d, cuParameters& p) {

			d.zero();

			//reset temp vars
			for (size_t t = 0; t < S; t++) {

				cuda_zero_matrix(&dy[t]);

			}

			cuda_zero_matrix(&dh);
			cuda_zero_matrix(&dhnext);
			cuda_zero_matrix(&dc);
			cuda_zero_matrix(&dcnext);
			cuda_zero_matrix(&dcprime);
			cuda_zero_matrix(&dg);

			for (size_t t = S - 1; t > 0; t--) {

				cuda_elementwise_sub(&dy[t], &probs[t], &target[t]);

				cuda_matmul(&d.Why, &dy[t], &h[t], 1, false, true);
				cuda_matmul(&d.by, &dy[t], &B_ones);

				cuda_matmul(&dh, &p.Why, &dy[t], 0, true, false);
				cuda_elementwise_add(&dh, &dh, &dhnext);

				cuda_zero_matrix(&dc);
				cuda_elementwise_mult_block(&dc, 0, N, &dh, 0, N, &g[t], N, 2 * N);
				cuda_elementwise_add(&dc, &dc, &dcnext);

				cuda_copy_device_to_device(&c[t], &dcprime);
				cuda_elementwise_tanh_prime(&dcprime);
				cuda_elementwise_mult(&dc, &dc, &dcprime, 0);

				cuda_zero_matrix(&dg);
				cuda_elementwise_mult_block(&dg, N, 2 * N, &dh, 0, N, &c[t], 0, N);
				cuda_elementwise_mult_block(&dg, 0, N, &dc, 0, N, &g[t], 3 * N, 4 * N);
				cuda_elementwise_mult_block(&dg, 2 * N, 3 * N, &dc, 0, N, &c[t - 1], 0, N);
				cuda_elementwise_mult_block(&dg, 3 * N, 4 * N, &dc, 0, N, &g[t], 0, N);

				cuda_copy_device_to_device(&g[t], &dgprime);
				cuda_elementwise_tanh_prime_block(&dgprime, 3 * N, 4 * N);
				cuda_elementwise_logistic_prime_block(&dgprime, 0, 3 * N);
				cuda_elementwise_mult(&dg, &dg, &dgprime, 0);

				cuda_matmul(&d.U, &dg, &h[t - 1], 1, false, true);
				cuda_matmul(&d.W, &dg, &x[t], 1, false, true);
				cuda_matmul(&d.b, &dg, &B_ones, 1, false, true);

				cuda_matmul(&dhnext, &p.U, &dg, 0, true, false);

				cuda_zero_matrix(&dcnext);
				cuda_elementwise_mult_block(&dcnext, 0, N, &dc, 0, N, &g[t], 2 * N, 3 * N);


			}

		}

		const size_t M;
		const size_t N;
		const size_t B;

		cuda_matrix h[S];							// hidden states
		cuda_matrix c[S];							// context states
		cuda_matrix g[S];							// gates' states //g = [i o f u]
		cuda_matrix y[S];							// outputs
		cuda_matrix probs[S];						// output probs - normalized y
		cuda_matrix neglogprobs[S];					// output neg log prob
		cuda_matrix x[S];							// outputs
		cuda_matrix target[S];						// output probs - normalized y

		//additional temp vars
		cuda_matrix sums;
		cuda_matrix M_ones;
		cuda_matrix B_ones;

		//for derivatives
		cuda_matrix dy[S];
		cuda_matrix dh;
		cuda_matrix dc;
		cuda_matrix dhnext;
		cuda_matrix dcnext;
		cuda_matrix dcprime;
		cuda_matrix dg;
		cuda_matrix dgprime;
};


void copy_parameters_to_device(Parameters& src, cuParameters& dst) {

	cuda_copy_host_to_device(src.W, &dst.W);
	cuda_copy_host_to_device(src.U, &dst.U);
	cuda_copy_host_to_device(src.b, &dst.b);
	cuda_copy_host_to_device(src.Why, &dst.Why);
	cuda_copy_host_to_device(src.by, &dst.by);

}

void copy_parameters_to_host(cuParameters& src, Parameters& dst) {

	cuda_copy_device_to_host(&src.W, dst.W);
	cuda_copy_device_to_host(&src.U, dst.U);
	cuda_copy_device_to_host(&src.b, dst.b);
	cuda_copy_device_to_host(&src.Why, dst.Why);
	cuda_copy_device_to_host(&src.by, dst.by);

}

void compare_parameters(Parameters& p, cuParameters& cuda_p) {

	compare_matrices("p.W", &cuda_p.W, p.W);
	compare_matrices("p.U", &cuda_p.U, p.U);
	compare_matrices("p.b", &cuda_p.b, p.b);
	compare_matrices("p.Why", &cuda_p.Why, p.Why);
	compare_matrices("p.by", &cuda_p.by, p.by);

}

template <size_t S>
void copy_lstm_to_device(LSTM<S>& lstm, cuLSTM<S>& cuda_lstm) {

	for (size_t t = 0; t < S; t++) {

		cuda_copy_host_to_device(lstm.h[t], &cuda_lstm.h[t]);
		cuda_copy_host_to_device(lstm.c[t], &cuda_lstm.c[t]);
		cuda_copy_host_to_device(lstm.g[t], &cuda_lstm.g[t]);
		cuda_copy_host_to_device(lstm.y[t], &cuda_lstm.y[t]);
		cuda_copy_host_to_device(lstm.probs[t], &cuda_lstm.probs[t]);
		cuda_copy_host_to_device(lstm.x[t], &cuda_lstm.x[t]);
		cuda_copy_host_to_device(lstm.target[t], &cuda_lstm.target[t]);
	}

	cuda_copy_host_to_device(lstm.M_ones, &cuda_lstm.M_ones);
	cuda_copy_host_to_device(lstm.B_ones, &cuda_lstm.B_ones);

}

template <size_t S>
void copy_context_to_host(cuLSTM<S>& cuda_lstm, LSTM<S>& lstm) {

	cuda_copy_device_to_host(&cuda_lstm.h[0], lstm.h[0]);
	cuda_copy_device_to_host(&cuda_lstm.c[0], lstm.c[0]);

}

template <size_t S>
void copy_inputs_to_device(LSTM<S>& lstm, cuLSTM<S>& cuda_lstm) {

	cuda_copy_host_to_device(lstm.h[0], &cuda_lstm.h[0]);
	cuda_copy_host_to_device(lstm.c[0], &cuda_lstm.c[0]);

	for (size_t t = 0; t < S; t++) {


		cuda_copy_host_to_device(lstm.x[t], &cuda_lstm.x[t]);
		cuda_copy_host_to_device(lstm.target[t], &cuda_lstm.target[t]);
	}

}

template <size_t S>
void copy_lstm_to_host(cuLSTM<S>& cuda_lstm, LSTM<S>& lstm) {

	for (size_t t = 0; t < S; t++) {

		cuda_copy_device_to_host(&cuda_lstm.h[t], lstm.h[t]);
		cuda_copy_device_to_host(&cuda_lstm.c[t], lstm.c[t]);
		cuda_copy_device_to_host(&cuda_lstm.g[t], lstm.g[t]);
		cuda_copy_device_to_host(&cuda_lstm.y[t], lstm.y[t]);
		cuda_copy_device_to_host(&cuda_lstm.probs[t], lstm.probs[t]);
		cuda_copy_device_to_host(&cuda_lstm.x[t], lstm.x[t]);
		cuda_copy_device_to_host(&cuda_lstm.target[t], lstm.target[t]);
	}

	cuda_copy_device_to_host(&cuda_lstm.M_ones, lstm.M_ones);
	cuda_copy_device_to_host(&cuda_lstm.B_ones, lstm.B_ones);

}

template <size_t S>
void compare_lstm_states(LSTM<S>& lstm, cuLSTM<S>& cuda_lstm) {

	for (size_t t = 0; t < S; t++) {

		std::cout << "t = " << t << std::endl;
		compare_matrices("h[t]", &cuda_lstm.h[t], lstm.h[t]);
		compare_matrices("c[t]", &cuda_lstm.c[t], lstm.c[t]);
		compare_matrices("g[t]", &cuda_lstm.g[t], lstm.g[t]);
		compare_matrices("y[t]", &cuda_lstm.y[t], lstm.y[t]);
		compare_matrices("probs[t]", &cuda_lstm.probs[t], lstm.probs[t]);
		compare_matrices("x[t]", &cuda_lstm.x[t], lstm.x[t]);
		compare_matrices("target[t]", &cuda_lstm.target[t], lstm.target[t]);
	}

	compare_matrices("M_ones", &cuda_lstm.M_ones, lstm.M_ones);
	compare_matrices("B_ones", &cuda_lstm.B_ones, lstm.B_ones);
}

void cuda_adagrad(cuParameters* p, cuParameters* d, cuParameters* m, double learning_rate) {

	cuda_elementwise_mult(&m->Why, &d->Why, &d->Why, 1);
	cuda_elementwise_mult(&m->by, &d->by, &d->by, 1);
	cuda_elementwise_mult(&m->U, &d->U, &d->U, 1);
	cuda_elementwise_mult(&m->W, &d->W, &d->W, 1);
	cuda_elementwise_mult(&m->b, &d->b, &d->b, 1);

	// change weights:
	cuda_elementwise_adagrad(learning_rate, &p->Why, &d->Why, &m->Why);
	cuda_elementwise_adagrad(learning_rate, &p->by, &d->by, &m->by);
	cuda_elementwise_adagrad(learning_rate, &p->U, &d->U, &m->U);
	cuda_elementwise_adagrad(learning_rate, &p->W, &d->W, &m->W);
	cuda_elementwise_adagrad(learning_rate, &p->b, &d->b, &m->b);

}

#endif

#endif