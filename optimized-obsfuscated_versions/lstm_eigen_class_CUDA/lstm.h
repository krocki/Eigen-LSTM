//
//  lstm.h
//
//  LSTM code
//
//  Author: Kamil Rocki <kmrocki@us.ibm.com>
//  Created on: 02/19/2016
//

#ifndef __LSTM_H_
#define __LSTM_H_

#include <matrix.h>
#include <datatype.h>
#include <fstream>
#include <io.h>

#ifdef USE_BLAS
#include <cblas.h>
void BLAS_mmul( Matrix& c, const Matrix& a,
				const Matrix& b, bool aT = false, bool bT = false );
#endif


//f(x) = sigm(x)
inline dtype logistic(const dtype x) {
	return (dtype)1.0 / ((dtype)1.0 +::exp(-x));
}

//f'(x) = 1-(f(x))^2
inline dtype tanh_prime(const dtype x) {
	return (dtype)1.0 - x * x;
}

//f'(x) = f(x)(1-f(x))
inline dtype logistic_prime(const dtype x) {
	return (dtype)x * (1.0 - x);
}


void save_matrix_to_file(Matrix& m, std::string filename);

class Parameters {

	public:

		Parameters(size_t _M, size_t _N) : M(_M), N(_N) {

			reset();
		}

		Parameters( const Parameters& src) : M(src.M), N(src.N) {

			W = src.W;
			U = src.U;
			b = src.b;
			Why = src.Why;
			by = src.by;

		}

		Parameters& operator=(const Parameters& src) {

			W = src.W;
			U = src.U;
			b = src.b;
			Why = src.Why;
			by = src.by;

			return *this;
		}

		void reset() {

			W = Matrix::Zero(N * 4, M);
			U = Matrix::Zero(N * 4, N);
			b = Matrix::Zero(N * 4, 1);
			Why = Matrix::Zero(M, N);
			by = Matrix::Zero(M, 1);

		}

		void save_to_disk(std::string prefix) {

			save_matrix_to_file(W, prefix + "_W.txt");
			save_matrix_to_file(U, prefix + "_U.txt");
			save_matrix_to_file(Why, prefix + "_Why.txt");
			save_matrix_to_file(b, prefix + "_b.txt");
			save_matrix_to_file(by, prefix + "_by.txt");

		}

		void load_from_disk(std::string prefix) {

			load_matrix_from_file(W, prefix + "_W.txt");
			load_matrix_from_file(U, prefix + "_U.txt");
			load_matrix_from_file(Why, prefix + "_Why.txt");
			load_matrix_from_file(b, prefix + "_b.txt");
			load_matrix_from_file(by, prefix + "_by.txt");

		}

		const size_t M;
		const size_t N;

		Matrix W; 		// x -> h matrix (i, o, f, c connections bundled)
		Matrix U; 		// h -> h matrix (i, o, f, c connections bundled)
		Matrix b; 		// biases to gates
		Matrix Why; 	// h -> y
		Matrix by;		// y biases

};

template <size_t S>
class LSTM {

	public:

		LSTM(size_t _M, size_t _N, size_t _B) : M(_M), N(_N), B(_B) {

			for (size_t t = 0; t < S; t++) {

				h[t] = Matrix::Zero(N, B);
				c[t] = Matrix::Zero(N, B);
				g[t] = Matrix::Zero(N * 4, B);
				y[t] = Matrix::Zero(M, B);
				probs[t] = Matrix::Zero(M, B);
				x[t] = Matrix::Zero(M, B);
				target[t] = Matrix::Zero(M, B);

			}

			M_ones = Matrix::Ones(1, M);
			B_ones = Matrix::Ones(1, B);
		}

		void forward(const Parameters& p) {

			// forward:
			for (size_t t = 1; t < S; t++) { // compute activations for sequence

				//Gates: Linear activations
#ifdef USE_BLAS
				g[t].setZero();
				BLAS_mmul(g[t], p.W, x[t]);
				BLAS_mmul(g[t], p.U, h[t - 1]);
				g[t] = g[t] + p.b.replicate(1, B);
#else
				g[t] = p.W * x[t] + p.U * h[t - 1] + p.b.replicate(1, B);
#endif
				// nonlinearities - sigmoid on i, o, f gates
				g[t].block(0, 0, 3 * N, B) =
					g[t].block(0, 0, 3 * N, B).unaryExpr((double (*)(const double))logistic);

#ifdef PRECISE_MATH
				// nonlinearities - tanh on c gates
				g[t].block(3 * N, 0, N, B) =
					g[t].block(3 * N, 0, N, B).unaryExpr(std::ptr_fun(::tanh));
#else
				g[t].block(3 * N, 0, N, B) =
					g[t].block(3 * N, 0, N, B).unaryExpr(std::ptr_fun(::tanhf));
#endif
				// new context state, c(t) = i(t) * cc(t) + f(t) * c(t-1)
				c[t] = g[t].block(0, 0, N, B).cwiseProduct(g[t].block(3 * N, 0, N, B)) +
					   g[t].block(2 * N, 0, N, B).cwiseProduct(c[t - 1]);

#ifdef PRECISE_MATH
				// c(t) = tanh(c(t))
				c[t] = c[t].unaryExpr(std::ptr_fun(::tanh));
#else
				c[t] = c[t].unaryExpr(std::ptr_fun(::tanhf));
#endif
				// new hidden state h(t) = o(t) .* c(t)
				h[t] = g[t].block(N, 0, N, B).cwiseProduct(c[t]);

				// update y = Why * h(t) + by
#ifdef USE_BLAS
				y[t].setZero();
				BLAS_mmul(y[t], p.Why, h[t]);
				y[t] = y[t] + p.by.replicate(1, B);

#else
				y[t] = p.Why * h[t] + p.by.replicate(1, B);
#endif
				// compute probs: normalize y outputs to sum to 1
				// probs(:, t) = exp(y(:, t)) ./ sum(exp(y(:, t)));
				//TODO:
				//y[t] -= max(y[t]);
#ifdef PRECISE_MATH
				probs[t] = y[t].unaryExpr(std::ptr_fun(::exp));
#else
				probs[t] = y[t].unaryExpr(std::ptr_fun(::expf));
#endif
				Matrix sums = probs[t].colwise().sum();
				probs[t] = probs[t].cwiseQuotient( sums.replicate(probs[t].rows(), 1 ));

			}

		}

		dtype forward_loss(const Parameters& p) {

			dtype loss = 0.0;
			forward(p);
			Matrix surprisals(M, B);

			for (size_t t = S - 1; t < S; t++) {

#ifdef PRECISE_MATH
				surprisals = (-probs[t].unaryExpr(
								  std::ptr_fun(::log))).cwiseProduct(target[t]);
#else
				surprisals = (-probs[t].unaryExpr(
								  std::ptr_fun(::logf))).cwiseProduct(target[t]);
#endif
				loss += (dtype)surprisals.sum() / B;

			}

			return loss;
		}

		void numerical_grads(Matrix& n, Matrix& p, Parameters& P) {

			dtype delta = 1e-5;
			size_t grads_checked = 0;
			size_t grads_to_check = 100;
			//check only a fraction
			dtype percentage_to_check = (dtype)grads_to_check / (dtype)(p.cols() * p.rows());
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<> dis(0, 1);

			for (size_t i = 0; i < n.rows(); i++) {
				for (size_t j = 0; j < n.cols(); j++) {

					dtype r = dis(gen);
					if (r >= percentage_to_check)
						continue;

					else {

						dtype minus_loss, plus_loss;
						dtype original_value = p(i, j);

						p(i, j) = original_value - delta;
						minus_loss = forward_loss(P);

						p(i, j) = original_value + delta;
						plus_loss = forward_loss(P);

						dtype grad = (plus_loss - minus_loss) / (delta * 2);

						n(i, j) = grad;
						p(i, j) = original_value;

						grads_checked++;

						std::cout << std::setw(9) << i << std::setw(9) << j << std::setw(9) << grads_checked << "\r" <<  std::flush;
					}
				}
			}

		}

		void compute_all_numerical_grads(Parameters& n, Parameters& p) {

			std::cout << std::endl << std::setw(9) << "by" << std::endl;
			numerical_grads(n.by, p.by, p);
			std::cout << std::endl << std::endl << std::setw(9) << "Why" << std::endl;
			numerical_grads(n.Why, p.Why, p);
			std::cout << std::endl << std::endl << std::setw(9) << "b" << std::endl;
			numerical_grads(n.b, p.b, p);
			std::cout << std::endl << std::endl << std::setw(9) << "U" << std::endl;
			numerical_grads(n.U, p.U, p);
			std::cout << std::endl << std::endl << std::setw(9) << "W" << std::endl;
			numerical_grads(n.W, p.W, p);

		}

		void backward(Parameters& d, Parameters& p) {

			d.reset();

			// temp storage for gradients
			Matrix dy[S];

			for (size_t t = 0; t < S; t++) {
				dy[t] = Matrix::Zero(M, B);
			}

			Matrix dh = Matrix::Zero(N, B);
			Matrix dc = Matrix::Zero(N, B);
			Matrix dg = Matrix::Zero(N * 4, B);
			Matrix dhnext = Matrix::Zero(N, B);
			Matrix dcnext = Matrix::Zero(N, B);

			// backward:
			for (size_t t = S - 1; t > 0; t--) {

				dy[t] = probs[t] - target[t]; 	// global error dE/dy

#ifdef USE_BLAS
				BLAS_mmul(d.Why, dy[t], h[t], false, true);

#else
				d.Why += dy[t] * h[t].transpose();
#endif
				d.by += dy[t].rowwise().sum();
				// propagate through linear layer h->y
#ifdef USE_BLAS
				dh.setZero();
				BLAS_mmul(dh, p.Why, dy[t], true, false);
				dh += dhnext;

#else
				dh = p.Why.transpose() * dy[t] + dhnext;
#endif
				// THE complicated part

				// dc = (dh .* iofc(K*1+1:K*2, t) + dcnext);
				dc = dh.cwiseProduct(g[t].block(N, 0, N, B)) + dcnext;
				//propagate through tanh
				dc = dc.cwiseProduct(c[t].unaryExpr(std::ptr_fun(tanh_prime)));

				//gates
				dg.block(N, 0, N, B) = dh.cwiseProduct(c[t]); 							// do
				dg.block(0, 0, N, B) = dc.cwiseProduct(g[t].block(3 * N, 0, N, B)); 	// di
				dg.block(2 * N, 0, N, B) = dc.cwiseProduct(c[t - 1]); 					// df
				dg.block(3 * N, 0, N, B) = dc.cwiseProduct(g[t].block(0, 0, N, B));		// du

				// propagate do, di, df though sigmoids
				dg.block(0, 0, 3 * N, B) =
					dg.block(0, 0, 3 * N, B).cwiseProduct(
						g[t].block(0, 0, 3 * N, B).unaryExpr(std::ptr_fun(logistic_prime))
					);

				// propagate u through tanh
				dg.block(3 * N, 0, N, B) =
					dg.block(3 * N, 0, N, B).cwiseProduct(
						g[t].block(3 * N, 0, N, B).unaryExpr(std::ptr_fun(tanh_prime))
					);

				// first linear layers
#ifdef USE_BLAS
				BLAS_mmul(d.U, dg, h[t - 1], false, true);
				BLAS_mmul(d.W, dg, x[t], false, true);
#else
				d.U += dg * h[t - 1].transpose();
				d.W += dg * x[t].transpose();
#endif
				d.b += dg.rowwise().sum();

				// IMPORTANT - update gradients for next iteration
#ifdef USE_BLAS
				dhnext.setZero();
				BLAS_mmul(dhnext, p.U, dg, true, false);
#else
				dhnext = p.U.transpose() * dg;
#endif
				dcnext = dc.cwiseProduct(g[t].block(2 * N, 0, N, B));
			}

			//gradients are in d

		}

		LSTM( const LSTM& src) : M(src.M), N(src.N), B(src.B) {

			for (size_t t = 0; t < S; t++) {
				h[t] = src.h[t];
				c[t] = src.c[t];
				g[t] = src.g[t];
				y[t] = src.y[t];
				probs[t] = src.probs[t];
				x[t] = src.x[t];
				target[t] = src.target[t];
			}

		}

		const size_t M;
		const size_t N;
		const size_t B;

		Matrix h[S];							// hidden states
		Matrix c[S];							// context states
		Matrix g[S];							// gates' states //g = [i o f u]
		Matrix y[S];							// outputs
		Matrix probs[S];						// output probs - normalized y
		Matrix x[S];
		Matrix target[S];

		//temp vars
		Matrix M_ones;
		Matrix B_ones;
};


#ifdef USE_BLAS
// c = a * b
void BLAS_mmul( Matrix& c, const Matrix& a,
				const Matrix& b, bool aT, bool bT ) {

	enum CBLAS_TRANSPOSE transA = aT ? CblasTrans : CblasNoTrans;
	enum CBLAS_TRANSPOSE transB = bT ? CblasTrans : CblasNoTrans;

	size_t M = c.rows();
	size_t N = c.cols();
	size_t K = aT ? a.rows() : a.cols();

	dtype alpha = 1.0f;
	dtype beta = 1.0f;

	size_t lda = aT ? K : M;
	size_t ldb = bT ? N : K;
	size_t ldc = M;

#ifdef PRECISE_MATH
	cblas_dgemm( CblasColMajor, transA, transB, M, N, K, alpha,
				 a.data(), lda,
				 b.data(), ldb, beta, c.data(), ldc );
#else

	cblas_sgemm( CblasColMajor, transA, transB, M, N, K, alpha,
				 a.data(), lda,
				 b.data(), ldb, beta, c.data(), ldc );

#endif

}
#endif /* USE_BLAS */

#endif