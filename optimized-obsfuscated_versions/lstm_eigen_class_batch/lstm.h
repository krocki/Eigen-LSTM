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

#include <Eigen/Dense>
#include <io.h>

#ifdef USE_BLAS
#include <cblas.h>
void BLAS_mmul( Eigen::MatrixXd& c, const Eigen::MatrixXd& a,
				const Eigen::MatrixXd& b, bool aT = false, bool bT = false );
#endif


//f(x) = sigm(x)
inline double logistic(const double x) {
	return 1.0 / (1.0 +::exp(-x));
}

//f'(x) = 1-(f(x))^2
inline double tanh_prime(const double x) {
	return 1.0 - x * x;
}

//f'(x) = f(x)(1-f(x))
inline double logistic_prime(const double x) {
	return x * (1.0 - x);
}

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

			W = Eigen::MatrixXd::Zero(N * 4, M);
			U = Eigen::MatrixXd::Zero(N * 4, N);
			b = Eigen::MatrixXd::Zero(N * 4, 1);
			Why = Eigen::MatrixXd::Zero(M, N);
			by = Eigen::MatrixXd::Zero(M, 1);

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

		Eigen::MatrixXd W; 		// x -> h matrix (i, o, f, c connections bundled)
		Eigen::MatrixXd U; 		// h -> h matrix (i, o, f, c connections bundled)
		Eigen::MatrixXd b; 		// biases to gates
		Eigen::MatrixXd Why; 	// h -> y
		Eigen::MatrixXd by;		// y biases

};

template <size_t S>
class LSTM {

	public:

		LSTM(size_t _M, size_t _N, size_t _B) : M(_M), N(_N), B(_B) {

			for (size_t t = 0; t < S; t++) {

				h[t] = Eigen::MatrixXd::Zero(N, B);
				c[t] = Eigen::MatrixXd::Zero(N, B);
				g[t] = Eigen::MatrixXd::Zero(N * 4, B);
				y[t] = Eigen::MatrixXd::Zero(M, B);
				probs[t] = Eigen::MatrixXd::Zero(M, B);


			}

		}

		void forward(const Parameters& p, Eigen::MatrixXd* x) {

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

				// nonlinearities - tanh on c gates
				g[t].block(3 * N, 0, N, B) =
					g[t].block(3 * N, 0, N, B).unaryExpr(std::ptr_fun(::tanh));

				// new context state, c(t) = i(t) * cc(t) + f(t) * c(t-1)
				c[t] = g[t].block(0, 0, N, B).cwiseProduct(g[t].block(3 * N, 0, N, B)) +
					   g[t].block(2 * N, 0, N, B).cwiseProduct(c[t - 1]);

				// c(t) = tanh(c(t))
				c[t] = c[t].unaryExpr(std::ptr_fun(::tanh));

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

				//for numerical stability
				y[t].array() -= y[t].maxCoeff();

				probs[t] = y[t].unaryExpr(std::ptr_fun(::exp));
				Eigen::MatrixXd sums = probs[t].colwise().sum();
				probs[t] = probs[t].cwiseQuotient( sums.replicate(probs[t].rows(), 1 ));

			}

		}

		double forward_loss(const Parameters& p, Eigen::MatrixXd* x, Eigen::MatrixXd* target) {

			double loss = 0.0;
			forward(p, x);
			Eigen::MatrixXd surprisals(M, B);

			for (size_t t = 1; t < S; t++) {

				surprisals = (-probs[t].unaryExpr(
								  std::ptr_fun(::log))).cwiseProduct(target[t]);

				loss += surprisals.sum();

			}

			return loss;
		}

		void numerical_grads(Eigen::MatrixXd& n, Eigen::MatrixXd& p, Parameters& P,
							 Eigen::MatrixXd* x, Eigen::MatrixXd* target) {

			double delta = 1e-5;
			size_t grads_checked = 0;
			size_t grads_to_check = 100;
			//check only a fraction
			double percentage_to_check = (double)grads_to_check / (double)(p.cols() * p.rows());
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<> dis(0, 1);

			for (size_t i = 0; i < n.rows(); i++) {
				for (size_t j = 0; j < n.cols(); j++) {

					double r = dis(gen);
					if (r >= percentage_to_check)
						continue;

					else {

						double minus_loss, plus_loss;
						double original_value = p(i, j);

						p(i, j) = original_value - delta;
						minus_loss = forward_loss(P, x, target);

						p(i, j) = original_value + delta;
						plus_loss = forward_loss(P, x, target);

						double grad = (plus_loss - minus_loss) / (delta * 2);

						n(i, j) = grad;
						p(i, j) = original_value;

						grads_checked++;

						std::cout << std::setw(9) << i << std::setw(9) << j << std::setw(9) << grads_checked << "\r" <<  std::flush;
					}
				}
			}

		}

		void compute_all_numerical_grads(Parameters& n, Parameters& p,
										 Eigen::MatrixXd* x, Eigen::MatrixXd* target) {

			std::cout << std::endl << std::setw(9) << "by" << std::endl;
			numerical_grads(n.by, p.by, p, x, target);
			std::cout << std::endl << std::endl << std::setw(9) << "Why" << std::endl;
			numerical_grads(n.Why, p.Why, p, x, target);
			std::cout << std::endl << std::endl << std::setw(9) << "b" << std::endl;
			numerical_grads(n.b, p.b, p, x, target);
			std::cout << std::endl << std::endl << std::setw(9) << "U" << std::endl;
			numerical_grads(n.U, p.U, p, x, target);
			std::cout << std::endl << std::endl << std::setw(9) << "W" << std::endl;
			numerical_grads(n.W, p.W, p, x, target);

		}

		void backward(Parameters& d, Parameters& p, Eigen::MatrixXd* x, Eigen::MatrixXd* target) {

			d.reset();

			// temp storage for gradients
			Eigen::MatrixXd dy[S];

			for (size_t t = 0; t < S; t++) {
				dy[t] = Eigen::MatrixXd::Zero(M, B);
			}

			Eigen::MatrixXd dh = Eigen::MatrixXd::Zero(N, B);
			Eigen::MatrixXd dc = Eigen::MatrixXd::Zero(N, B);
			Eigen::MatrixXd dg = Eigen::MatrixXd::Zero(N * 4, B);
			Eigen::MatrixXd dhnext = Eigen::MatrixXd::Zero(N, B);
			Eigen::MatrixXd dcnext = Eigen::MatrixXd::Zero(N, B);

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
			}

		}

		const size_t M;
		const size_t N;
		const size_t B;

		Eigen::MatrixXd h[S];							// hidden states
		Eigen::MatrixXd c[S];							// context states
		Eigen::MatrixXd g[S];							// gates' states //g = [i o f u]
		Eigen::MatrixXd y[S];							// outputs
		Eigen::MatrixXd probs[S];						// output probs - normalized y

};


#ifdef USE_BLAS
// c = a * b
void BLAS_mmul( Eigen::MatrixXd& c, const Eigen::MatrixXd& a,
				const Eigen::MatrixXd& b, bool aT, bool bT ) {

	enum CBLAS_TRANSPOSE transA = aT ? CblasTrans : CblasNoTrans;
	enum CBLAS_TRANSPOSE transB = bT ? CblasTrans : CblasNoTrans;

	size_t M = c.rows();
	size_t N = c.cols();
	size_t K = aT ? a.rows() : a.cols();

	double alpha = 1.0f;
	double beta = 1.0f;

	size_t lda = aT ? K : M;
	size_t ldb = bT ? N : K;
	size_t ldc = M;

	cblas_dgemm( CblasColMajor, transA, transB, M, N, K, alpha,
				 a.data(), lda,
				 b.data(), ldb, beta, c.data(), ldc );


}
#endif /* USE_BLAS */

#endif