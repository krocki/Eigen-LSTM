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

		void reset() {

			W = Eigen::MatrixXd::Zero(N * 4, M);
			U = Eigen::MatrixXd::Zero(N * 4, N);
			b = Eigen::MatrixXd::Zero(N * 4, 1);
			Why = Eigen::MatrixXd::Zero(M, N);
			by = Eigen::MatrixXd::Zero(M, 1);

		}

		const size_t M;
		const size_t N;

		Eigen::MatrixXd W; 		// x -> h matrix (i, o, f, c connections bundled)
		Eigen::MatrixXd U; 		// h -> h matrix (i, o, f, c connections bundled)
		Eigen::MatrixXd b; 		// biases to gates
		Eigen::MatrixXd Why; 	// h -> y
		Eigen::MatrixXd by;		// y biases

};

class LSTM {

	public:

		LSTM(size_t _M, size_t _N, size_t _S) : M(_M), N(_N), S(_S) {

			h = Eigen::MatrixXd::Zero(N, S);
			c = Eigen::MatrixXd::Zero(N, S);
			g = Eigen::MatrixXd::Zero(N * 4, S);
			y = Eigen::MatrixXd::Zero(M, S);
			probs = Eigen::MatrixXd::Zero(M, S);

		}

		void forward(const Parameters& p, Eigen::MatrixXd& x) {

			// forward:
			for (size_t t = 1; t < S; t++) { // compute activations for sequence

				//Gates: Linear activations
				g.col(t) = p.W * x.col(t) + p.U * h.col(t - 1) + p.b;

				// nonlinearities - sigmoid on i, o, f gates
				g.block(0, t, 3 * N, 1) =
					g.block(0, t, 3 * N, 1).unaryExpr((double (*)(const double))logistic);

				// nonlinearities - tanh on c gates
				g.block(3 * N, t, N, 1) =
					g.block(3 * N, t, N, 1).unaryExpr(std::ptr_fun(::tanh));

				// new context state, c(t) = i(t) * cc(t) + f(t) * c(t-1)
				c.col(t) = g.block(0, t, N, 1).cwiseProduct(g.block(3 * N, t, N, 1)) +
						   g.block(2 * N, t, N, 1).cwiseProduct(c.col(t - 1));

				// c(t) = tanh(c(t))
				c.col(t) = c.col(t).unaryExpr(std::ptr_fun(::tanh));

				// new hidden state h(t) = o(t) .* c(t)
				h.col(t) = g.block(N, t, N, 1).cwiseProduct(c.col(t));

				// update y = Why * h(t) + by
				y.col(t) = p.Why * h.col(t) + p.by;

				// compute probs: normalize y outputs to sum to 1
				// probs(:, t) = exp(y(:, t)) ./ sum(exp(y(:, t)));
				probs.col(t) = y.col(t).unaryExpr(std::ptr_fun(::exp));
				double sum = probs.col(t).sum();
				probs.col(t) = probs.col(t) / sum;

			}

		}

		double forward_loss(const Parameters& p, Eigen::MatrixXd& x, Eigen::MatrixXd& target) {

			double loss = 0.0;
			forward(p, x);
			Eigen::MatrixXd surprisals(M, 1);

			for (size_t t = 1; t < S; t++) {

				surprisals = (-probs.col(t).unaryExpr(
								  std::ptr_fun(::log))).cwiseProduct(target.col(t));

				loss += surprisals.sum();

			}

			return loss;
		}

		void numerical_grads(Eigen::MatrixXd& n, Eigen::MatrixXd& p, Parameters& P,
							 Eigen::MatrixXd& x, Eigen::MatrixXd& target) {

			double delta = 1e-5;
			double minus_loss, plus_loss;

			for (size_t i = 0; i < n.rows(); i++) {
				for (size_t j = 0; j < n.cols(); j++) {

					double original_value = p(i, j);

					p(i, j) = original_value - delta;
					minus_loss = forward_loss(P, x, target);

					p(i, j) = original_value + delta;
					plus_loss = forward_loss(P, x, target);


					double grad = (plus_loss - minus_loss) / (delta * 2);

					n(i, j) = grad;
					p(i, j) = original_value;
				}
			}

		}

		void compute_all_numerical_grads(Parameters& n, Parameters& p,
										 Eigen::MatrixXd& x, Eigen::MatrixXd& target) {

			n.reset();

			numerical_grads(n.by, p.by, p, x, target);
			numerical_grads(n.Why, p.Why, p, x, target);

			numerical_grads(n.b, p.b, p, x, target);
			numerical_grads(n.U, p.U, p, x, target);
			numerical_grads(n.W, p.W, p, x, target);

		}

		void backward(Parameters& d, Parameters& p, Eigen::MatrixXd& x, Eigen::MatrixXd& target) {

			d.reset();

			// temp storage for gradients
			Eigen::MatrixXd dy = Eigen::MatrixXd::Zero(M, S);
			Eigen::MatrixXd dh = Eigen::MatrixXd::Zero(N, 1);
			Eigen::MatrixXd dc = Eigen::MatrixXd::Zero(N, 1);
			Eigen::MatrixXd dg = Eigen::MatrixXd::Zero(N * 4, 1);
			Eigen::MatrixXd dhnext = Eigen::MatrixXd::Zero(N, 1);
			Eigen::MatrixXd dcnext = Eigen::MatrixXd::Zero(N, 1);

			// backward:
			for (size_t t = S - 1; t > 0; t--) {

				dy.col(t) = probs.col(t) - target.col(t); 	// global error dE/dy
				d.Why += dy.col(t) * h.col(t).transpose();
				d.by += dy.col(t);
				// propagate through linear layer h->y
				dh = p.Why.transpose() * dy.col(t) + dhnext;

				// THE complicated part

				// dc = (dh .* iofc(K*1+1:K*2, t) + dcnext);
				dc = dh.cwiseProduct(g.block(N, t, N, 1)) + dcnext;
				//propagate through tanh
				dc = dc.cwiseProduct(c.col(t).unaryExpr(std::ptr_fun(tanh_prime)));

				//gates
				dg.block(N, 0, N, 1) = dh.cwiseProduct(c.col(t)); 						// do
				dg.block(0, 0, N, 1) = dc.cwiseProduct(g.block(3 * N, t, N, 1)); 		// di
				dg.block(2 * N, 0, N, 1) = dc.cwiseProduct(c.col(t - 1)); 				// df
				dg.block(3 * N, 0, N, 1) = dc.cwiseProduct(g.block(0, t, N, 1));		// du

				// propagate do, di, df though sigmoids
				dg.block(0, 0, 3 * N, 1) =
					dg.block(0, 0, 3 * N, 1).cwiseProduct(
						g.block(0, t, 3 * N, 1).unaryExpr(std::ptr_fun(logistic_prime))
					);

				// propagate u through tanh
				dg.block(3 * N, 0, N, 1) =
					dg.block(3 * N, 0, N, 1).cwiseProduct(
						g.block(3 * N, t, N, 1).unaryExpr(std::ptr_fun(tanh_prime))
					);

				// first linear layers
				d.U += dg * h.col(t - 1).transpose();
				d.W += dg * x.col(t).transpose();
				d.b += dg;

				// IMPORTANT - update gradients for next iteration
				dhnext = p.U.transpose() * dg;
				dcnext = dc.cwiseProduct(g.block(2 * N, t, N, 1));
			}

			//gradients are in d

		}

		LSTM( const LSTM& src) : M(src.M), N(src.N), S(src.S) {

			h = src.h;
			c = src.c;
			g = src.g;
			y = src.y;
			probs = src.probs;

		}

		const size_t M;
		const size_t N;
		const size_t S;

		Eigen::MatrixXd h;							// hidden states
		Eigen::MatrixXd c;							// context states
		Eigen::MatrixXd g;							// gates' states //g = [i o f u]
		Eigen::MatrixXd y;							// outputs
		Eigen::MatrixXd probs;						// output probs - normalized y

};

#endif