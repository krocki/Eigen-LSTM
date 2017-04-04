//
//  lstm.cc
//
//  LSTM code - Eigen implementation
//
//  Author: Kamil Rocki <kmrocki@us.ibm.com>
//  Created on: 02/09/2016
//
// this version is not optimized for speed or shortness, just 'working'
//
// TODO:
//		- make it more parallel
//		- make it more modular
//

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <timer.h>

#include <Eigen/Dense>

#define eps 1e-10

void randn(Eigen::MatrixXf& m, float mean, float stddev);
Eigen::MatrixXi rawread(const char* filename);

//f(x) = sigm(x)
inline float logistic(const float x) {
	return 1.0f / (1.0f +::expf(-x));
}

//f'(x) = 1-(f(x))^2
inline float tanh_prime(const float x) {
	return 1.0f - x * x;
}

//f'(x) = f(x)(1-f(x))
inline float logistic_prime(const float x) {
	return x * (1.0f - x);
}

//y = sqrt(x + eps)
inline float sqrt_eps(const float x) {
	return sqrtf(x + eps);
}

int main() {

	// hidden size
	const size_t N = 64;
	// vocab size (# of distinct observable events)
	const size_t M = 256;
	// sequence length for learning
	const size_t S = 3;

	float learning_rate = 1e-1;
	size_t epochs = 1000;

	// read text
	Eigen::MatrixXi data = rawread("alice29.txt");

	Eigen::MatrixXf W(N * 4, M); 						// x -> h matrix (i, o, f, c connections bundled)
	Eigen::MatrixXf U(N * 4, N); 						// x -> h matrix (i, o, f, c connections bundled)
	Eigen::MatrixXf b(N * 4, 1); 						// biases to gates

	Eigen::MatrixXf Why(M, N);							// h -> y
	Eigen::MatrixXf by(M, 1);							// y biases

	Eigen::MatrixXf codes(N, N); 						// this is an identity matrix that
	// is used to encode inputs, 1 of K encoding

	Eigen::MatrixXf h(N, S);							// hidden states
	Eigen::MatrixXf c(N, S);							// context states
	Eigen::MatrixXf g(N * 4, S);						// gates' states //g = [i o f u]

	Eigen::MatrixXf target(M, S);						// targets - desired outputs
	Eigen::MatrixXf y(M, S);							// outputs
	Eigen::MatrixXf probs(M, S);						// output probs - normalized y
	Eigen::MatrixXf surprisals(M, 1);					// losses: errors = p - target

	Eigen::MatrixXf x(M, S);							// temp matrix for storing input

	// storage for gradients
	Eigen::MatrixXf dy = Eigen::MatrixXf::Zero(M, S);
	Eigen::MatrixXf dWhy = Eigen::MatrixXf::Zero(M, N);
	Eigen::MatrixXf dby = Eigen::MatrixXf::Zero(M, 1);
	Eigen::MatrixXf dh = Eigen::MatrixXf::Zero(N, 1);
	Eigen::MatrixXf dc = Eigen::MatrixXf::Zero(N, 1);
	Eigen::MatrixXf dg = Eigen::MatrixXf::Zero(N * 4, 1);
	Eigen::MatrixXf dU = Eigen::MatrixXf::Zero(N * 4, N);
	Eigen::MatrixXf dW = Eigen::MatrixXf::Zero(N * 4, M);
	Eigen::MatrixXf db = Eigen::MatrixXf::Zero(N * 4, 1);
	Eigen::MatrixXf dhnext = Eigen::MatrixXf::Zero(N, 1);
	Eigen::MatrixXf dcnext = Eigen::MatrixXf::Zero(N, 1);

	// storage for adagrad update
	Eigen::MatrixXf my = Eigen::MatrixXf::Zero(M, S);
	Eigen::MatrixXf mWhy = Eigen::MatrixXf::Zero(M, N);
	Eigen::MatrixXf mby = Eigen::MatrixXf::Zero(M, 1);
	Eigen::MatrixXf mh = Eigen::MatrixXf::Zero(N, S);
	Eigen::MatrixXf mc = Eigen::MatrixXf::Zero(N, S);
	Eigen::MatrixXf mg = Eigen::MatrixXf::Zero(N * 4, S);
	Eigen::MatrixXf mU = Eigen::MatrixXf::Zero(N * 4, N);
	Eigen::MatrixXf mW = Eigen::MatrixXf::Zero(N * 4, M);
	Eigen::MatrixXf mb = Eigen::MatrixXf::Zero(N * 4, 1);
	Eigen::MatrixXf mhnext = Eigen::MatrixXf::Zero(N, 1);
	Eigen::MatrixXf mcnext = Eigen::MatrixXf::Zero(N, 1);

	// init matrices
	randn(W, 0, 0.01); 									//normal distr
	randn(U, 0, 0.01); 									//normal distr
	randn(Why, 0, 0.01); 								//normal distr

	// TODO: can also call .setZero()
	b = Eigen::MatrixXf::Zero(N * 4, 1); 				//zeros
	by = Eigen::MatrixXf::Zero(M, 1); 					//zeros
	h = Eigen::MatrixXf::Zero(N, S); 					//zeros
	c = Eigen::MatrixXf::Zero(N, S); 					//zeros
	g = Eigen::MatrixXf::Zero(N * 4, S); 				//zeros

	target = Eigen::MatrixXf::Zero(M, S);				//zeros
	y = Eigen::MatrixXf::Zero(M, S);					//zeros
	probs = Eigen::MatrixXf::Zero(M, S);				//zeros
	surprisals = Eigen::MatrixXf::Zero(M, 1);			//zeros

	codes = Eigen::MatrixXf::Identity(M, M); 			//Identity matrix (MATLAB's eye())

	Timer t;
	size_t length = data.rows();
	size_t position = 0;
	double loss, epoch_loss;

	/************************************/

	// some lower approximation on the number of flops in an epoch
	// only MMUL flops for now
	double flops_per_epoch = S * (4 * N * M * 2 + 4 * N * N * 2) * length;

	for (size_t e = 0; e < epochs; e++) {

		epoch_loss = 0.0;

		randn(h, 0, 0.1); 									//normal distr
		randn(c, 0, 0.1); 									//normal distr

		t.start();

		for (size_t i = S; i < length; i++) {

			loss = 0;

			size_t event = ((int*)data.data())[i];		// current observation, uchar (0-255)

			for (size_t s = 1; s < S; s++) {

				// shift inputs by 1
				x.col(s - 1) = x.col(s);
				// shift targets by 1
				target.col(s - 1) = target.col(s);
				h.col(s - 1) = h.col(s);
				c.col(s - 1) = c.col(s);

			}

			// column S - 1 hold the most recent events, column 0 - oldest
			target.col(S - 1) = codes.row(event);		// current observation, encoded
			x.col(S - 1) = target.col(S - 2);			// previous observation is current input

			// forward:
			for (size_t t = 1; t < S; t++) { // compute activations for sequence

				//Gates: Linear activations
				g.col(t) = W * x.col(t) + U * h.col(t - 1) + b;

				// nonlinearities - sigmoid on i, o, f gates
				g.block<3 * N, 1>(0, t) = g.block<3 * N, 1>(0, t).unaryExpr((float (*)(const float))logistic);

				// nonlinearities - tanh on c gates
				g.block<N, 1>(3 * N, t) = g.block<N, 1>(3 * N, t).unaryExpr(std::ptr_fun(::tanhf));

				// new context state, c(t) = i(t) * cc(t) + f(t) * c(t-1)
				c.col(t) = g.block<N, 1>(0, t).cwiseProduct(g.block<N, 1>(3 * N, t)) +
						   g.block<N, 1>(2 * N, t).cwiseProduct(c.col(t - 1));

				// c(t) = tanh(c(t))
				c.col(t) = c.col(t).unaryExpr(std::ptr_fun(::tanhf));

				// new hidden state h(t) = o(t) .* c(t)
				h.col(t) = g.block<N, 1>(N, t).cwiseProduct(c.col(t));

				// update y = Why * h(t) + by
				y.col(t) = Why * h.col(t) + by;

				// compute probs: normalize y outputs to sum to 1
				// probs(:, t) = exp(y(:, t)) ./ sum(exp(y(:, t)));
				probs.col(t) = y.col(t).unaryExpr(std::ptr_fun(::expf));
				float sum = probs.col(t).sum();
				probs.col(t) = probs.col(t) / sum;

				// -log2(probs(:, t)) .* target(:, t);
				surprisals = (-probs.col(t).unaryExpr(std::ptr_fun(::log2f))).cwiseProduct(target.col(t));

				// cross-entropy loss, sum logs of probabilities of target outputs
				loss += surprisals.sum();

			}

			epoch_loss += loss;

			// reset gradients
			dWhy.setZero();
			dby.setZero();
			dhnext.setZero();
			dcnext.setZero();
			dU.setZero();
			dW.setZero();
			db.setZero();

			// backward:
			for (size_t t = S - 1; t > 0; t--) {

				dy.col(t) = probs.col(t) - target.col(t); 	// global error dE/dy
				dWhy += dy.col(t) * h.col(t).transpose();
				dby += dy.col(t);
				dh = Why.transpose() * dy.col(t) + dhnext;	// propagate through linear layer h->y

				// THE complicated part

				// dc = (dh .* iofc(K*1+1:K*2, t) + dcnext);
				dc = dh.cwiseProduct(g.block<N, 1>(N, t)) + dcnext;
				//propagate through tanh
				dc = dc.cwiseProduct(c.col(t).unaryExpr(std::ptr_fun(tanh_prime)));

				//gates
				dg.block<N, 1>(N, 0) = dh.cwiseProduct(c.col(t)); 						// do
				dg.block<N, 1>(0, 0) = dc.cwiseProduct(g.block<N, 1>(3 * N, t)); 		// di
				dg.block<N, 1>(2 * N, 0) = dc.cwiseProduct(c.col(t - 1)); 				// df
				dg.block<N, 1>(3 * N, 0) = dc.cwiseProduct(g.block<N, 1>(0, t));		// du

				// propagate do, di, df though sigmoids
				dg.block<3 * N, 1>(0, 0) = dg.block<3 * N, 1>(0, 0).cwiseProduct(g.block<3 * N, 1>(0, t).unaryExpr(std::ptr_fun(logistic_prime)));

				// propagate u through tanh
				dg.block<N, 1>(3 * N, 0) = dg.block<N, 1>(3 * N, 0).cwiseProduct(g.block< N, 1>(3 * N, t).unaryExpr(std::ptr_fun(tanh_prime)));

				// first linear layers
				dU += dg * h.col(t - 1).transpose();
				dW += dg * x.col(t).transpose();
				db += dg;

				// IMPORTANT - update gradients for next iteration
				dhnext = U.transpose() * dg;
				dcnext = dc.cwiseProduct(g.block<N, 1>(2 * N, t));
			}

			// adagrad memory:

			mWhy += dWhy.cwiseProduct(dWhy);
			mby += dby.cwiseProduct(dby);
			mU += dU.cwiseProduct(dU);
			mW += dW.cwiseProduct(dW);
			mb += db.cwiseProduct(db);

			// change weights:
			Why -= learning_rate * dWhy.cwiseQuotient(mWhy.unaryExpr(std::ptr_fun(sqrt_eps)));
			by -= learning_rate * dby.cwiseQuotient(mby.unaryExpr(std::ptr_fun(sqrt_eps)));
			U -= learning_rate * dU.cwiseQuotient(mU.unaryExpr(std::ptr_fun(sqrt_eps)));
			W -= learning_rate * dW.cwiseQuotient(mW.unaryExpr(std::ptr_fun(sqrt_eps)));
			b -= learning_rate * db.cwiseQuotient(mb.unaryExpr(std::ptr_fun(sqrt_eps)));

			if ((i % 100) == 0) {

				std::cout << std::fixed <<
						  std::setw(7) << std::setprecision(2) <<
						  100.0f * (float)i / (float)length << "%\r" << std::flush;
			}
		}

		double epoch_time = t.end();

		std::cout  << std::endl <<
				   "====================================================================================" << std::endl <<
				   "Epoch " << e + 1 << "/" << epochs <<
				   std::fixed << std::setprecision(3) <<
				   ", t = " << epoch_time << " s" << ", est GFLOP/s = " <<
				   (flops_per_epoch / powf(2, 30)) / epoch_time <<
				   ", avg loss = " << epoch_loss / (S * length) <<
				   " bits/char" << std::endl;

		std::vector<char> generated_text;

		size_t characters_to_generate = 1000;

		//temp vars
		Eigen::MatrixXf _h(N, 1);
		Eigen::MatrixXf _c(N, 1);
		Eigen::MatrixXf _x(M, 1);
		Eigen::MatrixXf _y(M, 1);
		Eigen::MatrixXf _probs(M, 1);
		Eigen::MatrixXf cdf = Eigen::MatrixXf::Zero(M, 1);
		Eigen::MatrixXf _g(N * 4, 1);

		randn(_h, 0, 0.1);
		randn(_c, 0, 0.1);

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(0, 1);

		for (size_t i = 0; i < characters_to_generate; i++) {

			_y = Why * _h + by;
			_probs = _y.unaryExpr(std::ptr_fun(::expf));
			float sum = _probs.sum();
			_probs = _probs / sum;

			//cumsum, TODO: something nicer
			cdf(0, 0) = _probs(0, 0);
			for (size_t ii = 1; ii < _probs.rows(); ii++) {
				cdf(ii, 0) = cdf(ii - 1, 0) + _probs(ii, 0);
			}

			float r = dis(gen);

			// find the lowest number in cdf that's larger or equal to r
			size_t index = 0;
			for (size_t ii = 0; ii < cdf.rows(); ii++) {

				if (r < cdf(ii, 0)) {

					index = ii;
					break;
				}

			}

			generated_text.push_back(char(index));

			_x.col(0) = codes.row(index);
			_g = W * _x + U * _h + b;
			_g.block<3 * N, 1>(0, 0) = _g.block<3 * N, 1>(0, 0).unaryExpr((float (*)(const float))logistic);
			_g.block<N, 1>(3 * N, 0) = _g.block<N, 1>(3 * N, 0).unaryExpr(std::ptr_fun(::tanhf));
			_c = _g.block<N, 1>(0, 0).cwiseProduct(_g.block<N, 1>(3 * N, 0)) + _g.block<N, 1>(2 * N, 0).cwiseProduct(_c);
			_c = _c.unaryExpr(std::ptr_fun(::tanhf));
			_h = _g.block<N, 1>(N, 0).cwiseProduct(_c);

		}

		std::cout << std::endl << std::endl << "************ Generated text |";
		for (std::vector<char>::const_iterator i = generated_text.begin(); i != generated_text.end(); ++i)
			std::cout << *i;

		std::cout << "| Generated text END ************" << std::endl;
	}

	return 0;

}


void randn(Eigen::MatrixXf& m, float mean, float stddev) {

	// random number generator
	// unfortunately, Eigen does not implement normal distribution
	// TODO: make it cleaner, more parallel

	std::random_device rd;
	std::mt19937 mt(rd());
	std::normal_distribution<> randn(mean, stddev);

	for (int i = 0; i < m.rows(); i++) {
		for (int j = 0; j < m.cols(); j++) {
			m.coeffRef(i, j) = randn(mt);
		}
	}

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