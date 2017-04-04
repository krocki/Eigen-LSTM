//
//  lstm.cc
//
//  LSTM code - Eigen/BLAS implementation
//
//	+ multiple sequence streams
//	+ some optimizations
//	+ BLAS support
//
//  Author: Kamil Rocki <kmrocki@us.ibm.com>
//  Created on: 02/12/2016
//

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <timer.h>

#include <Eigen/Dense>

#ifdef USE_BLAS
#include <cblas.h>
void BLAS_mmul( Eigen::MatrixXf& __restrict c, Eigen::MatrixXf& __restrict a,
				Eigen::MatrixXf& __restrict b, bool aT = false, bool bT = false );
#endif

#define eps 1e-10

void randn(Eigen::MatrixXf& m, float mean, float stddev);
Eigen::MatrixXi rawread(const char* filename);
void save_to_file(Eigen::MatrixXf& m, const char* filename);

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
	// number of concurrent sequence streams (batch size)
	const size_t B = 4;

	float learning_rate = 1e-1;
	size_t epochs = 1000;

	// read text
	Eigen::MatrixXi data = rawread("enwik5.txt");

	Eigen::MatrixXf W(N * 4, M); 						// x -> h matrix (i, o, f, c connections bundled)
	Eigen::MatrixXf U(N * 4, N); 						// x -> h matrix (i, o, f, c connections bundled)
	Eigen::VectorXf b(N * 4); 							// biases to gates

	Eigen::MatrixXf Why(M, N);							// h -> y
	Eigen::VectorXf by(M);								// y biases

	Eigen::MatrixXf codes(N, N); 						// this is an identity matrix that
	// is used to encode inputs, 1 of K encoding

	Eigen::MatrixXf h[S];							// hidden states
	Eigen::MatrixXf c[S];							// context states
	Eigen::MatrixXf g[S];							// gates' states //g = [i o f u]
	Eigen::MatrixXf target[S];						// targets - desired outputs
	Eigen::MatrixXf y[S];							// outputs
	Eigen::MatrixXf probs[S];						// output probs - normalized y
	Eigen::MatrixXf x[S];							// temp matrix for storing input

	Eigen::MatrixXf surprisals;						// losses: errors = p - target

	// storage for gradients
	Eigen::MatrixXf dy[S];
	Eigen::MatrixXf dWhy = Eigen::MatrixXf::Zero(M, N);
	Eigen::MatrixXf dby = Eigen::VectorXf::Zero(M);
	Eigen::MatrixXf dh = Eigen::MatrixXf::Zero(N, B);
	Eigen::MatrixXf dc = Eigen::MatrixXf::Zero(N, B);
	Eigen::MatrixXf dg = Eigen::MatrixXf::Zero(N * 4, B);
	Eigen::MatrixXf dU = Eigen::MatrixXf::Zero(N * 4, N);
	Eigen::MatrixXf dW = Eigen::MatrixXf::Zero(N * 4, M);
	Eigen::MatrixXf db = Eigen::VectorXf::Zero(N * 4);
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

	// init matrices
	randn(W, 0, 0.01); 									//normal distr
	randn(U, 0, 0.01); 									//normal distr
	randn(Why, 0, 0.01); 								//normal distr

	// TODO: can also call .setZero()
	b = Eigen::VectorXf::Zero(N * 4); 					//zeros
	by = Eigen::VectorXf::Zero(M); 						//zeros

	size_t positions[B];								// positions in text

	for (size_t t = 0; t < S; t++) {
		h[t] = Eigen::MatrixXf::Zero(N, B); 				//zeros
		c[t] = Eigen::MatrixXf::Zero(N, B); 				//zeros
		g[t] = Eigen::MatrixXf::Zero(N * 4, B); 			//zeros
		target[t] = Eigen::MatrixXf::Zero(M, B);			//zeros
		y[t] = Eigen::MatrixXf::Zero(M, B);					//zeros
		probs[t] = Eigen::MatrixXf::Zero(M, B);				//zeros
		x[t] = Eigen::MatrixXf::Zero(M, B);
		dy[t] = Eigen::MatrixXf::Zero(M, B);
	}

	surprisals = Eigen::MatrixXf::Zero(M, B);			//zeros

	codes = Eigen::MatrixXf::Identity(M, M); 			//Identity matrix (MATLAB's eye())

	Timer epoch_timer;
	Timer flops_timer;
	size_t length = data.rows();

	double loss, epoch_loss;

	// initial positions in text for every sequence stream
	for (size_t b = 0; b < B; b++) {

		positions[b] = rand() % (length - S) + S;		//rand ints [S, length]

	}

	/************************************/

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

	for (size_t e = 0; e < epochs; e++) {

		epoch_loss = 0.0;

		for (size_t t = 0; t < S; t++) {
			randn(h[t], 0, 0.1); 									//normal distr
			randn(c[t], 0, 0.1); 									//normal distr
			// h[t] = Eigen::MatrixXf::Zero(N, B);
			// c[t] = Eigen::MatrixXf::Zero(N, B);
		}

		epoch_timer.start();
		flops_timer.start();

		for (size_t i = S; i < length; i++) {

			loss = 0;

			for (size_t b = 0; b < B; b++) {

				size_t event = ((int*)data.data())[positions[b]];		// current observation, uchar (0-255)

				positions[b]++;

				if (positions[b] >= length)
					positions[b] = S;

				for (size_t s = 1; s < S; s++) {

					// shift inputs by 1
					x[s - 1].col(b) = x[s].col(b);
					// shift targets by 1
					target[s - 1].col(b) = target[s].col(b);
					h[s - 1].col(b) = h[s].col(b);
					c[s - 1].col(b) = c[s].col(b);

				}

				// column S - 1 hold the most recent events, column 0 - oldest
				target[S - 1].col(b) = codes.col(event);			// current observation, encoded
				x[S - 1].col(b) = target[S - 2].col(b);				// previous observation is current input
			}

			// forward:
			for (size_t t = 1; t < S; t++) { // compute activations for sequence

				//Gates: Linear activations
#ifdef USE_BLAS
				//reset, since we are doing BLAS_mmul does g[t] += W * x[t]
				g[t].setZero();
				BLAS_mmul(g[t], W, x[t]);
				BLAS_mmul(g[t], U, h[t - 1]);
				g[t].array() = g[t].array().colwise() + b.array();
#else
				g[t].array() = (W * x[t] + U * h[t - 1]).array().colwise() + b.array(); // g(t) = (W * x(t) + U * h(t-1) + b)
#endif
				// nonlinearities - sigmoid on i, o, f gates
				g[t].block<3 * N, B>(0, 0).noalias() = g[t].block<3 * N, B>(0, 0).unaryExpr((float (*)(const float))logistic);
				// nonlinearities - tanh on c gates
				g[t].block<N, B>(3 * N, 0).noalias() = g[t].block<N, B>(3 * N, 0).unaryExpr(std::ptr_fun(::tanhf));
				// new context state, c(t) = i(t) * cc(t) + f(t) * c(t-1)
				c[t] = g[t].block<N, B>(0, 0).array() * g[t].block<N, B>(3 * N, 0).array() +
					   g[t].block<N, B>(2 * N, 0).array() * c[t - 1].array();

				// c(t) = tanh(c(t))
				c[t].noalias() = c[t].unaryExpr(std::ptr_fun(::tanhf));

				// new hidden state h(t) = o(t) .* c(t)
				h[t] = g[t].block<N, B>(N, 0).array() * c[t].array();

				// update y = Why * h(t) + by
#ifdef USE_BLAS
				y[t].setZero();
				BLAS_mmul(y[t], Why, h[t]);
				y[t].array() = y[t].array().colwise() + by.array();
#else
				y[t].array() = (Why * h[t]).array().colwise() + by.array();
#endif
				// compute probs: normalize y outputs to sum to 1
				// probs(:, t) = exp(y(:, t)) ./ sum(exp(y(:, t)));
				probs[t] = y[t].unaryExpr(std::ptr_fun(::expf));
				sums = probs[t].colwise().sum();
				probs[t].noalias() = probs[t].cwiseQuotient( sums.replicate(probs[t].rows(), 1 ));


				// -log2(probs(:, t)) .* target(:, t);
				surprisals = (-probs[t].unaryExpr(std::ptr_fun(::log2f))).array() * target[t].array();

				// cross-entropy loss, sum logs of probabilities of target outputs
				loss += surprisals.sum() / (float)B;

			}

			epoch_loss += loss;

			// reset gradients
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

			// backward:
			for (size_t t = S - 1; t > 0; t--) {

				dy[t] = probs[t].array() - target[t].array(); 	// global error dE/dy
#ifdef USE_BLAS
				BLAS_mmul(dWhy, dy[t], h[t], false, true);
#else
				dWhy.noalias() += dy[t] * h[t].transpose();
#endif
				dby.noalias() += dy[t].rowwise().sum();
#ifdef USE_BLAS
				dh.setZero();
				BLAS_mmul(dh, Why, dy[t], true, false);
				dh += dhnext;

#else
				dh = Why.transpose() * dy[t] + dhnext;	// propagate through linear layer h->y
#endif
				// THE complicated part

				// dc = (dh .* iofc(K * 1 + 1: K * 2, t) + dcnext);
				dc = dh.array() * g[t].block<N, B>(N, 0).array() + dcnext.array();
				//propagate through tanh
				dc.array() *= c[t].unaryExpr(std::ptr_fun(tanh_prime)).array();

				//gates
				dg.block<N, B>(N, 0).array() = dh.array() * c[t].array(); 							// do
				dg.block<N, B>(0, 0).array() = dc.array() * g[t].block<N, B>(3 * N, 0).array(); 	// di
				dg.block<N, B>(2 * N, 0).array() = dc.array() * c[t - 1].array(); 					// df
				dg.block<N, B>(3 * N, 0).array() = dc.array() * g[t].block<N, B>(0, 0).array();		// du

				// propagate do, di, df though sigmoids
				dg.block<3 * N, B>(0, 0).array() =
					dg.block<3 * N, B>(0, 0).array() * g[t].block<3 * N, B>(0, 0).unaryExpr(std::ptr_fun(logistic_prime)).array();

				// propagate u through tanh
				dg.block<N, B>(3 * N, 0).array()
					= dg.block<N, B>(3 * N, 0).array() * g[t].block< N, B>(3 * N, 0).unaryExpr(std::ptr_fun(tanh_prime)).array();

#ifdef USE_BLAS
				BLAS_mmul(dU, dg, h[t - 1], false, true);
				BLAS_mmul(dW, dg, x[t], false, true);
#else
				// first linear layers
				dU += dg * h[t - 1].transpose();
				dW += dg * x[t].transpose();
#endif
				db += dg.rowwise().sum();

				// IMPORTANT - update gradients for next iteration
#ifdef USE_BLAS
				dhnext.setZero();
				BLAS_mmul(dhnext, U, dg, true, false);
#else
				dhnext = U.transpose() * dg;
#endif
				dcnext.array() = dc.array() * g[t].block<N, B>(2 * N, 0).array();
			}

			// adagrad
			mWhy.noalias() += dWhy.cwiseProduct(dWhy);
			mby.noalias() += dby.cwiseProduct(dby);
			mU.noalias() += dU.cwiseProduct(dU);
			mW.noalias() += dW.cwiseProduct(dW);
			mb.noalias() += db.cwiseProduct(db);

			// change weights:
			Why.noalias() -= learning_rate * dWhy.cwiseQuotient(mWhy.unaryExpr(std::ptr_fun(sqrt_eps)));
			by.noalias() -= learning_rate * dby.cwiseQuotient(mby.unaryExpr(std::ptr_fun(sqrt_eps)));
			U.noalias() -= learning_rate * dU.cwiseQuotient(mU.unaryExpr(std::ptr_fun(sqrt_eps)));
			W.noalias() -= learning_rate * dW.cwiseQuotient(mW.unaryExpr(std::ptr_fun(sqrt_eps)));
			b.noalias() -= learning_rate * db.cwiseQuotient(mb.unaryExpr(std::ptr_fun(sqrt_eps)));

			if ((i % 100) == 0) {

				double flops_time = flops_timer.end();
				size_t eta_sec =  (flops_time * ((float)length - (float)i)) / 100.0f;
				size_t eta_hours = eta_sec / 3600;
				size_t eta_min = (eta_sec % 3600) / 60;
				eta_sec = eta_sec % 60;

				std::cout << std::fixed <<
						  std::setw(9) << std::setprecision(2) <<
						  100.0f * (float)i / (float)length << "%   (eta " <<
						  std::setw(2) << eta_hours << " h " << std::setfill('0') <<
						  std::setw(2) << eta_min << " m " << std::setfill('0') <<
						  std::setw(2) << eta_sec << " s)" << std::setfill(' ') <<
						  std::setw(9) << std::setprecision(2) <<
						  (100.0f * flops_per_iteration / powf(2, 30)) / flops_time <<
						  " GFlOP/s\r" << std::flush;

				flops_timer.start();
			}
		}

		double epoch_time = epoch_timer.end();

		std::cout  << std::endl << std::setfill('=') << std::setw(80) << "=" << std::endl <<
				   "Epoch " << e + 1 << "/" << epochs <<
				   std::fixed << std::setprecision(3) <<
				   ", t = " << epoch_time << " s" << ", est GFLOP/s = " <<
				   (flops_per_epoch / powf(2, 30)) / epoch_time <<
				   ", avg loss = " << epoch_loss / (S * length) <<
				   " bits/char" << std::endl;

		std::vector<char> generated_text;

		size_t characters_to_generate = 2000;

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

		//TODO: make this better
		// save_to_file(W, "W.txt");
		// save_to_file(U, "U.txt");
		// Eigen::MatrixXf _b = b.cast <float> ();
		// save_to_file(_b, "b.txt");
		// save_to_file(Why, "Why.txt");
		// Eigen::MatrixXf _by = by.cast <float> ();
		// save_to_file(_by, "by.txt");

	}

	return 0;

}

void save_to_file(Eigen::MatrixXf& m, const char* filename) {

	std::cout << "Saving a matrix to " << filename << "... " << std::endl;
	std::ofstream file(filename);

	if (file.is_open()) {

		file << m;
		file.close();

	} else {

		std::cout << "file save error: (" << filename << ")" << std::endl;

	}
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

#ifdef USE_BLAS
// c = a * b
void BLAS_mmul( Eigen::MatrixXf& __restrict c, Eigen::MatrixXf& __restrict a,
				Eigen::MatrixXf& __restrict b, bool aT, bool bT ) {

	enum CBLAS_TRANSPOSE transA = aT ? CblasTrans : CblasNoTrans;
	enum CBLAS_TRANSPOSE transB = bT ? CblasTrans : CblasNoTrans;

	size_t M = c.rows();
	size_t N = c.cols();
	size_t K = aT ? a.rows() : a.cols();

	float alpha = 1.0f;
	float beta = 1.0f;

	size_t lda = aT ? K : M;
	size_t ldb = bT ? N : K;
	size_t ldc = M;

	cblas_sgemm( CblasColMajor, transA, transB, M, N, K, alpha,
				 a.data(), lda,
				 b.data(), ldb, beta, c.data(), ldc );


}
#endif /* USE_BLAS */
