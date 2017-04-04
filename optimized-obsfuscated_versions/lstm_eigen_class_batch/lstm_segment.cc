//
//  lstm.cc
//
//  LSTM code - Eigen implementation + classes, grad check
//
//  Author: Kamil Rocki <kmrocki@us.ibm.com>
//  Created on: 02/19/2016
//


#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <timer.h>
#include <lstm.h>

#include <Eigen/Dense>

#define eps 1e-10

void randn(Eigen::MatrixXd& m, double mean, double stddev);
void randnblock(Eigen::MatrixXd m, double mean, double stddev);
Eigen::MatrixXi rawread(const char* filename);
void adagrad(Parameters* p, Parameters* d, Parameters* m, double learning_rate);
bool check_gradient_error(const char* message, Eigen::MatrixXd& n, Eigen::MatrixXd& m);
bool check_gradients(Parameters& n, Parameters& d);

//y = sqrt(x + eps)
inline double sqrt_eps(const double x) {
	return sqrt(x + eps);
}

int main() {

	// hidden size
	const size_t N = 64;
	// vocab size (# of distinct observable events)
	const size_t M = 256;
	// sequence length for learning
	const size_t S = 12;
	// batch size
	const size_t B = 4;

	double learning_rate = 1e-1;
	size_t epochs = 1000;

	// read text
	Eigen::MatrixXi data = rawread("lstm.h");

	LSTM<S> lstm(M, N, B);								//LSTM state
	Parameters p(M, N);									//the actual weights
	Parameters d(M, N);									//gradients
	Parameters n(M, N);									//numrical gradients
	Parameters m(M, N);									//gradients' history

	Eigen::MatrixXd codes(N, N); 						// this is an identity matrix that
	// is used to encode inputs, 1 of K encoding

	Eigen::MatrixXd x[S];								// temp matrix for storing input
	Eigen::MatrixXd target[S];							// targets - desired outputs

	for (size_t t = 0; t < S; t++) {
		target[t] = Eigen::MatrixXd::Zero(M, B);			//zeros
		x[t] = Eigen::MatrixXd::Zero(M, B);
	}

	Eigen::MatrixXd surprisals(M, 1);					// losses: errors = p - target

	// init matrices
	randn(p.W, 0, 0.01); 								//normal distr
	randn(p.U, 0, 0.01); 								//normal distr
	randn(p.Why, 0, 0.01); 								//normal distr

	codes = Eigen::MatrixXd::Identity(M, M); 			//Identity matrix (MATLAB's eye())

	Timer epoch_timer;
	Timer flops_timer;
	size_t length = data.rows();
	size_t positions[B];
	double loss, epoch_loss;

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

	size_t seg = S / 2;

	for (size_t e = 0; e < epochs; e++) {

		epoch_loss = 0.0;

		//initial positions

		for (size_t b = 0; b < B; b++) {

			positions[b] = rand() % (length - S) + S;

		}

		randn(lstm.h[0], 0, 0.01);
		randn(lstm.c[0], 0, 0.01);

		epoch_timer.start();
		flops_timer.start();

		for (size_t i = S; i < length; i += seg) {

			loss = 0;

			if (((i + 1) % 100) == 0) {

				double flops_time = flops_timer.end();
				size_t eta_sec =  (flops_time * ((float)length - (float)i)) / 100.0f;
				size_t eta_hours = eta_sec / 3600;
				size_t eta_min = (eta_sec % 3600) / 60;
				eta_sec = eta_sec % 60;

				std::cout << std::setw(15) << "[Epoch " << e + 1 << "/" << epochs << "]" << std::fixed <<
						  std::setw(10) << std::setprecision(2) <<
						  100.0f * (float)(i + 1) / (float)length << "%     (eta " <<
						  std::setw(2) << eta_hours << " h " << std::setfill('0') <<
						  std::setw(2) << eta_min << " m " << std::setfill('0') <<
						  std::setw(2) << eta_sec << " s)" << std::setfill(' ') <<
						  std::setw(12) << std::setprecision(6) << "loss = " << epoch_loss* double(length) / double(i) <<
						  std::setw(9) << std::setprecision(2) <<
						  (100.0f * flops_per_iteration / powf(2, 30)) / flops_time <<

						  " GFlOP/s\r" <<  std::flush;

				flops_timer.start();
			}

			for (size_t b = 0; b < B; b++) {

				size_t event = ((int*)data.data())[positions[b]];		// current observation, uchar (0-255)

				if (positions[b] == S) {
					randnblock(lstm.h[0].col(b), 0, 0.01);
					randnblock(lstm.c[0].col(b), 0, 0.01);
				}

				for (size_t t = 0; t < S; t++) {
					size_t ev_x = ((int*)data.data())[positions[b] - S + t];
					size_t ev_t = ((int*)data.data())[positions[b] - S + t + 1];

					target[t].col(b) = codes.col(ev_t);
					x[t].col(b) = codes.col(ev_x);

				}


				// for (size_t t = 1; t < S; t++) {

				// 	lstm.h[t - 1].col(b) = lstm.h[t].col(b);
				// 	lstm.c[t - 1].col(b) = lstm.c[t].col(b);

				// }

				lstm.h[0].col(b) = lstm.h[seg - 1].col(b);
				lstm.c[0].col(b) = lstm.c[seg - 1].col(b);


				positions[b] += seg;

				if (positions[b] >= length)
					positions[b] = S;

			}

			lstm.forward(p, x);

			// loss:
			for (size_t t = 1; t < S; t++) { // compute activations for sequence

				// -log2(probs(:, t)) .* target(:, t);
				surprisals = (-lstm.probs[t].unaryExpr(
								  std::ptr_fun(::log))).cwiseProduct(target[t]);

				// cross-entropy loss, sum logs of probabilities of target outputs
				loss += surprisals.sum();

			}

			epoch_loss += loss / (S * B * length / seg); // loss/char

			lstm.backward(d, p, x, target);

			if (i > length - seg) {
				double epoch_time = epoch_timer.end();

				std::cout  << std::endl <<
						   "=======================" << std::endl <<
						   "Epoch " << e + 1 << "/" << epochs <<
						   std::fixed << std::setprecision(3) <<
						   ", t = " << epoch_time << " s" << ", est GFLOP/s = " <<
						   (flops_per_epoch / powf(2, 30)) / epoch_time <<
						   ", avg loss = " << epoch_loss <<
						   " bits/char" << std::endl;
				std::cout << std::endl << "Checking gradients..." << std::endl;
				n = d;
				lstm.compute_all_numerical_grads(n, p, x, target);
			}

			//gradcheck needs to compute num grads, compare with d

			// weight update
			adagrad(&p, &d, &m, learning_rate);


		}


		check_gradients(n, d);

		std::vector<char> generated_text;

		size_t characters_to_generate = 1500;

		Eigen::MatrixXd _h(N, 1);
		Eigen::MatrixXd _c(N, 1);
		Eigen::MatrixXd _x(M, 1);
		Eigen::MatrixXd _y(M, 1);
		Eigen::MatrixXd _probs(M, 1);
		Eigen::MatrixXd cdf = Eigen::MatrixXd::Zero(M, 1);
		Eigen::MatrixXd _g(N * 4, 1);

		randn(_h, 0, 0.01);
		randn(_c, 0, 0.01);

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(0, 1);

		for (size_t i = 0; i < characters_to_generate; i++) {

			_y = p.Why * _h + p.by;
			_probs = _y.unaryExpr(std::ptr_fun(::exp));
			double sum = _probs.sum();
			_probs = _probs / sum;

			//cumsum, TODO: something nicer
			cdf(0, 0) = _probs(0, 0);
			for (size_t ii = 1; ii < _probs.rows(); ii++) {
				cdf(ii, 0) = cdf(ii - 1, 0) + _probs(ii, 0);
			}

			double r = dis(gen);

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
			_g = p.W * _x + p.U * _h + p.b;
			_g.block<3 * N, 1>(0, 0) =
				_g.block<3 * N, 1>(0, 0).unaryExpr((double (*)(const double))logistic);
			_g.block<N, 1>(3 * N, 0) =
				_g.block<N, 1>(3 * N, 0).unaryExpr(std::ptr_fun(::tanh));
			_c = _g.block<N, 1>(0, 0).cwiseProduct(
					 _g.block<N, 1>(3 * N, 0)) + _g.block<N, 1>(2 * N, 0).cwiseProduct(_c);
			_c = _c.unaryExpr(std::ptr_fun(::tanh));
			_h = _g.block<N, 1>(N, 0).cwiseProduct(_c);

		}

		std::cout << std::endl << std::endl << "************ Generated text |";
		for (std::vector<char>::const_iterator i =
					generated_text.begin(); i != generated_text.end(); ++i)
			std::cout << *i;

		std::cout << "| Generated text END ************" << std::endl;

	}

	return 0;

}
//AdaGrad method of weight update
void adagrad(Parameters* p, Parameters* d, Parameters* m, double learning_rate) {

	m->Why += d->Why.cwiseProduct(d->Why);
	m->by += d->by.cwiseProduct(d->by);
	m->U += d->U.cwiseProduct(d->U);
	m->W += d->W.cwiseProduct(d->W);
	m->b += d->b.cwiseProduct(d->b);

	// change weights:
	p->Why -= learning_rate * d->Why.cwiseQuotient(m->Why.unaryExpr(std::ptr_fun(sqrt_eps)));
	p->by -= learning_rate * d->by.cwiseQuotient(m->by.unaryExpr(std::ptr_fun(sqrt_eps)));
	p->U -= learning_rate * d->U.cwiseQuotient(m->U.unaryExpr(std::ptr_fun(sqrt_eps)));
	p->W -= learning_rate * d->W.cwiseQuotient(m->W.unaryExpr(std::ptr_fun(sqrt_eps)));
	p->b -= learning_rate * d->b.cwiseQuotient(m->b.unaryExpr(std::ptr_fun(sqrt_eps)));
}

// returns true if everything is OK
bool check_gradient_error(const char* message, Eigen::MatrixXd& n, Eigen::MatrixXd& m) {

	Eigen::MatrixXd diff = m - n;
	Eigen::MatrixXd sum = n + m;
	Eigen::MatrixXd error(sum.rows(), sum.cols());

	bool okMean = true;
	bool okMax = true;

	diff = diff.cwiseAbs();
	sum = sum.cwiseAbs();

	std::cout << std::endl;
	//need to check div by 0
	for (int i = 0; i < sum.rows(); i++) {
		for (int j = 0; j < sum.cols(); j++) {

			if (sum(i, j) > 0.0)
				error(i, j) = diff(i, j) / sum(i, j);
			else
				error(i, j) = 0;

			if (error(i, j) > 1e-1)
				std::cout << i << ", " << j << ", m: " << m(i, j) << ", n: " <<
						  n(i, j) << ", e: " << error(i, j) << std::endl;

		}
	}

	double maxError = error.maxCoeff();
	double meanError = error.sum() / double(error.rows() * error.cols());

	if (maxError > 1e-1)
		okMax = false;
	if (meanError > 1e-3)
		okMean = false;

	std::cout 	<< std::endl
				<< std::setw(15) << std::setprecision(12) << "[" << message << "]" << std::endl
				<< std::setw(20) << " numerical range (" << std::setw(20) << n.minCoeff() <<
				", " << std::setw(20) << n.maxCoeff() << ")" << std::endl
				<< std::setw(20) << " analytical range (" << std::setw(20) << m.minCoeff() <<
				", " << std::setw(20) << m.maxCoeff() << ")" << std::endl
				<< std::setw(20) << " max rel. error " << std::setw(20) << maxError;

	if (okMax == false)
		std::cout << std::setw(23) << "!!!  >1e-1 !!!";

	std::cout << std::endl << std::setw(20) << " mean rel. error " << std::setw(20) << meanError;

	if (okMean == false)
		std::cout << std::setw(23) << "!!!  >1e-3  !!!";

	std::cout << std::endl;

	return okMean && okMax;
}

// returns true if everything is OK
bool check_gradients(Parameters& n, Parameters& d) {

	bool bU = check_gradient_error("U", n.U, d.U);
	bool bW = check_gradient_error("W", n.W, d.W);
	bool bWhy = check_gradient_error("Why", n.Why, d.Why);
	bool bb = check_gradient_error("b", n.b, d.b);
	bool bby = check_gradient_error("by", n.by, d.by);
	bool ok = bU && bW && bWhy && bb && bby;

	return ok;

}

void randn(Eigen::MatrixXd& m, double mean, double stddev) {

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

void randnblock(Eigen::MatrixXd m, double mean, double stddev) {

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
