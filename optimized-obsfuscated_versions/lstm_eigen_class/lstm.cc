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
	const size_t N = 20;
	// vocab size (# of distinct observable events)
	const size_t M = 256;
	// sequence length for learning
	const size_t S = 50;

	double learning_rate = 1e-1;
	size_t epochs = 1000;

	// read text
	Eigen::MatrixXi data = rawread("alice29.txt");

	LSTM lstm(M, N, S);									//LSTM state
	Parameters p(M, N);									//the actual weights
	Parameters d(M, N);									//gradients
	Parameters n(M, N);									//numrical gradients
	Parameters m(M, N);									//gradients' history

	Eigen::MatrixXd codes(N, N); 						// this is an identity matrix that
	// is used to encode inputs, 1 of K encoding

	Eigen::MatrixXd x(M, S);							// temp matrix for storing input
	Eigen::MatrixXd target(M, S);						// targets - desired outputs

	Eigen::MatrixXd surprisals(M, 1);					// losses: errors = p - target


	// init matrices
	randn(p.W, 0, 0.01); 								//normal distr
	randn(p.U, 0, 0.01); 								//normal distr
	randn(p.Why, 0, 0.01); 								//normal distr

	codes = Eigen::MatrixXd::Identity(M, M); 			//Identity matrix (MATLAB's eye())

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

		randn(lstm.h, 0, 0.01); 									//normal distr
		randn(lstm.c, 0, 0.01); 									//normal distr

		t.start();

		for (size_t i = S; i < length; i++) {

			loss = 0;

			size_t event = ((int*)data.data())[i];		// current observation, uchar (0-255)

			for (size_t s = 1; s < S; s++) {

				// shift inputs by 1
				x.col(s - 1) = x.col(s);
				// shift targets by 1
				target.col(s - 1) = target.col(s);
				lstm.h.col(s - 1) = lstm.h.col(s);
				lstm.c.col(s - 1) = lstm.c.col(s);

			}

			// column S - 1 hold the most recent events, column 0 - oldest
			target.col(S - 1) = codes.row(event);	// current observation, encoded
			x.col(S - 1) = target.col(S - 2);		// previous observation is current input

			lstm.forward(p, x);

			if (i == length - 1)
				lstm.compute_all_numerical_grads(n, p, x, target);

			// loss:
			for (size_t t = 1; t < S; t++) { // compute activations for sequence

				// -log2(probs(:, t)) .* target(:, t);
				surprisals = (-lstm.probs.col(t).unaryExpr(
								  std::ptr_fun(::log))).cwiseProduct(target.col(t));

				// cross-entropy loss, sum logs of probabilities of target outputs
				loss += surprisals.sum();

			}

			epoch_loss += loss / (S * length);

			lstm.backward(d, p, x, target);

			//gradcheck needs to compute num grads, compare with d

			// weight update
			adagrad(&p, &d, &m, learning_rate);

			if ((i % 100) == 0) {

				std::cout << std::fixed <<
						  std::setw(7) << std::setprecision(2) <<
						  100.0 * (double)i / (double)length << "%\r" << std::flush;
			}
		}

		check_gradients(n, d);

		double epoch_time = t.end();

		std::cout  << std::endl <<
				   "=======================" << std::endl <<
				   "Epoch " << e + 1 << "/" << epochs <<
				   std::fixed << std::setprecision(3) <<
				   ", t = " << epoch_time << " s" << ", est GFLOP/s = " <<
				   (flops_per_epoch / powf(2, 30)) / epoch_time <<
				   ", avg loss = " << epoch_loss <<
				   " bits/char" << std::endl;

		std::vector<char> generated_text;

		size_t characters_to_generate = 2500;

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
				<< std::setw(20) << " numerical range (" << std::setw(20) << n.minCoeff() << ", " << std::setw(20) << n.maxCoeff() << ")" << std::endl
				<< std::setw(20) << " analytical range (" << std::setw(20) << m.minCoeff() << ", " << std::setw(20) << m.maxCoeff() << ")" << std::endl
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