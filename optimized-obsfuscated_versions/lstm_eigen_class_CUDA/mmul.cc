#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <timer.h>
#include <cu_matrix.h>
#include <Eigen/Dense>

void randmat(Eigen::MatrixXd& m, double mean, double stddev) {

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

//cublas test
int main() {

	const size_t N = 16;
	const size_t M = 9;
	const size_t S = 9;

	Eigen::MatrixXd A(16, 256);
	Eigen::MatrixXd B(4, 256);
	Eigen::MatrixXd C(16, 4);
	Eigen::MatrixXd D(16, 9);
	Eigen::MatrixXd E(4, 9);

	randmat(A, 0, 0.01);
	randmat(B, 0, 0.01);
	randmat(C, 0, 0.01);
	randmat(D, 0, 0.01);
	randmat(E, 0, 0.01);

	init_curand();
	init_cublas();

	cuda_matrix a, b, c, d, e;

	cuda_alloc_matrix(&a, A.rows(), A.cols());
	cuda_alloc_matrix(&b, B.rows(), B.cols());
	cuda_alloc_matrix(&c, C.rows(), C.cols());
	cuda_alloc_matrix(&d, D.rows(), D.cols());
	cuda_alloc_matrix(&e, E.rows(), E.cols());

	cuda_copy_host_to_device(A, &a);
	cuda_copy_host_to_device(B, &b);
	cuda_copy_host_to_device(D, &d);

	// cuda_matmul(&c, &a, &b, 0, false, false);

	// std::cout << "C = A * B" << std::endl << C << std::endl << std::endl;
	// print_cuda_matrix(&c);

	// C = A * B.transpose();

	// cuda_matmul(&c, &a, &b, 0, false, true);

	// std::cout << "C = A * B'" << std::endl << C << std::endl << std::endl;
	// print_cuda_matrix(&c);

	C = A * B.transpose();

	cuda_matmul(&c, &a, &b, 0, false, true);

	std::cout << "C = A * B'" << std::endl << C << std::endl << std::endl;
	print_cuda_matrix(&c);

	E = C.transpose() * D;
	cuda_matmul(&e, &c, &d, 0, true, false);

	std::cout << "E = C' * D" << std::endl << E << std::endl << std::endl;
	print_cuda_matrix(&e);
	cuda_free_matrix(&a);
	cuda_free_matrix(&b);
	cuda_free_matrix(&c);
	cuda_free_matrix(&d);
	cuda_free_matrix(&e);
	teardown_cublas();
}