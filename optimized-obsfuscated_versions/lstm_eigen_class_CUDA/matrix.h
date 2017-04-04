#include <Eigen/Dense>

#ifndef __MATRIX__
#define __MATRIX__

#ifdef PRECISE_MATH
#define Matrix Eigen::MatrixXd
#else
#define Matrix Eigen::MatrixXf
#endif

#endif