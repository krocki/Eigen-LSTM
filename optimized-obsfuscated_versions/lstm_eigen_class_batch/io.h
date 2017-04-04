/*
* @Author: kmrocki
* @Date:   2016-03-24 15:25:43
* @Last Modified by:   kmrocki
* @Last Modified time: 2016-03-24 18:36:57
*
* Matrix IO
*
*/

#ifndef __IO_H__
#define __IO_H__

#include <Eigen/Dense>
#include <fstream>

void save_matrix_to_file(Eigen::MatrixXd& m, std::string filename) {

	// std::cout << "Saving a matrix to " << filename << "... " << std::endl;
	// std::ofstream file(filename.c_str());

	// if (file.is_open()) {

	// 	file << m;
	// 	file.close();

	// } else {

	// 	std::cout << "file save error: (" << filename << ")" << std::endl;

	// }

}

void load_matrix_from_file(Eigen::MatrixXd& m, std::string filename) {

	// std::cout << "Loading a matrix from " << filename << "... " << std::endl;
	// std::ifstream file(filename.c_str());

	// if (file.is_open()) {

	// 	file >> m;
	// 	file.close();

	// } else {

	// 	std::cout << "file load error: (" << filename << ")" << std::endl;

	// }

}

#endif