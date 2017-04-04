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

#include <datatype.h>

void save_matrix_to_file(Matrix& m, std::string filename) {

	std::cout << "Saving a matrix to " << filename << "... " << std::endl;
	std::ofstream file(filename.c_str());

	if (file.is_open()) {

		file << m;
		file.close();

	} else {

		std::cout << "file save error: (" << filename << ")" << std::endl;

	}

}

#define MAXBUFSIZE  ((int) 1e6)

void readMatrix(Matrix& m, const char* filename) {

	std::ifstream infile;
	infile.open(filename);

	size_t row, col;

	row = 0;

	if (infile.is_open()) {

		while (! infile.eof()) {

			std::string line;
			getline(infile, line);

			std::stringstream stream(line);

			col = 0;

			while (! stream.eof()) {
				stream >> m(row, col);
				col++;
			}

			row++;

		}

		infile.close();

		std::cout << "! Found " << row << " x " << col << " matrix: (" << filename << ")" << std::endl;

	} else {

		std::cout << "file read error: (" << filename << ")" << std::endl;
	}

};

void load_matrix_from_file(Matrix& m, std::string filename) {

	// assume that m has proper dimensions
	readMatrix(m, filename.c_str());

}

#endif