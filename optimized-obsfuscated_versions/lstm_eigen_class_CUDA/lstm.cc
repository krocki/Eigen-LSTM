/*
* @Author: kmrocki
* @Date:   2016-02-19 10:25:43
* @Last Modified by:   kmrocki
* @Last Modified time: 2016-03-09 11:36:57
*
* Eigen + CUDA version of LSTM
*
*/


#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <timer.h>
#include <lstm.h>
#include <cu_lstm.h>
#include <cu_matrix.h>
#include <matrix.h>
#include <fstream>

#define eps 1e-10

void randn ( Matrix &m, double mean, double stddev );
void randnblock ( Matrix m, double mean, double stddev );
Eigen::MatrixXi rawread ( const char *filename );
void adagrad ( Parameters *p, Parameters *d, Parameters *m,
			   double learning_rate );
bool check_gradient_error ( const char *message, Matrix &n,
							Matrix &m );
bool check_gradients ( Parameters &n, Parameters &d );
double test ( Parameters &p, Eigen::MatrixXi testdata );
std::vector<char> sample ( Parameters &p,
						   size_t characters_to_generate );
double count_flops ( size_t M, size_t N, size_t S,
					 size_t B );

//y = sqrt(x + eps)
inline dtype sqrt_eps ( const dtype x ) {
	return ( dtype ) sqrt ( x + eps );
}

float reset_std = 0.00f;

int main() {

	/* process only if we are using NVCC */
	#ifdef __GPU__
	cudaSetDevice ( 4 );
	init_curand();
	init_cublas();
	#endif
	
	// hidden size
	const size_t N = 256;
	// vocab size (# of distinct observable events)
	const size_t M = 256;
	// sequence length for learning
	const size_t S = 25;
	// batch size
	const size_t B = 16;
	
	double learning_rate = 1e-1;
	
	
	size_t epochs = 1000;
	
	size_t test_every_seconds = 900;
	
	// read text
	Eigen::MatrixXi _data = rawread ( "enwik8.txt" );
	std::string in_filename = "enwik8_test_25_256";
	std::string out_filename = "enwik8_test_25_256";
	
	//select 'train_percent' % of data as training data
	size_t train_percent = 99;
	size_t percent_size = _data.size() / 100;
	size_t test_fraction = 1;
	Eigen::MatrixXi data = _data.block ( 0, 0,
										 train_percent * percent_size,
										 1 ); // first x% * percent_size
	Eigen::MatrixXi testdata = _data.block ( (
								   100 - test_fraction ) * percent_size, 0,
							   _data.size() - ( 100 - test_fraction ) * percent_size, 1 );
							   
	// DEBUG
	std::cout 	<< "Train set size: " 	<< data.size() 						<<
				", "
				<< "Test set size: " 	<< testdata.size() 					<< ", "
				<< "Total: " 			<< data.size() + testdata.size()	<<
				std::endl;
				
	if ( data.size() + testdata.size() != _data.size() )
		std::cout << data.size() + testdata.size() << " != " <<
				  _data.size() << " !!!" << std::endl;
				  
	LSTM<S> lstm ( M, N, B );								//LSTM state
	
	Parameters p ( M, N );									//the actual weights
	Parameters d ( M, N );									//gradients
	Parameters n ( M, N );									//numrical gradients
	Parameters m ( M, N );
	m.reset();											//gradients' history
	
	#ifdef __GPU__
	cuLSTM<S> cuda_lstm ( M, N, B );						//LSTM state
	cuParameters cuda_p ( M, N );
	cuParameters cuda_d ( M, N );
	cuParameters cuda_m ( M, N );
	cuParameters cuda_n ( M, N );
	cuda_m.zero();
	#endif
	
	Matrix codes ( N,
				   N ); 						// this is an identity matrix that
	// is used to encode inputs, 1 of K encoding
	
	Matrix surprisals ( M,
						1 );					// losses: errors = p - target
						
	// init matrices
	randn ( p.W, 0, 0.01 ); 								//normal distr
	randn ( p.U, 0, 0.01 ); 								//normal distr
	randn ( p.Why, 0, 0.01 ); 								//normal distr
	
	//set f gates biases to 1
	p.b.block ( 2 * N, 0, N, 1 ) = Matrix::Ones ( N, 1 );
	
	codes = Matrix::Identity ( M,
							   M ); 			//Identity matrix (MATLAB's eye())
							   
	Timer epoch_timer;
	Timer test_timer;
	Timer flops_timer;
	size_t length = data.rows();
	size_t positions[B];
	double loss, epoch_loss;
	
	Matrix results;
	size_t results_size = 0;
	
	/************************************/
	
	// some approximation on the number of operations for benchmarking
	double flops_per_iteration = count_flops ( M, N, S, B );
	
	double flops_per_epoch = flops_per_iteration *
							 ( length - S );
							 
	double gflops_per_sec = 0;
	
	p.load_from_disk ( "models/" + in_filename );
	
	#ifdef __GPU__
	
	//synchronize cpu and gpu copies
	copy_parameters_to_device ( p, cuda_p );
	//inputs and targets are still generated on cpu
	copy_lstm_to_device<S> ( lstm, cuda_lstm );
	
	#endif
	
	test_timer.start();
	
	for ( size_t e = 0; e < epochs; e++ ) {
	
		epoch_loss = 0.0;
		
		//initial positions
		
		for ( size_t b = 0; b < B; b++ )
		
			positions[b] = rand() % ( length - S ) + S;
			
			
		randn ( lstm.h[0], 0, reset_std );
		randn ( lstm.c[0], 0, reset_std );
		
		epoch_timer.start();
		flops_timer.start();
		
		for ( size_t i = S; i < length; i++ ) {
		
			loss = 0;
			
			double test_time = test_timer.end();
			
			if ( test_time > test_every_seconds ) {
			
				#ifdef __GPU__
				copy_parameters_to_host ( cuda_p, p );
				#endif
				double train_error = epoch_loss * double (
										 length ) / double ( i );
										 
				double test_error = test ( p, testdata );
				
				std::cout << "Train error: " << train_error <<
						  ", Test error: " << test_error << std::endl;
						  
				results_size++;
				
				Matrix current = Matrix::Zero ( 1, 5 );
				
				current ( 0, 0 ) = ( float ) ( results_size - 1 );
				current ( 0, 1 ) = test_time;
				current ( 0, 2 ) = train_error;
				current ( 0, 3 ) = test_error;
				current ( 0, 4 ) = gflops_per_sec;
				
				std::cout << current << std::endl << std::endl;
				
				Matrix new_results = Matrix::Zero ( results.rows() + 1, 5 );
				
				if ( results_size > 1 )
					new_results.block ( 0, 0, results.rows(), 5 ) = results;
					
				new_results.block ( results_size - 1, 0, 1, 5 ) = current;
				
				results = Matrix::Zero ( results.rows() + 1, 5 );
				results = new_results;
				
				save_matrix_to_file ( results,
									  "models/" + out_filename + ".txt" );
				p.save_to_disk ( "models/" + out_filename );
				
				/* sample */
				std::vector<char> generated_text = sample ( p, 5000 );
				
				std::ofstream FILE ( "models/" + out_filename +
									 "_sample.txt", std::ios::out | std::ofstream::binary );
				std::copy ( generated_text.begin(), generated_text.end(),
							std::ostreambuf_iterator<char> ( FILE ) );
							
				test_timer.start();
			}
			
			if ( ( ( i + 1 ) % 100 ) == 0 ) {
			
				double flops_time = flops_timer.end();
				size_t eta_sec = ( flops_time * ( ( float ) length -
												  ( float ) i ) ) / 100.0f;
				size_t eta_hours = eta_sec / 3600;
				size_t eta_min = ( eta_sec % 3600 ) / 60;
				eta_sec = eta_sec % 60;
				
				gflops_per_sec = ( 100.0f * flops_per_iteration / powf ( 2,
								   30 ) ) / flops_time;
								   
				std::cout << std::setw ( 15 ) << "[Epoch " << e + 1 << "/"
						  << epochs << "]" << std::fixed <<
						  std::setw ( 10 ) << std::setprecision ( 2 ) <<
						  100.0f * ( float ) ( i + 1 ) / ( float ) length <<
						  "%     (eta " <<
						  std::setw ( 2 ) << eta_hours << " h " <<
						  std::setfill ( '0' ) <<
						  std::setw ( 2 ) << eta_min << " m " << std::setfill ( '0' )
						  <<
						  std::setw ( 2 ) << eta_sec << " s)" << std::setfill ( ' ' )
						  <<
						  std::setw ( 12 ) << std::setprecision ( 6 ) << "loss = " <<
						  epoch_loss *double ( length ) / double ( i ) <<
						  std::setw ( 9 ) << std::setprecision ( 2 ) <<
						  gflops_per_sec <<
						  
						  " GFlOP/s\r" <<  std::flush;
						  
				flops_timer.start();
			}
			
			#ifdef __GPU__
			copy_context_to_host ( cuda_lstm, lstm );
			#endif
			
			for ( size_t b = 0; b < B; b++ ) {
			
				size_t event = ( ( int * )
								 data.data() ) [positions[b]];		// current observation, uchar (0-255)
								 
				if ( positions[b] == S ) {
					randnblock ( lstm.h[0].col ( b ), 0, reset_std );
					randnblock ( lstm.c[0].col ( b ), 0, reset_std );
				}
				
				for ( size_t t = 0; t < S; t++ ) {
					size_t ev_x = ( ( int * ) data.data() ) [positions[b] - S +
								  t];
					size_t ev_t = ( ( int * ) data.data() ) [positions[b] - S +
								  t + 1];
								  
					lstm.target[t].col ( b ) = codes.col ( ev_t );
					lstm.x[t].col ( b ) = codes.col ( ev_x );
					
				}
				
				
				for ( size_t t = 1; t < S; t++ ) {
				
					lstm.h[t - 1].col ( b ) = lstm.h[t].col ( b );
					lstm.c[t - 1].col ( b ) = lstm.c[t].col ( b );
					
				}
				
				positions[b]++;
				
				if ( positions[b] >= length )
					positions[b] = S;
					
			}
			
			
			//inputs and targets are still generated on cpu
			//copy_inputs_to_device<S>(lstm, cuda_lstm);
			#ifdef __GPU__
			copy_lstm_to_device<S> ( lstm, cuda_lstm );
			cuda_lstm.forward ( cuda_p );
			loss += cuda_lstm.calculate_loss();
			#else
			lstm.forward ( p );
			loss += lstm.forward_loss ( p );
			#endif
			
			if ( !std::isnan ( loss ) )
				epoch_loss += loss / ( length ); // loss/char
				
			#ifdef __GPU__
			cuda_lstm.backward ( cuda_d, cuda_p );
			#else
			lstm.backward ( d, p );
			#endif
			
			if ( i == length - 1 ) {
				#ifdef __GPU__
				//gradient check on CPU still
				copy_parameters_to_host ( cuda_d, d );
				copy_parameters_to_host ( cuda_p, p );
				#endif
				double epoch_time = epoch_timer.end();
				
				std::cout  << std::endl <<
						   "=======================" << std::endl <<
						   "Epoch " << e + 1 << "/" << epochs <<
						   std::fixed << std::setprecision ( 3 ) <<
						   ", t = " << epoch_time << " s" << ", est GFLOP/s = " <<
						   ( flops_per_epoch / powf ( 2, 30 ) ) / epoch_time <<
						   ", avg loss = " << epoch_loss <<
						   " bits/char" << std::endl;
				#ifdef PRECISE_MATH
				std::cout << std::endl << "Checking gradients..." <<
						  std::endl;
				n = d;
				lstm.compute_all_numerical_grads ( n, p );
				#endif
			}
			
			//gradcheck needs to compute num grads, compare with d
			
			// weight update
			
			#ifdef __GPU__
			
			if ( i > 50 * S )
				cuda_adagrad ( &cuda_p, &cuda_d, &cuda_m, learning_rate );
			else
				cuda_adagrad ( &cuda_p, &cuda_d, &cuda_m, 0 );
				
			#else
			adagrad ( &p, &d, &m, learning_rate );
			#endif
				
				
			#ifdef __GPU__
			copy_lstm_to_host<S> ( cuda_lstm, lstm );
			#endif
		}
		
		#ifdef __GPU__
		copy_parameters_to_host ( cuda_p, p );
		#endif
		#ifdef PRECISE_MATH
		check_gradients ( n, d );
		#endif
		
		
	}
	
	#ifdef __GPU__
	teardown_cublas();
	#endif
	
	return 0;
	
}
//AdaGrad method of weight update
void adagrad ( Parameters *p, Parameters *d, Parameters *m,
			   double learning_rate ) {
			   
	m->Why += d->Why.cwiseProduct ( d->Why );
	m->by += d->by.cwiseProduct ( d->by );
	m->U += d->U.cwiseProduct ( d->U );
	m->W += d->W.cwiseProduct ( d->W );
	m->b += d->b.cwiseProduct ( d->b );
	
	// change weights:
	p->Why -= learning_rate * d->Why.cwiseQuotient (
				  m->Why.unaryExpr ( std::ptr_fun ( sqrt_eps ) ) );
	p->by -= learning_rate * d->by.cwiseQuotient (
				 m->by.unaryExpr ( std::ptr_fun ( sqrt_eps ) ) );
	p->U -= learning_rate * d->U.cwiseQuotient (
				m->U.unaryExpr ( std::ptr_fun ( sqrt_eps ) ) );
	p->W -= learning_rate * d->W.cwiseQuotient (
				m->W.unaryExpr ( std::ptr_fun ( sqrt_eps ) ) );
	p->b -= learning_rate * d->b.cwiseQuotient (
				m->b.unaryExpr ( std::ptr_fun ( sqrt_eps ) ) );
}

// returns true if everything is OK
bool check_gradient_error ( const char *message, Matrix &n,
							Matrix &m ) {
							
	Matrix diff = m - n;
	Matrix sum = n + m;
	Matrix error ( sum.rows(), sum.cols() );
	
	bool okMean = true;
	bool okMax = true;
	
	diff = diff.cwiseAbs();
	sum = sum.cwiseAbs();
	
	std::cout << std::endl;
	
	//need to check div by 0
	for ( int i = 0; i < sum.rows(); i++ ) {
		for ( int j = 0; j < sum.cols(); j++ ) {
		
			if ( sum ( i, j ) > 0.0 )
				error ( i, j ) = diff ( i, j ) / sum ( i, j );
			else
				error ( i, j ) = 0;
				
			if ( error ( i, j ) > 1e-1 )
				std::cout << i << ", " << j << ", m: " << m ( i,
						  j ) << ", n: " <<
						  n ( i, j ) << ", e: " << error ( i, j ) << std::endl;
						  
		}
	}
	
	double maxError = error.maxCoeff();
	double meanError = error.sum() / double ( error.rows() *
					   error.cols() );
					   
	if ( maxError > 1e-1 )
		okMax = false;
		
	if ( meanError > 1e-3 )
		okMean = false;
		
	std::cout 	<< std::endl
				<< std::setw ( 15 ) << std::setprecision (
					12 ) << "[" << message << "]" << std::endl
				<< std::setw ( 20 ) << " numerical range (" << std::setw (
					20 ) << n.minCoeff() <<
				", " << std::setw ( 20 ) << n.maxCoeff() << ")" << std::endl
				<< std::setw ( 20 ) << " analytical range (" << std::setw (
					20 ) << m.minCoeff() <<
				", " << std::setw ( 20 ) << m.maxCoeff() << ")" << std::endl
				<< std::setw ( 20 ) << " max rel. error " << std::setw (
					20 ) << maxError;
					
	if ( okMax == false )
		std::cout << std::setw ( 23 ) << "!!!  >1e-1 !!!";
		
	std::cout << std::endl << std::setw ( 20 ) <<
			  " mean rel. error " << std::setw ( 20 ) << meanError;
			  
	if ( okMean == false )
		std::cout << std::setw ( 23 ) << "!!!  >1e-3  !!!";
		
	std::cout << std::endl;
	
	return okMean && okMax;
}

// returns true if everything is OK
bool check_gradients ( Parameters &n, Parameters &d ) {

	bool bU = check_gradient_error ( "U", n.U, d.U );
	bool bW = check_gradient_error ( "W", n.W, d.W );
	bool bWhy = check_gradient_error ( "Why", n.Why, d.Why );
	bool bb = check_gradient_error ( "b", n.b, d.b );
	bool bby = check_gradient_error ( "by", n.by, d.by );
	bool ok = bU && bW && bWhy && bb && bby;
	
	return ok;
	
}

void randn ( Matrix &m, double mean, double stddev ) {

	// random number generator
	// unfortunately, Eigen does not implement normal distribution
	// TODO: make it cleaner, more parallel
	
	std::random_device rd;
	std::mt19937 mt ( rd() );
	std::normal_distribution<> randn ( mean, stddev );
	
	for ( int i = 0; i < m.rows(); i++ ) {
		for ( int j = 0; j < m.cols(); j++ )
			m.coeffRef ( i, j ) = randn ( mt );
	}
	
}

void randnblock ( Matrix m, double mean, double stddev ) {

	// random number generator
	// unfortunately, Eigen does not implement normal distribution
	// TODO: make it cleaner, more parallel
	
	std::random_device rd;
	std::mt19937 mt ( rd() );
	std::normal_distribution<> randn ( mean, stddev );
	
	for ( int i = 0; i < m.rows(); i++ ) {
		for ( int j = 0; j < m.cols(); j++ )
			m.coeffRef ( i, j ) = randn ( mt );
	}
	
}

Eigen::MatrixXi rawread ( const char *filename ) {

	Eigen::MatrixXi m ( 0, 0 );
	
	if ( FILE *fp = fopen ( filename, "rb" ) ) {
	
		std::vector<unsigned char> v;
		char buf[1024];
		
		while ( size_t len = fread ( buf, 1, sizeof ( buf ), fp ) )
			v.insert ( v.end(), buf, buf + len );
			
		fclose ( fp );
		
		if ( v.size() > 0 ) {
		
			std::cout << "Read " << v.size() << " bytes (" << filename
					  << ")" << std::endl;
			m.resize ( v.size(), 1 );
			
			// TODO: probably there is a better way to map std::vector to Eigen::MatrixXi
			for ( int i = 0; i < v.size(); i++ )
			
				( ( int * ) m.data() ) [i] = ( int ) v[i];
				
				
		}
		else
		
			std::cout << "Empty file! (" << filename << ")" <<
					  std::endl;
					  
					  
	}
	else
	
		std::cout << "fopen error: (" << filename << ")" <<
				  std::endl;
				  
	return m;
}

std::vector<char> sample ( Parameters &p,
						   size_t characters_to_generate ) {
						   
	size_t M = p.M;
	size_t N = p.N;
	
	Matrix _h ( N, 1 );
	Matrix _c ( N, 1 );
	Matrix _x ( M, 1 );
	Matrix _y ( M, 1 );
	Matrix _probs ( M, 1 );
	Matrix cdf = Matrix::Zero ( M, 1 );
	Matrix _g ( N * 4, 1 );
	Matrix codes = Matrix::Identity ( M, M );
	
	randn ( _h, 0, reset_std );
	randn ( _c, 0, reset_std );
	
	std::random_device rd;
	std::mt19937 gen ( rd() );
	std::uniform_real_distribution<> dis ( 0, 1 );
	std::vector<char> generated_text;
	
	for ( size_t i = 0; i < characters_to_generate; i++ ) {
	
		_y = p.Why * _h + p.by;
		#ifdef PRECISE_MATH
		_probs = _y.unaryExpr ( std::ptr_fun ( ::exp ) );
		#else
		_probs = _y.unaryExpr ( std::ptr_fun ( ::expf ) );
		#endif
		double sum = _probs.sum();
		_probs = _probs / sum;
		
		//cumsum, TODO: something nicer
		cdf ( 0, 0 ) = _probs ( 0, 0 );
		
		for ( size_t ii = 1; ii < _probs.rows(); ii++ )
			cdf ( ii, 0 ) = cdf ( ii - 1, 0 ) + _probs ( ii, 0 );
			
		double r = dis ( gen );
		
		// find the lowest number in cdf that's larger or equal to r
		size_t index = 0;
		
		for ( size_t ii = 0; ii < cdf.rows(); ii++ ) {
		
			if ( r < cdf ( ii, 0 ) ) {
			
				index = ii;
				break;
			}
			
		}
		
		generated_text.push_back ( char ( index ) );
		
		_x.col ( 0 ) = codes.row ( index );
		_g = p.W * _x + p.U * _h + p.b;
		_g.block ( 0, 0, 3 * N, 1 ) = _g.block ( 0, 0, 3 * N,
									  1 ).unaryExpr ( ( dtype ( * ) ( const dtype ) ) logistic );
		#ifdef PRECISE_MATH
		_g.block ( 3 * N, 0, N, 1 ) = _g.block ( 3 * N, 0, N,
									  1 ).unaryExpr ( std::ptr_fun ( ::tanh ) );
		#else
		_g.block ( 3 * N, 0, N, 1 ) = _g.block ( 3 * N, 0, N,
									  1 ).unaryExpr ( std::ptr_fun ( ::tanhf ) );
		#endif
		_c = _g.block ( 0, 0, N,
						1 ).cwiseProduct ( _g.block ( 3 * N, 0, N,
										   1 ) ) + _g.block ( 2 * N, 0, N, 1 ).cwiseProduct ( _c );
		#ifdef PRECISE_MATH
		_c = _c.unaryExpr ( std::ptr_fun ( ::tanh ) );
		#else
		_c = _c.unaryExpr ( std::ptr_fun ( ::tanhf ) );
		#endif
		_h = _g.block ( N, 0, N, 1 ).cwiseProduct ( _c );
		
	}
	
	return generated_text;
}

double test ( Parameters &p, Eigen::MatrixXi testdata ) {

	size_t M = p.M;
	size_t N = p.N;
	
	Matrix _h ( N, 1 );
	Matrix _c ( N, 1 );
	Matrix _x ( M, 1 );
	Matrix _y ( M, 1 );
	Matrix _probs ( M, 1 );
	Matrix cdf = Matrix::Zero ( M, 1 );
	Matrix _g ( N * 4, 1 );
	Matrix codes = Matrix::Identity ( M, M );
	double test_error = 0;
	size_t test_length = testdata.rows();
	randn ( _h, 0, reset_std );
	randn ( _c, 0, reset_std );
	
	for ( size_t ii = 0; ii < test_length - 1; ii++ ) {
	
		size_t ev_x = ( ( int * ) testdata.data() ) [ii];
		size_t ev_t = ( ( int * ) testdata.data() ) [ii + 1];
		
		_x.col ( 0 ) = codes.row ( ev_x );
		_g = p.W * _x + p.U * _h + p.b;
		_g.block ( 0, 0, 3 * N, 1 ) = _g.block ( 0, 0, 3 * N,
									  1 ).unaryExpr ( ( dtype ( * ) ( const dtype ) ) logistic );
		#ifdef PRECISE_MATH
		_g.block ( 3 * N, 0, N, 1 ) = _g.block ( 3 * N, 0, N,
									  1 ).unaryExpr ( std::ptr_fun ( ::tanh ) );
		#else
		_g.block ( 3 * N, 0, N, 1 ) = _g.block ( 3 * N, 0, N,
									  1 ).unaryExpr ( std::ptr_fun ( ::tanhf ) );
		#endif
		_c = _g.block ( 0, 0, N,
						1 ).cwiseProduct ( _g.block ( 3 * N, 0, N,
										   1 ) ) + _g.block ( 2 * N, 0, N, 1 ).cwiseProduct ( _c );
		#ifdef PRECISE_MATH
		_c = _c.unaryExpr ( std::ptr_fun ( ::tanh ) );
		#else
		_c = _c.unaryExpr ( std::ptr_fun ( ::tanhf ) );
		#endif
		_h = _g.block ( N, 0, N, 1 ).cwiseProduct ( _c );
		_y = p.Why * _h + p.by;
		
		#ifdef PRECISE_MATH
		_probs = _y.unaryExpr ( std::ptr_fun ( ::exp ) );
		#else
		_probs = _y.unaryExpr ( std::ptr_fun ( ::expf ) );
		#endif
		
		double sum = _probs.sum();
		_probs = _probs / sum;
		
		test_error += -log2 ( _probs ( ev_t ) );
		
	}
	
	return test_error / double ( ( test_length - 1 ) );
}

double count_flops ( size_t M, size_t N, size_t S,
					 size_t B ) {
					 
					 
	return ( S - 1 ) * (
			   //forward
			   ( N * M * B * 2 ) + ( N * 4 * N * B ) + ( N * 4 * B * 2 ) +
			   ( 5 * N * 4 * B ) + //nolinearities
			   ( 6 * N * B ) + //c(t) + h(t)
			   ( M * N * B * 2 ) + // y[t].array()
			   ( 8 * N * B ) + //probs[t]
			   //backward
			   ( N * B ) +
			   ( M * B * N * 3 ) +
			   ( N * B * 6 ) +
			   ( N * M * B * 4 )
			   + // dh = Why.transpose() * dy[t] + dhnext;
			   ( N * B * 8 ) +
			   ( N * 4 * B * M * 3 ) + // dU += dg * h[t - 1].transpose();
			   ( N * 4 * B * N * 3 ) + // dW += dg * lstm.x[t].transpose();
			   ( N * 4 * B ) +
			   ( N * 4 * N * B * 2 ) + //dhnext = U.transpose() * dg;
			   ( N * B ) //dcnext.array() = dc.array() * g[t].block<N, B>(2 * N, 0).array();
		   ) +
		   8 * ( M * N + M + N * 4 * N + N * 4 * M + N * 4 ); //adapt
};
