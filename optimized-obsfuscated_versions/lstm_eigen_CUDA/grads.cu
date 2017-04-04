//check gradients

double cuda_lstm_forward(	cuda_matrix* d_g, cuda_matrix* d_x, cuda_matrix d_W, cuda_matrix d_U, cuda_matrix d_b,
							cuda_matrix* d_c, cuda_matrix* d_h, size_t S) {

	double loss = 0.0;

	for (size_t t = 1; t < S; t++) { // compute activations for sequence

		//CUDA
		cuda_zero_matrix(&d_g[t]);
		cuda_matmul(&d_g[t], &d_W, &d_x[t]);
		cuda_matmul(&d_g[t], &d_U, &d_h[t - 1]);
		cuda_gmav(&d_g[t], &d_g[t], &d_b);
		cuda_elementwise_logistic_block(&d_g[t], 0, 3 * N);
		cuda_elementwise_tanh_block(&d_g[t], 3 * N, 4 * N);
		cuda_zero_matrix(&d_c[t]);
		cuda_elementwise_mult_block(&d_c[t], 0, N, &d_g[t], 0, N, &d_g[t], 3 * N, 4 * N);
		cuda_elementwise_mult_block(&d_c[t], 0, N, &d_g[t], 2 * N, 3 * N, &d_c[t - 1], 0, N);
		cuda_elementwise_tanh(&d_c[t]);
		cuda_zero_matrix(&d_h[t]);
		cuda_elementwise_mult_block(&d_h[t], 0, N, &d_g[t], N, 2 * N, &d_c[t], 0, N);
		cuda_zero_matrix(&d_y[t]);
		cuda_matmul(&d_y[t], &d_Why, &d_h[t]);
		cuda_gmav(&d_y[t], &d_y[t], &d_by);

		cuda_copy_device_to_device(&d_y[t], &d_probs[t]);
		cuda_elementwise_exp(&d_probs[t]);

		cuda_zero_matrix(&d_sums);
		cuda_matmul(&d_sums, &d_probs_ones, &d_probs[t]);

		cuda_gmdv(&d_probs[t], &d_probs[t], &d_sums);

		cuda_copy_device_to_device(&d_probs[t], &d_neglogprobs[t]);
		cuda_elementwise_neglog2(&d_neglogprobs[t]);
		cuda_zero_matrix(&d_neglogprobs_out[t]);
		cuda_elementwise_mult(&d_neglogprobs_out[t], &d_targets[t], &d_neglogprobs[t]);

		loss += (double)cuda_matrix_sum(&d_neglogprobs_out[t]);

	}

	return loss;
}