#pragma once
#ifndef BLOCK_LANCZOSBUILD
#define BLOCK_LANCZOSBUILD

 
namespace lanczos 
{
	/// @brief Calculate convergence of algorithm
	/// @tparam _ty type of input Hamiltonian (enforces type on Krylov basis) 
	/// @tparam converge_type enum type for convergence criterion (energies or states)
	/// @param Eprev lowest energies from previous iteration (to compare to)
	template <typename _ty, converge converge_type>
	inline
	double BlockLanczos<_ty, converge_type>::_calculate_convergence(arma::vec& Eprev, const arma::Mat<_ty>& beta)
	{
		// arma::eig_sym(Econv, Vconv, this->H_lanczos.submat(0, 0, j * this->bundle_size - 1, j * this->bundle_size - 1) );
		double _error = 1e2;
		if constexpr (converge_type == converge::energies)
		{
			arma::vec Econv = arma::eig_sym(this->H_lanczos);
			Econv = Econv.rows(0, this->lanczos_steps - 1);
			_error = arma::max(arma::abs(Econv - Eprev));
			Eprev = Econv;
		} else if constexpr (converge_type == converge::states)
		{
			arma::vec Econv; arma::Mat<_ty> Vconv;
			arma::eig_sym(Econv, Vconv, this->H_lanczos);
			arma::Row<_ty> conv2(this->lanczos_steps);
			for(int s = 0; s < this->lanczos_steps; s++)
				conv2(s) = arma::norm(beta * Vconv.col(s).rows(Vconv.n_rows - this->bundle_size, Vconv.n_rows - 1));
			_error = arma::max(conv2);
			// _error = arma::norm(beta * Vconv.submat(Vconv.n_rows - this->bundle_size, 0, Vconv.n_rows - 1, this->lanczos_steps - 1), "inf");
		} else{
			static_check((converge_type == converge::energies) 
						|| (converge_type == converge::states)
						|| (converge_type == converge::none), "Not implemented other convergence criteria");
		}
		return _error;
	}
	


	/// @brief Routine to build lanczos tri-block-diagonal matrix without re-orthogonalization until converged this->l_steps of states
	/// @tparam _ty type of input Hamiltonian (enforces type on Krylov basis) 
	/// @tparam converge_type enum type for convergence criterion (energies or states)
	template <typename _ty, converge converge_type>
	inline
	void BlockLanczos<_ty, converge_type>::_build_lanczos_converged()
	{
		try_alloc_matrix(this->H_lanczos, this->bundle_size, this->bundle_size);

		arma::Mat<_ty> beta(this->bundle_size, this->bundle_size, arma::fill::zeros);
		//if (this->mymemory_over_performance)
		//	this->model->hamil_vec_onthefly(random_vec, fi_next);
		//else
		arma::Mat<_ty> Vk = this->initial_bundle;
		arma::Mat<_ty> Vk_1 = Vk;
		arma::vec E0(this->lanczos_steps, arma::fill::zeros);
		int M = this->lanczos_steps;
		//<! preallocate variables in loop
		arma::Mat<_ty> alfa(this->bundle_size, this->bundle_size), 
						W(this->N, this->bundle_size);
		for (int j = 0; j < this->maxiter; j++) {
			W = Hamiltonian(Vk);

			alfa = Vk.t() * W;
			this->H_lanczos.submat(j * this->bundle_size, j * this->bundle_size, (j+1) * this->bundle_size - 1, (j+1) * this->bundle_size - 1) = alfa;

			//<! convergence
			if(j > this->lanczos_steps / this->bundle_size && (j - this->lanczos_steps / this->bundle_size) % 1 == 0)
			{	
				double conv = this->_calculate_convergence(E0, beta);
				_extra_debug( std::cout << "BlockLanczos: "; printSeparated(std::cout, "\t", 15, true, this->N, j, conv); )
				if(conv < this->tolerance){
					// printSeparated(std::cout, "\t", 15, false, this->N, j, conv, conv2);
					this->lanczos_steps = j;
					break;
				}
			}

			W = W - Vk_1 * beta.t() - Vk * alfa;
			Vk_1 = Vk;
			if(j < this->maxiter - 1)
			{
				//<! Resize matrices
				try_realloc_matrix(this->H_lanczos, (j+2) * this->bundle_size, (j+2) * this->bundle_size)

				arma::qr_econ(Vk, beta, W);

				this->H_lanczos.submat(j * this->bundle_size, (j+1) * this->bundle_size, (j+1) * this->bundle_size - 1, (j+2) * this->bundle_size - 1) = beta.t();
				this->H_lanczos.submat((j+1) * this->bundle_size, j * this->bundle_size, (j+2) * this->bundle_size - 1, (j+1) * this->bundle_size - 1) = beta;
			}
		}
	}

	/// @brief Routine to build lanczos tri-block-diagonal matrix with re-orthogonalization and krylov space until converged this->l_steps of states
	/// @tparam _ty type of input Hamiltonian (enforces type on Krylov basis) 
	/// @tparam converge_type enum type for convergence criterion (energies or states)
	template <typename _ty, converge converge_type>
	inline 
	void BlockLanczos<_ty, converge_type>::_build_krylov_converged()
	{
		try_alloc_matrix(this->krylov_space, this->N, this->bundle_size);
		try_alloc_matrix(this->H_lanczos, this->bundle_size, this->bundle_size);

		// try_alloc_matrix(this->krylov_space, this->N, this->maxiter);
		// try_alloc_matrix(this->H_lanczos, this->maxiter, this->maxiter);
		
		this->krylov_space.cols(0, this->bundle_size-1) = this->initial_bundle;
		arma::vec E0(this->lanczos_steps, arma::fill::zeros);
		int M = this->lanczos_steps;

		//<! preallocate variables in loop
		arma::Mat<_ty> beta(this->bundle_size, this->bundle_size),
						alfa(this->bundle_size, this->bundle_size), 
						Vk(this->N, this->bundle_size), 
						W(this->N, this->bundle_size);
		std::cout << "Using num  of threads = " << omp_get_num_threads() << std::endl;
		for (int j = 0; j < this->maxiter; j++) 
		{
			Vk = this->krylov_space.cols(j * this->bundle_size, (j+1) * this->bundle_size - 1);
			clk::time_point start = std::chrono::system_clock::now();
			W = Hamiltonian(Vk);
			alfa = Vk.t() * W;
			this->H_lanczos.submat(j * this->bundle_size, j * this->bundle_size, (j + 1) * this->bundle_size - 1, (j + 1) * this->bundle_size - 1) = alfa;

			//<! convergence
			if(j >= this->lanczos_steps / this->bundle_size && (j - this->lanczos_steps / this->bundle_size) % 1 == 0)
			{
				double conv = this->_calculate_convergence(E0, beta);
				_extra_debug( std::cout << "BlockLanczos: "; printSeparated(std::cout, "\t", 15, true, this->N, j, conv, "time=", tim_s(start)); )
				// printSeparated(std::cout, "\t", 15, true, this->N, j, conv, arma::norm(beta));
				if(conv < this->tolerance){
					// printSeparated(std::cout, "\t", 15, false, this->N, j, conv, conv2);
					this->lanczos_steps = j;
					break;
				}
			}


			//<! off - diagonals
			if(j < this->maxiter - 1)
			{
				this->orthogonalize(W, j);

				//<! Resize matrices
				try_realloc_matrix(this->H_lanczos,    (j+2) * this->bundle_size, (j+2) * this->bundle_size)
				try_realloc_matrix(this->krylov_space, this->N, 				  (j+2) * this->bundle_size)
				
				arma::qr_econ(Vk, beta, W);
				
				this->krylov_space.cols((j+1) * this->bundle_size, (j+2) * this->bundle_size - 1) = Vk;

				this->H_lanczos.submat(j * this->bundle_size, (j+1) * this->bundle_size, (j+1) * this->bundle_size - 1, (j+2) * this->bundle_size - 1) = beta.t();
				this->H_lanczos.submat((j+1) * this->bundle_size, j * this->bundle_size, (j+2) * this->bundle_size - 1, (j+1) * this->bundle_size - 1) = beta;
			}
			_extra_debug( std::cout << "BlockLanczos iteration: "; printSeparated(std::cout, "\t", 15, true, j, "time=", tim_s(start)); )

		}
		// this->H_lanczos = this->H_lanczos.submat(0, 0, this->lanczos_steps * this->bundle_size - 1, this->lanczos_steps * this->bundle_size - 1);
		// this->krylov_space = this->krylov_space.submat(0, 0, this->N - 1, this->lanczos_steps * this->bundle_size - 1);
	}
}

#endif