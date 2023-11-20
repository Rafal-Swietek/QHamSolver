#pragma once
#ifndef BLOCK_LANCZOSBUILD
#define BLOCK_LANCZOSBUILD

 
namespace lanczos 
{

	//<! builds lanczos tridiagonal matrix with or without
	//<! orthogonalization and no krylov space in memory
	template <typename _ty, converge converge_type>
	inline
	void BlockLanczos<_ty, converge_type>::build_lanczos_converged()
	{
		// this->randVec_inKrylovSpace = arma::Col<_ty>(
		// 	this->lanczos_steps,
		// 	arma::fill::zeros
		// 	);
		this->H_lanczos = arma::Mat<_ty>(
			this->matrix_size,
			this->matrix_size,
			arma::fill::zeros
			);

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
						W(this->N, this->matrix_size);
		for (int j = 0; j < this->maxiter; j++) {
			W = this->use_on_the_fly? Hmultiply(Vk) : H * Vk;

			alfa = Vk.t() * W;
			this->H_lanczos.submat(j * this->bundle_size, j * this->bundle_size, (j+1) * this->bundle_size - 1, (j+1) * this->bundle_size - 1) = alfa;
			
			W = W - Vk_1 * beta.t() - Vk * alfa;
			Vk_1 = Vk;
			if(j < this->maxiter - 1)
			{
				arma::qr_econ(Vk, beta, W);

				this->H_lanczos.submat(j * this->bundle_size, (j+1) * this->bundle_size, (j+1) * this->bundle_size - 1, (j+2) * this->bundle_size - 1) = beta.t();
				this->H_lanczos.submat((j+1) * this->bundle_size, j * this->bundle_size, (j+2) * this->bundle_size - 1, (j+1) * this->bundle_size - 1) = beta;
			}

			//<! convergence
			if(j > this->lanczos_steps / this->bundle_size && (j - this->lanczos_steps / this->bundle_size) % 10 == 0){
				//<! convergence criterion
				arma::vec Econv; arma::Mat<_ty> Vconv;
				arma::eig_sym(Econv, Vconv, this->H_lanczos.submat(0, 0, j * this->bundle_size - 1, j * this->bundle_size - 1) );
				double conv = 1e2;
				if constexpr (converge_type == converge::energies){
					Econv = Econv.rows(0, this->lanczos_steps - 1);
					conv = arma::max(arma::abs(Econv - E0));
					E0 = Econv;
				} else if constexpr (converge_type == converge::states){ 
					arma::Row<_ty> conv2(this->lanczos_steps);
					for(int s = 0; s < this->lanczos_steps; s++)
						conv2(s) = arma::norm(beta * Vconv.submat(Vconv.n_rows - this->bundle_size, s, Vconv.n_rows - 1, s));
					conv = arma::max(conv2);
				} else{
					static_check((converge_type == converge::energies) || (converge_type == converge::states), "Not implemented other convergence criteria");
				}
				_extra_debug( printSeparated(std::cout, "\t", 15, false, this->N, j, conv, conv2); )
				if(conv < this->tolerance){
					// printSeparated(std::cout, "\t", 15, false, this->N, j, conv, conv2);
					this->lanczos_steps = j;
					break;
				}
			}
		}
		this->matrix_size = this->lanczos_steps * this->bundle_size;

		this->H_lanczos = this->H_lanczos.submat(0, 0, this->lanczos_steps - 1, this->lanczos_steps - 1);
	}

	//<! builds lanczos tridiagonal matrix
	//<! with orthogonalization and krylov space
	template <typename _ty, converge converge_type>
	inline 
	void BlockLanczos<_ty, converge_type>::build_krylov_converged()
	{
		this->krylov_space = arma::Mat<_ty>(
			this->N,
			this->matrix_size,
			arma::fill::zeros
			);
		this->H_lanczos = arma::Mat<_ty>(
			this->matrix_size,
			this->matrix_size,
			arma::fill::zeros
			);

		this->krylov_space.cols(0, this->bundle_size-1) = this->initial_bundle;
		arma::vec E0(this->lanczos_steps, arma::fill::zeros);
		int M = this->lanczos_steps;

		//<! preallocate variables in loop
		arma::Mat<_ty> beta(this->bundle_size, this->bundle_size), 
						alfa(this->bundle_size, this->bundle_size), 
						Vk(this->N, this->matrix_size), 
						W(this->N, this->matrix_size);
		for (int j = 0; j < this->maxiter; j++) {
			Vk = this->krylov_space.cols(j * this->bundle_size, (j+1) * this->bundle_size - 1);
			
			W = this->use_on_the_fly? Hmultiply(Vk) : H * Vk;
			alfa = Vk.t() * W;
			this->H_lanczos.submat(j * this->bundle_size, j * this->bundle_size, (j + 1) * this->bundle_size - 1, (j + 1) * this->bundle_size - 1) = alfa;

			this->orthogonalize(W, j);
			if(j < this->maxiter - 1)
			{
				arma::qr_econ(Vk, beta, W);
				
				this->krylov_space.cols((j+1) * this->bundle_size, (j+2) * this->bundle_size - 1) = Vk;

				this->H_lanczos.submat(j * this->bundle_size, (j+1) * this->bundle_size, (j+1) * this->bundle_size - 1, (j+2) * this->bundle_size - 1) = beta.t();
				this->H_lanczos.submat((j+1) * this->bundle_size, j * this->bundle_size, (j+2) * this->bundle_size - 1, (j+1) * this->bundle_size - 1) = beta;
			}

			//<! convergence
			if(j > this->lanczos_steps / this->bundle_size && (j - this->lanczos_steps / this->bundle_size) % 10 == 0){
				//<! convergence criterion
				arma::vec Econv; arma::Mat<_ty> Vconv;
				arma::eig_sym(Econv, Vconv, this->H_lanczos.submat(0, 0, j * this->bundle_size - 1, j * this->bundle_size - 1) );
				double conv = 1e2;
				if constexpr (converge_type == converge::energies)
				{
					Econv = Econv.rows(0, this->lanczos_steps - 1);
					conv = arma::max(arma::abs(Econv - E0));
					E0 = Econv;
				} else if constexpr (converge_type == converge::states)
				{
					arma::Row<_ty> conv2(this->lanczos_steps);
					for(int s = 0; s < this->lanczos_steps; s++)
						conv2(s) = arma::norm(beta * Vconv.submat(Vconv.n_rows - this->bundle_size, s, Vconv.n_rows - 1, s));
					conv = arma::max(conv2);
				} else{
					static_check((converge_type == converge::energies) || (converge_type == converge::states), "Not implemented other convergence criteria");
				}
				_extra_debug( printSeparated(std::cout, "\t", 15, false, this->N, j, conv); )
				if(conv < this->tolerance){
					// printSeparated(std::cout, "\t", 15, false, this->N, j, conv, conv2);
					this->lanczos_steps = j;
					break;
				}
			}
		}
		this->matrix_size = this->lanczos_steps * this->bundle_size;

		this->H_lanczos = this->H_lanczos.submat(0, 0, this->matrix_size - 1, this->matrix_size - 1);
		this->krylov_space = this->krylov_space.cols(0, this->matrix_size - 1);
	}
}

#endif