#pragma once
#ifndef _LANCZOSBUILD
#define _LANCZOSBUILD

 
namespace lanczos 
{

	//<! builds lanczos tridiagonal matrix with or without
	//<! orthogonalization and no krylov space in memory
	template <typename _ty, converge converge_type>
	inline
	void Lanczos<_ty, converge_type>::_build_lanczos_converged()
	{
		this->randVec_inKrylovSpace = arma::Col<_ty>(
			this->maxiter,
			arma::fill::zeros
			);
		this->H_lanczos = arma::Mat<_ty>(
			this->maxiter,
			this->maxiter,
			arma::fill::zeros
			);

		//<! set intial steps
		const u64 N = H.n_cols; //<! dimension of Hilbert space
		randVec_inKrylovSpace(0) = arma::cdot(this->initial_random_vec, this->initial_random_vec); // =1

		arma::Col<_ty> fi_next(N, arma::fill::zeros);
		//if (this->mymemory_over_performance)
		//	this->model->hamil_vec_onthefly(random_vec, fi_next);
		//else
		fi_next = H * this->initial_random_vec;

		arma::Col<_ty> fi_prev = this->initial_random_vec;
		_ty alfa = arma::cdot(this->initial_random_vec, fi_next);
		fi_next = fi_next - alfa * this->initial_random_vec;
		H_lanczos(0, 0) = alfa;

		//<! lanczos procedure
		int M = this->lanczos_steps;
		arma::vec Econv, E0(this->lanczos_steps, arma::fill::zeros);; 
		arma::Mat<_ty> Vconv;
			
		_ty beta = arma::norm(fi_next);

		for (int j = 1; j < this->maxiter; j++) {
			arma::Col<_ty> fi = fi_next / beta;
			randVec_inKrylovSpace(j) = arma::cdot(fi, this->initial_random_vec);

			//if (this->mymemory_over_performance)
			//	this->model->hamil_vec_onthefly(fi, fi_next);
			//else
			fi_next = H * fi;


			alfa = arma::cdot(fi, fi_next);
			fi_next = fi_next - alfa * fi - beta * fi_prev;

			H_lanczos(j, j) = alfa;
			H_lanczos(j, j - 1) = beta;
			H_lanczos(j - 1, j) = my_conjungate(beta);

			fi_prev = fi;
			
			beta = arma::norm(fi_next);
			
			//<! convergence
			if(j >= this->lanczos_steps && (j-this->lanczos_steps) % 10 == 0){	
				arma::eig_sym(Econv, Vconv, this->H_lanczos.submat(0, 0, j - 1, j - 1) );
				
				double conv = 1e2;
				if constexpr (converge_type == converge::energies){
					Econv = Econv.rows(0, this->lanczos_steps - 1);
					conv = arma::mean(arma::abs(Econv - E0));
					E0 = Econv;
				} else if constexpr (converge_type == converge::states){ 
					arma::Row<_ty> conv2 = arma::abs(beta * Vconv.cols(0, this->lanczos_steps-1).row(Vconv.n_rows - 1));
					conv = arma::max(conv2);
				} else{
					static_check((converge_type == converge::energies) || (converge_type == converge::states), "Not implemented other convergence criteria");
				}
				// printSeparated(std::cout, "\t", 15, false, this->N, j, conv, beta, conv2);
				if(conv < this->tolerance){
					// printSeparated(std::cout, "\t", 15, true, this->N, j, conv);
					this->lanczos_steps = j;
					break;
				}
			}
		}
		if(this->lanczos_steps == M)
			this->lanczos_steps = this->maxiter;

		this->H_lanczos = this->H_lanczos.submat(0, 0, this->lanczos_steps - 1, this->lanczos_steps - 1);
		this->randVec_inKrylovSpace = arma::normalise(this->randVec_inKrylovSpace.rows(0, this->lanczos_steps - 1));
	}

	//<! builds lanczos tridiagonal matrix
	//<! with orthogonalization and krylov space
	template <typename _ty, converge converge_type>
	inline 
	void Lanczos<_ty, converge_type>::_build_krylov_converged()
	{
		this->krylov_space = arma::Mat<_ty>(
			this->N,
			this->maxiter,
			arma::fill::zeros
			);
		this->H_lanczos = arma::Mat<_ty>(
			this->maxiter,
			this->maxiter,
			arma::fill::zeros
			);

		this->krylov_space.col(0) = this->initial_random_vec;
		arma::Col<_ty> fi_next = this->H * krylov_space.col(0);

		_ty alfa = arma::cdot(this->krylov_space.col(0), fi_next);
		fi_next = fi_next - alfa * this->krylov_space.col(0);
		H_lanczos(0, 0) = alfa;
		int M = this->lanczos_steps;

		arma::vec Econv, E0(this->lanczos_steps, arma::fill::zeros);; 
		arma::Mat<_ty> Vconv;
		
		_ty beta = arma::norm(fi_next);
		for (int j = 1; j < this->maxiter; j++) {

			this->krylov_space.col(j) = fi_next / beta;

			fi_next = this->H * this->krylov_space.col(j);

			alfa = arma::cdot(this->krylov_space.col(j), fi_next);
			this->orthogonalize(fi_next, j);

			this->H_lanczos(j, j) = alfa;
			this->H_lanczos(j, j - 1) = beta;
			this->H_lanczos(j - 1, j) = my_conjungate(beta);

			beta = arma::norm(fi_next);
			
			//<! convergence
			if(j >= this->lanczos_steps && (j-this->lanczos_steps) % 10 == 0){	
				arma::eig_sym(Econv, Vconv, this->H_lanczos.submat(0, 0, j - 1, j - 1) );
				
				double conv = 1e2;
				if constexpr (converge_type == converge::energies){
					Econv = Econv.rows(0, this->lanczos_steps - 1);
					conv = arma::mean(arma::abs(Econv - E0));
					E0 = Econv;
				} else if constexpr (converge_type == converge::states){ 
					arma::Row<_ty> conv2 = arma::abs(beta * Vconv.cols(0, this->lanczos_steps-1).row(Vconv.n_rows - 1));
					conv = arma::max(conv2);
				} else{
					static_check((converge_type == converge::energies) || (converge_type == converge::states), "Not implemented other convergence criteria");
				}
				// printSeparated(std::cout, "\t", 15, false, this->N, j, conv, beta, conv2);
				if(conv < this->tolerance){
					// printSeparated(std::cout, "\t", 15, true, this->N, j, conv);
					this->lanczos_steps = j;
					break;
				}
			}
		}
		if(this->lanczos_steps == M)
			this->lanczos_steps = this->maxiter;

		this->H_lanczos = this->H_lanczos.submat(0, 0, this->lanczos_steps - 1, this->lanczos_steps - 1);
		this->krylov_space = this->krylov_space.cols(0, this->lanczos_steps - 1);

		this->randVec_inKrylovSpace = this->krylov_space.t() * this->initial_random_vec;
	}

}

#endif