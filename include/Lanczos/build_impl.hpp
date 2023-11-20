#pragma once
#ifndef _LANCZOSBUILD
#define _LANCZOSBUILD

 
namespace lanczos 
{

	//<! builds lanczos tridiagonal matrix with or without
	//<! orthogonalization and no krylov space in memory
	template <typename _ty>
	inline
	void Lanczos<_ty>::build_lanczos()
	{
		this->randVec_inKrylovSpace = arma::Col<_ty>(
			this->lanczos_steps,
			arma::fill::zeros
			);
		this->H_lanczos = arma::Mat<_ty>(
			this->lanczos_steps,
			this->lanczos_steps,
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
		for (int j = 1; j < this->lanczos_steps; j++) {
			_ty beta = arma::norm(fi_next);
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
		}
		randVec_inKrylovSpace = arma::normalise(randVec_inKrylovSpace);
	}

	//<! builds lanczos tridiagonal matrix
	//<! with orthogonalization and krylov space
	template <typename _ty>
	inline 
	void Lanczos<_ty>::build_krylov()
	{
		this->krylov_space = arma::Mat<_ty>(
			this->N,
			this->lanczos_steps,
			arma::fill::zeros
			);
		this->H_lanczos = arma::Mat<_ty>(
			this->lanczos_steps,
			this->lanczos_steps,
			arma::fill::zeros
			);

		this->krylov_space.col(0) = this->initial_random_vec;
		arma::Col<_ty> fi_next = this->H * krylov_space.col(0);

		_ty alfa = arma::cdot(this->krylov_space.col(0), fi_next);
		fi_next = fi_next - alfa * this->krylov_space.col(0);
		H_lanczos(0, 0) = alfa;

		double E0 = 0.0;
		for (int j = 1; j < this->lanczos_steps; j++) {
			_ty beta = arma::norm(fi_next);
			this->krylov_space.col(j) = fi_next / beta;

			fi_next = this->H * this->krylov_space.col(j);

			alfa = arma::cdot(this->krylov_space.col(j), fi_next);
			this->orthogonalize(fi_next, j);

			this->H_lanczos(j, j) = alfa;
			this->H_lanczos(j, j - 1) = beta;
			this->H_lanczos(j - 1, j) = my_conjungate(beta);

			//<! convergence
			// double Enew = arma::eig_sym(this->H_lanczos.submat(0, 0, j, j))(0);
			// printSeparated(std::cout, "\t", 15, true, j, std::abs(Enew - E0));
			// E0 = Enew;
		}
		this->randVec_inKrylovSpace = this->krylov_space.t() * this->initial_random_vec;
	}

	//<! general lanczos build selecting either memory efficient or with krylov space
	template <typename _ty>
	inline
	void Lanczos<_ty>::build(const arma::Col<_ty>& random) {
		this->initial_random_vec = random;
		this->build();
	}

	template <typename _ty>
	inline
	void Lanczos<_ty>::build() {
		if (this->use_krylov)
			this->build_krylov();
		else
			this->build_lanczos();
	}

	//<! orthogonalizes input vector to the krylov space,
	//<! spanned by the first j vectors
	template <typename _ty>
	inline
	void Lanczos<_ty>::orthogonalize(
			arma::Col<_ty>& vec_to_ortho,			//<! vector to orthogonalize
			int j									//<! current dimension of Krylov space
		) {
		arma::Col<_ty> temporary(this->N, arma::fill::zeros);
		for (int k = 0; k <= j; k++)
			temporary += arma::cdot(this->krylov_space.col(k), vec_to_ortho) * this->krylov_space.col(k);

		vec_to_ortho = vec_to_ortho - temporary;

		// temporary.zeros();
		// for (int k = 0; k <= j; k++)
		// 	temporary += arma::cdot(this->krylov_space.col(k), vec_to_ortho) * this->krylov_space.col(k);

		// vec_to_ortho = vec_to_ortho - temporary;
	};

	

}

#endif