#pragma once
#ifndef BLOCK_LANCZOSBUILD
#define BLOCK_LANCZOSBUILD

 
namespace lanczos 
{

	//<! builds lanczos tridiagonal matrix with or without
	//<! orthogonalization and no krylov space in memory
	template <typename _ty>
	inline
	void BlockLanczos<_ty>::build_lanczos()
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
		double E0 = 0;
		for (int j = 0; j < this->lanczos_steps; j++) {
			arma::Mat<_ty> W = H * Vk;
			arma::Mat<_ty> alfa = Vk.t() * W;
			this->H_lanczos.submat(j * this->bundle_size, j * this->bundle_size, (j+1) * this->bundle_size - 1, (j+1) * this->bundle_size - 1) = alfa;
			
			W = W - Vk_1 * beta.t() - Vk * alfa;
			Vk_1 = Vk;
			if(j < this->lanczos_steps - 1)
			{
				arma::qr_econ(Vk, beta, W);

				this->H_lanczos.submat(j * this->bundle_size, (j+1) * this->bundle_size, (j+1) * this->bundle_size - 1, (j+2) * this->bundle_size - 1) = beta.t();
				this->H_lanczos.submat((j+1) * this->bundle_size, j * this->bundle_size, (j+2) * this->bundle_size - 1, (j+1) * this->bundle_size - 1) = beta;
			}

			//<! convergence
			// double Enew = arma::eig_sym(this->H_lanczos.submat(0, 0, (j+1) * this->bundle_size - 1, (j+1) * this->bundle_size - 1))(0);
			// printSeparated(std::cout, "\t", 15, true, j, std::abs(Enew - E0));
			// E0 = Enew;
		}
	}

	//<! builds lanczos tridiagonal matrix
	//<! with orthogonalization and krylov space
	template <typename _ty>
	inline 
	void BlockLanczos<_ty>::build_krylov()
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
		double E0 = 0.0;
		for (int j = 0; j < this->lanczos_steps; j++) {
			arma::Mat<_ty> Vk = this->krylov_space.cols(j * this->bundle_size, (j+1) * this->bundle_size - 1);
			
			arma::Mat<_ty> W = H * Vk;
			arma::Mat<_ty> alfa = Vk.t() * W;
			this->H_lanczos.submat(j * this->bundle_size, j * this->bundle_size, (j + 1) * this->bundle_size - 1, (j + 1) * this->bundle_size - 1) = alfa;
			
			this->orthogonalize(W, j);
			if(j < this->lanczos_steps - 1)
			{
				
				arma::Mat<_ty> beta(this->bundle_size, this->bundle_size, arma::fill::zeros);
				arma::qr_econ(Vk, beta, W);
				this->krylov_space.cols((j+1) * this->bundle_size, (j+2) * this->bundle_size - 1) = Vk;

				this->H_lanczos.submat(j * this->bundle_size, (j+1) * this->bundle_size, (j+1) * this->bundle_size - 1, (j+2) * this->bundle_size - 1) = beta.t();
				this->H_lanczos.submat((j+1) * this->bundle_size, j * this->bundle_size, (j+2) * this->bundle_size - 1, (j+1) * this->bundle_size - 1) = beta;
			}
			//<! convergence
			// double Enew = arma::eig_sym(this->H_lanczos.submat(0, 0, (j+1) * this->bundle_size - 1, (j+1) * this->bundle_size - 1))(0);
			// printSeparated(std::cout, "\t", 15, true, j, std::abs(Enew - E0));
			// E0 = Enew;
		}
	}

	/// @brief Build Block-Lanczos matrix with random initial input states
	/// @tparam _ty Type of Hamiltonian matrix (-> type of lanczos matrix and vectors)
	template <typename _ty>
	inline
	void BlockLanczos<_ty>::build(const arma::Mat<_ty>& random) {
		this->initial_bundle = random;
		this->build();
	}

	/// @brief Build Block-Lanczos matrix with random initial states stored in class
	/// @tparam _ty Type of Hamiltonian matrix (-> type of lanczos matrix and vectors)
	template <typename _ty>
	inline
	void BlockLanczos<_ty>::build() {
		if (this->use_krylov)
			this->build_krylov();
		else
			this->build_lanczos();
	}

	/// @brief orthogonalizes input vector to the krylov space spanned by the first j bundles
	/// @tparam _ty Type of Hamiltonian matrix (-> type of lanczos matrix and vectors)
	/// @param vec_to_ortho Input state (in Hilbert basis) to orthogonalize in Krylov basis
	/// @param j dimension of Krylov basis
	template <typename _ty>
	inline
	void BlockLanczos<_ty>::orthogonalize(
			arma::Col<_ty>& vec_to_ortho,			//<! vector to orthogonalize
			int j									//<! current dimension of Krylov space
		) {
		arma::Col<_ty> temporary(this->N, arma::fill::zeros);
		for (int k = 0; k < j * this->bundle_size; k++)
			temporary += arma::cdot(this->krylov_space.col(k), vec_to_ortho) * this->krylov_space.col(k);

		vec_to_ortho = vec_to_ortho - temporary;
	};

	/// @brief orthogonalizes input matrix (set of vectors) to the krylov space spanned by the first j bundles
	/// @tparam _ty Type of Hamiltonian matrix (-> type of lanczos matrix and vectors)
	/// @param mat_to_ortho Input matrix (bundle of states in Hilbert basis) to orthogonalize in Krylov basis
	/// @param j dimension of Krylov basis (lanczos step)
	template <typename _ty>
	inline
	void BlockLanczos<_ty>::orthogonalize(
			arma::Mat<_ty>& mat_to_ortho,			//<! vector to orthogonalize
			int k									//<! current dimension of Krylov space
		) {
		// for(int s = 0; s < mat_to_ortho.n_cols; s++){
		// 	arma::Col<_ty> temporary(this->N, arma::fill::zeros);
		// 	for (int k = 0; k < j * this->bundle_size; k++)
		// 		temporary += arma::cdot(this->krylov_space.col(k), mat_to_ortho.col(s)) * this->krylov_space.col(k);

		// 	mat_to_ortho.col(s) = mat_to_ortho.col(s) - temporary;
		// }
		arma::Mat<_ty> temporary(this->N, this->bundle_size, arma::fill::zeros);
			for (int j = 0; j <= k; j++){
				auto Vs = this->krylov_space.cols(j * this->bundle_size, (j+1) * this->bundle_size - 1);
				temporary += Vs * (Vs.t() * mat_to_ortho);
			}

			mat_to_ortho = mat_to_ortho - temporary;
	};

	

}

#endif