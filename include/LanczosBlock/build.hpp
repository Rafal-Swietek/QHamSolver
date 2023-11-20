#pragma once
#ifndef BLOCK_LANCZOSBUILD
	#include "build_converged.hpp"
#endif
 
namespace lanczos 
{

	//<! builds lanczos tridiagonal matrix with or without
	//<! orthogonalization and no krylov space in memory
	template <typename _ty, converge converge_type>
	inline
	void BlockLanczos<_ty, converge_type>::build_lanczos()
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
		

		arma::Mat<_ty> Vk = this->initial_bundle;
		arma::Mat<_ty> Vk_1 = Vk;
		double E0 = 0;
		for (int j = 0; j < this->lanczos_steps; j++) {
			arma::Mat<_ty> W = this->use_on_the_fly? Hmultiply(Vk) : H * Vk;
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
		}
	}

	//<! builds lanczos tridiagonal matrix
	//<! with orthogonalization and krylov space
	template <typename _ty, converge converge_type>
	inline 
	void BlockLanczos<_ty, converge_type>::build_krylov()
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
			
			arma::Mat<_ty> W = this->use_on_the_fly? Hmultiply(Vk) : H * Vk;
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
		}
	}

	/// @brief Build Block-Lanczos matrix with random initial input states
	/// @tparam _ty Type of Hamiltonian matrix (-> type of lanczos matrix and vectors)
	template <typename _ty, converge converge_type>
	inline
	void BlockLanczos<_ty, converge_type>::build(const arma::Mat<_ty>& random) {
		this->initial_bundle = random;
		this->build();
	}

	/// @brief Build Block-Lanczos matrix with random initial states stored in class
	/// @tparam _ty Type of Hamiltonian matrix (-> type of lanczos matrix and vectors)
	template <typename _ty, converge converge_type>
	inline
	void BlockLanczos<_ty, converge_type>::build() {
		if(this->use_full_convergence){
			if (this->use_krylov)	this->build_krylov_converged();
			else					this->build_lanczos_converged();
		} else {
			if (this->use_krylov)	this->build_krylov();
			else					this->build_lanczos();
		}
	}

	

}
