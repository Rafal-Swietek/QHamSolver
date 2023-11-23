#pragma once
#ifndef BLOCK_LANCZOSBUILD
	#include "_build_converged.hpp"
#endif
 
namespace lanczos 
{
	/// @brief Routine to build lanczos tri-block-diagonal matrix without re-orthogonalization
	/// @tparam _ty type of input Hamiltonian (enforces type on Krylov basis) 
	/// @tparam converge_type enum type for convergence criterion (energies or states)
	template <typename _ty, converge converge_type>
	inline
	void BlockLanczos<_ty, converge_type>::_build_lanczos()
	{
		try_alloc_matrix(this->H_lanczos, this->matrix_size, this->matrix_size);

		arma::Mat<_ty> beta(this->bundle_size, this->bundle_size, arma::fill::zeros);
		

		arma::Mat<_ty> Vk = this->initial_bundle;
		arma::Mat<_ty> Vk_1 = Vk;
		double E0 = 0;
		for (int j = 0; j < this->lanczos_steps; j++) {
			arma::Mat<_ty> W = Hamiltonian(Vk);
			arma::Mat<_ty> alfa = Vk.t() * W;
			this->H_lanczos.submat(j * this->bundle_size, j * this->bundle_size, (j+1) * this->bundle_size - 1, (j+1) * this->bundle_size - 1) = alfa;
			
			W = W - Vk_1 * beta.t() - Vk * alfa;
			Vk_1 = Vk;
			if(j < this->lanczos_steps - 1)
			{
				//<! Resize matrices
				this->H_lanczos.resize(		(j+2) * this->bundle_size, (j+2) * this->bundle_size);
				
				arma::qr_econ(Vk, beta, W);

				this->H_lanczos.submat(j * this->bundle_size, (j+1) * this->bundle_size, (j+1) * this->bundle_size - 1, (j+2) * this->bundle_size - 1) = beta.t();
				this->H_lanczos.submat((j+1) * this->bundle_size, j * this->bundle_size, (j+2) * this->bundle_size - 1, (j+1) * this->bundle_size - 1) = beta;
			}
		}
	}

	/// @brief Routine to build lanczos tri-block-diagonal matrix with re-orthogonalization and krylov space
	/// @tparam _ty type of input Hamiltonian (enforces type on Krylov basis) 
	/// @tparam converge_type enum type for convergence criterion (energies or states)
	template <typename _ty, converge converge_type>
	inline 
	void BlockLanczos<_ty, converge_type>::_build_krylov()
	{
		
		try_alloc_matrix(this->krylov_space, this->N, this->matrix_size);
		try_alloc_matrix(this->H_lanczos, this->matrix_size, this->matrix_size);

		this->krylov_space.cols(0, this->bundle_size-1) = this->initial_bundle;
		double E0 = 0.0;
		for (int j = 0; j < this->lanczos_steps; j++) {
			arma::Mat<_ty> Vk = this->krylov_space.cols(j * this->bundle_size, (j+1) * this->bundle_size - 1);
			
			arma::Mat<_ty> W = Hamiltonian(Vk);
			arma::Mat<_ty> alfa = Vk.t() * W;
			this->H_lanczos.submat(j * this->bundle_size, j * this->bundle_size, (j + 1) * this->bundle_size - 1, (j + 1) * this->bundle_size - 1) = alfa;
			
			this->orthogonalize(W, j);
			if(j < this->lanczos_steps - 1)
			{
				//<! Resize matrices
				this->H_lanczos.resize(		(j+2) * this->bundle_size, (j+2) * this->bundle_size);
				this->krylov_space.resize(	this->N, 				   (j+2) * this->bundle_size);
				
				arma::Mat<_ty> beta(this->bundle_size, this->bundle_size, arma::fill::zeros);
				arma::qr_econ(Vk, beta, W);
				
				this->krylov_space.cols((j+1) * this->bundle_size, (j+2) * this->bundle_size - 1) = Vk;

				this->H_lanczos.submat(j * this->bundle_size, (j+1) * this->bundle_size, (j+1) * this->bundle_size - 1, (j+2) * this->bundle_size - 1) = beta.t();
				this->H_lanczos.submat((j+1) * this->bundle_size, j * this->bundle_size, (j+2) * this->bundle_size - 1, (j+1) * this->bundle_size - 1) = beta;
			}
		}
	}

	/// @brief Build Block-Lanczos matrix with random initial input states
	/// @tparam _ty type of input Hamiltonian (enforces type on Krylov basis) 
	/// @tparam converge_type enum type for convergence criterion (energies or states)
	template <typename _ty, converge converge_type>
	inline
	void BlockLanczos<_ty, converge_type>::build(const arma::Mat<_ty>& random) {
		this->initial_bundle = random;
		this->build();
	}

	/// @brief Build Block-Lanczos matrix with random initial states stored in class
	/// @tparam _ty type of input Hamiltonian (enforces type on Krylov basis) 
	/// @tparam converge_type enum type for convergence criterion (energies or states)
	template <typename _ty, converge converge_type>
	inline
	void BlockLanczos<_ty, converge_type>::build() {
		// if constexpr (converge_type == converge::none) -> no convergence (no need for bool)
		if(this->use_full_convergence){
			if (this->use_krylov)	this->_build_krylov_converged();
			else					this->_build_lanczos_converged();
		} else {
			if (this->use_krylov)	this->_build_krylov();
			else					this->_build_lanczos();
		}
	}

	

}
