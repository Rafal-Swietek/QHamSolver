#pragma once

namespace lanczos {

	
	template <typename _ty, converge converge_type>
	inline
		void Lanczos<_ty, converge_type>::diagonalization()
	{
		this->build();
		arma::eig_sym(
			this->eigenvalues,
			this->eigenvectors,
			this->H_lanczos
		);

	}

	template <typename _ty, converge converge_type>
	inline
	void Lanczos<_ty, converge_type>::diagonalization(
		const arma::Col<_ty>& random	//<! random input
	) 
	{
		this->build(random);
		arma::eig_sym(
				this->eigenvalues,
				this->eigenvectors,
				this->H_lanczos
			);
	}

}