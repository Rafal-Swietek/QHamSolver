#pragma once

#define try_alloc(code) try{ code;}\
						catch(const std::bad_alloc& e) {						\
							_assert_(false, "Memory exceeded: " + std::string(e.what()));		\
						}

namespace lanczos {

	/// @brief 
	/// @tparam _ty 
	template <typename _ty, converge converge_type>
	inline
		void BlockLanczos<_ty, converge_type>::diagonalization()
	{
		try_alloc(this->build(););
		arma::eig_sym(
			this->eigenvalues,
			this->eigenvectors,
			this->H_lanczos
		);

	}

	/// @brief 
	/// @tparam _ty 
	/// @param random 
	template <typename _ty, converge converge_type>
	inline
	void BlockLanczos<_ty, converge_type>::diagonalization(
		const arma::Mat<_ty>& random	//<! random input
	) 
	{
		try_alloc(this->build(random););
		arma::eig_sym(
				this->eigenvalues,
				this->eigenvectors,
				this->H_lanczos
			);
	}

	/// @brief 
	/// @tparam _ty 
	/// @return 
	template <typename _ty, converge converge_type>
	inline
	auto 
	BlockLanczos<_ty, converge_type>::get_eigenstates() -> arma::Mat<_ty>
	{
		if(this->use_krylov){
			return this->krylov_space * this->eigenvectors;
		} else {
			_assert_(this->use_krylov, "Note implemented generating states without krylov subspace.");
		}
	}
}