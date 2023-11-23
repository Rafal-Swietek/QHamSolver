#pragma once


namespace lanczos {

	/// @brief Runs the building of the Lanczos matrix and diagonalizes to find energies and eigenstates (in Krylov basis)
	/// @tparam _ty type of input Hamiltonian (enforces type on Krylov basis) 
	/// @tparam converge_type enum type for convergence criterion (energies or states)
	template <typename _ty, converge converge_type>
	inline
		void BlockLanczos<_ty, converge_type>::diagonalization()
	{
		this->build();
		arma::eig_sym(
			this->eigenvalues,
			this->eigenvectors,
			this->H_lanczos
		);

	}

	/// @brief Runs the building of the Lanczos matrix and diagonalizes to find energies and eigenstates (in Krylov basis)
	/// @tparam _ty type of input Hamiltonian (enforces type on Krylov basis) 
	/// @tparam converge_type enum type for convergence criterion (energies or states)
	/// @param random Initial random bundle of vectors
	template <typename _ty, converge converge_type>
	inline
	void BlockLanczos<_ty, converge_type>::diagonalization(
		const arma::Mat<_ty>& random	//<! random input
	) 
	{
		this->build(random);
		arma::eig_sym(
				this->eigenvalues,
				this->eigenvectors,
				this->H_lanczos
			);
	}

	/// @brief Get eigenstates in Hilbert space
	/// @tparam _ty type of input Hamiltonian (enforces type on Krylov basis) 
	/// @tparam converge_type enum type for convergence criterion (energies or states)
	/// @return Matrix with transformed eigenstates
	template <typename _ty, converge converge_type>
	inline
	auto 
	BlockLanczos<_ty, converge_type>::get_eigenstates() -> arma::Mat<_ty>
	{
		if(this->use_krylov){
			return this->krylov_space * this->eigenvectors;
		} else {
			_assert_(this->use_krylov, "Note implemented generating states without krylov subspace.");
			return arma::Mat<_ty>();
		}
	}
}