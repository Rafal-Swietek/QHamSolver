#pragma once

namespace polfed {

	/// @brief Initialize the BlockLanczos instance with set class members
	/// @tparam _ty type of input Hamiltonian (enforces type on Krylov basis) 
	/// @tparam converge_type enum type for convergence criterion (energies or states)
	template <typename _ty, converge converge_type>
	inline 
    void POLFED<_ty, converge_type>::initialize() {
		CONSTRUCTOR_CALL;
		// if(!this->use_on_the_fly)
			this->N = this->H.n_rows;

		//<! transform input Hamiltonian
		this->transform_matrix();
	}
};