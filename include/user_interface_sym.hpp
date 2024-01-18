#pragma once

#include "user_interface.hpp"

/// @brief General UI class for symmetric models
/// @tparam Hamiltonian template model (haamiltonian) type 
template <class Hamiltonian>
class user_interface_sym : public user_interface<Hamiltonian>{
	static_check((std::is_base_of_v<QHS::_hamiltonian, Hamiltonian>), 
                    "\n" BAD_INHERITANCE "\n\t base class is: hamiltonian_base<element_type, hilbert_space>");

protected:
	typedef typename user_interface<Hamiltonian>::model_pointer model_pointer;
    typedef typename user_interface<Hamiltonian>::element_type element_type;

public:
    virtual model_pointer create_new_model_pointer() override = 0;
	virtual void reset_model_pointer() override = 0;

	virtual void make_sim() override = 0;			//<! main simulation funciton

	// ------------------------------------------------- MAIN ROUTINES
	

	virtual void diagonalize() 							override;
	virtual void spectral_form_factor() 				override {};
	virtual void average_sff() 							override {};
	virtual void analyze_spectra() 						override;

	virtual void multifractality()						override {};
	
	virtual void eigenstate_entanglement() 				override;
	virtual void eigenstate_entanglement_degenerate() 	override;
	
	virtual void diagonal_matrix_elements() 			override;
	virtual void matrix_elements()						override{};
	
	virtual 
	arma::Col<element_type> 
	cast_state(const arma::Col<element_type>& state) override 
		{ return state; }

	// dummy override to not write for all models yet
	virtual arma::sp_mat energy_current() override { return arma::sp_mat(); };
	
	virtual element_type
	jE_mat_elem_kernel(
		const arma::Col<element_type>& state1, 
		const arma::Col<element_type>& state2,
		int i, u64 k, const QOps::_ifun& check_spin
		) override { return element_type(0); };
};

// include definitions 
#include "user_interface_sym_aux.hpp"