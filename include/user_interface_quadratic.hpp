#pragma once

#include "user_interface_dis.hpp"
#include "hilbert_space/u1.hpp"

#include "lattices/_base.hpp"

#include "single_particle/many_body_configurations.hpp"
#include "single_particle/correlators.hpp"
#include "single_particle/entanglement.hpp"
#include "single_particle/many_body_state.hpp"

/// @brief General UI class for disordered models (includes locally perturbed models)
/// @tparam Hamiltonian template model (haamiltonian) type
template <class Hamiltonian>
class user_interface_quadratic : public user_interface_dis<Hamiltonian>{
	static_check((std::is_base_of_v<QHS::_hamiltonian, Hamiltonian>), 
                    "\n" BAD_INHERITANCE "\n\t base class is: hamiltonian_base<element_type, hilbert_space>");
protected:
    int V;				// volume of system (L^d)
	u64 dim;		    // Hilbert-space dimension of system ( L^d or 2^(L^d) )

	typedef typename user_interface_dis<Hamiltonian>::model_pointer model_pointer;
    typedef typename user_interface_dis<Hamiltonian>::element_type element_type;
public:
    virtual model_pointer create_new_model_pointer() override = 0;
	virtual void reset_model_pointer() override = 0;

	virtual void make_sim() override = 0;			//<! main simulation funciton

	virtual void set_default() override;			
	virtual void parse_cmd_options(int argc, std::vector<std::string> argv) override;		// the function to parse the command line

	virtual void printAllOptions() const override;
	// ------------------------------------------------- MAIN ROUTINES
	template <
		typename callable,	//<! callable lambda function
		typename... _types	//<! argument-types passed to lambda
		> 
	void average_over_realisations(
		callable& lambda, 			//!< callable function
		_types... args				//!< arguments passed to callable interface lambda
	){
	#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
		for(int r = 0; r < this->realisations; r++){
			auto dummy_lambda = [&lambda](int real, auto... args){
				lambda(real, args...);
			};
			dummy_lambda(r, args...);
		}
    };

	// virtual void diagonalize() 				override;
	// virtual void spectral_form_factor() 	override;
	// virtual void average_sff() 				override;
	virtual void eigenstate_entanglement() 	override;
    virtual void eigenstate_entanglement_degenerate() override;

	virtual void eigenstate_entanglement_manybody();
	// virtual void analyze_spectra() 			override;
	virtual void diagonal_matrix_elements() override;
};

// include definitions 
#include "user_interface_quadratic_aux.hpp"