#pragma once

#include "user_interface.hpp"

/// @brief General UI class for disordered models (includes locally perturbed models)
/// @tparam Hamiltonian template model (haamiltonian) type
template <class Hamiltonian>
class user_interface_dis : public user_interface<Hamiltonian>{
	static_check((std::is_base_of_v<_hamiltonian, Hamiltonian>), 
                    "\n" BAD_INHERITANCE "\n\t base class is: hamiltonian_base<element_type, hilbert_space>");
protected:
    int realisations;									// number of realisations to average on for disordered case - symmetries got 1
    size_t seed;										// radnom seed for random generator
    int jobid;											// unique _id given to current job

	typedef typename user_interface<Hamiltonian>::model_pointer model_pointer;
    typedef typename user_interface<Hamiltonian>::element_type element_type;
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

	virtual void diagonalize() 							override;
	virtual void spectral_form_factor() 				override;
	virtual void average_sff() 							override;
	virtual void eigenstate_entanglement() 				override;
	virtual void eigenstate_entanglement_degenerate() 	override;
	virtual void analyze_spectra() 						override;
	virtual void diagonal_matrix_elements() 			override;
	virtual void multifractality()						override;
	virtual void matrix_elements()						override {};

	virtual 
	arma::Col<element_type> 
	cast_state(const arma::Col<element_type>& state) override 
		{ return state; }
};

// include definitions 
#include "user_interface_dis_aux.hpp"