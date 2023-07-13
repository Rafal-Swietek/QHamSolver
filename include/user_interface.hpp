#pragma once

#include "QHamSolver.h"

#include "statistics.hpp"
#include "statistics_dist.hpp"
#include "spectrals.hpp"
#include "thermodynamics.hpp"
#include "entanglement.hpp"
#include "adiabatic_gauges.hpp"

extern int outer_threads;

inline
std::vector<std::string> 
change_input_to_vec_of_str(int argc, char** argv)
{
	// -1 because first is the name of the file
	std::vector<std::string> tmp(argc - 1, "");
	for (int i = 0; i < argc - 1; i++)
		tmp[i] = argv[i + 1];
	return tmp;
}

//<! -------------------------------------------------------------------------------------------
//<! ----------------------------------------------------------------------------- BASE UI CLASS

/// @brief General UI class
/// @tparam Hamiltonian template model (haamiltonian) type
template <class Hamiltonian>
class user_interface {
	static_check((std::is_base_of_v<_hamiltonian, Hamiltonian>), 
                    "\n" BAD_INHERITANCE "\n\t base class is: hamiltonian_base<element_type, hilbert_space>");
protected:
	unsigned int thread_number = 1;									// number of threads
	int boundary_conditions = 1;									// boundary conditions - 0 - PBC, 1 - OBC, 2 - ABC,...
	std::string saving_dir = "";									// directory for files to be saved onto

	// ----------------------------------- FUNCTIONS FOR READING THE INPUT

	std::string getCmdOption(const v_1d<std::string>& vec, std::string option) const;					 	// get the option from cmd input

	// ----------------------------------- TEMPLATES
	typedef std::unique_ptr<QHamSolver<Hamiltonian>> model_pointer;
    typedef typename Hamiltonian::element_type element_type;
	
	template <typename _ty>
	void set_option(_ty& value, const v_1d<std::string>& argv, 
                        std::string choosen_option, bool geq_0 = false);					        // set an option

	//template <typename _ty>
	//void set_default_msg(_ty& value, string option, string message, 
    //                                const unordered_map <string, string>& map) const;	    // setting value to default and sending a message

    unsigned int L, Ls, Ln;								// lattice params
    bool ch;											// boolean values

    double q_ipr;										// q for participation ratio calculation
    double beta;										// inverse temperature
    
	int mu;												// small bucket for the operator fluctuations to be averaged onto
	int num_of_points;									// number of points for time evolution/spectral functions/SFF/etc..
    
	int site;											// site for operator averages
    int op;												// choose operator
    int fun;											// choose function to start calculations

	model_pointer ptr_to_model;
public:
	virtual ~user_interface() = default;

	// ----------------------------------- HELPING FUNCIONS
	virtual void set_default();			
	virtual void print_help() const;													// set default parameters

	/// @brief Exits the program and prints description of all UI parameters.
	virtual void exit_with_help() const
		{ this->print_help(); std::exit(1); }
	
	virtual void printAllOptions() const;
	virtual std::string set_info(std::vector<std::string> skip = {}, 
										std::string sep = "_") const = 0;
	
	// ------------------------------------------------- REAL PARSING
	virtual void parse_cmd_options(int argc, std::vector<std::string> argv);			// the function to parse the command line

	// ------------------------------------------------- NON-VIRTUALS

	std::vector<std::string> parse_input_file(string filename) const;					// if the input is taken from file we need to make it look the same way as the command line does

	// ------------------------------------------------- SIMULATIONS
	virtual model_pointer create_new_model_pointer() = 0;
	virtual void reset_model_pointer() = 0;

	virtual void make_sim() = 0;			//<! main simulation funciton

	// ------------------------------------------------- MAIN ROUTINES

	arma::vec get_eigenvalues(std::string _suffix = "", bool diag_if_empty = true);
	virtual void diagonalize() = 0;
	virtual void analyze_spectra() = 0;

	virtual void spectral_form_factor() = 0;
	virtual void average_sff() = 0;
	
	virtual void eigenstate_entanglement() = 0;
	virtual void eigenstate_entanglement_degenerate() = 0;
	virtual void diagonal_matrix_elements() = 0;
};

// include implementation
#include "user_interface_aux.hpp"
