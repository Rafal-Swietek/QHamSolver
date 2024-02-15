#include "includes/config.hpp"
#include "../include/QHamSolver.h"
#include "includes/pertIsing.hpp"



//<! ---------------------------------------------------------------------------------------------------------------------------------------
//<! ------------------------------------------------------------------------------------------------------------------------ IMPLEMENTATION

//<! ------------------------------------------------------------------------------ CONSTRUCTORS

/// @brief Constructor for pertIsing model class
/// @param _BC boundary condition
/// @param L system size
/// @param g perturbation strength
pertIsing::pertIsing(int _BC, unsigned int L, double g, const u64 seed)
{ 
    CONSTRUCTOR_CALL;

    this->_boundary_condition = _BC;
    this->system_size = L; 
    this->_g = g;
    
    init(); 
}

/// @brief Constructor from input stream
/// @param os input stream
pertIsing::pertIsing(std::istream& os)
    { os >> *this; }

//<! ------------------------------------------------------------------------------ HAMILTONIAN BUILDERS
/// @brief Set hamiltonian matrix element given with value and new index
/// @param k current basis state
/// @param value value of matrix element
/// @param new_idx new index to be found in hilbert space
void pertIsing::set_hamiltonian_elements(u64 k, double value, u64 new_idx)
{
    u64 idx = this->_hilbert_space.find(new_idx);
    try {
        H(idx, k) += value;
    } 
    catch (const std::exception& err) {
        std::cout << "Exception:\t" << err.what() << "\n";
        std::cout << "SHit ehhh..." << std::endl;
        printSeparated(std::cout, "\t", 14, true, new_idx, idx, this->_hilbert_space(k), value);
    }
}


/// @brief Method to create hamiltonian within the class
void pertIsing::create_hamiltonian()
{
    this->H = sparse_matrix(this->dim, this->dim);
    
    this->_disorder = this->disorder_generator.gaussian(this->system_size, 0, 1);
    #ifdef EXTRA_DEBUG
        std::cout << this->_disorder << std::endl;
    #endif

    this->_couplings = this->disorder_generator.gaussian_matrix(this->system_size, 0, 1);
    #ifdef EXTRA_DEBUG
        std::cout << this->_couplings << std::endl;
    #endif

    // const double rescale = 1. / (this->system_size * std::log(this->system_size));
    const double rescale = 1. / std::sqrt(this->system_size);
    for (size_t k = 0; k < this->dim; k++) {
		size_t base_state = this->_hilbert_space(k);
	    for (int i = 0; i < this->system_size; i++) 
        {
            //<! random fields
            auto [Sz_i, _] = operators::sigma_z(base_state, this->system_size, i);
            this->H(k, k) += this->_disorder(i) * std::real(Sz_i);

            //<! perturbation (transverse for now)
            auto [val, state] = operators::sigma_x(base_state, this->system_size, i);
            this->set_hamiltonian_elements(k, this->_g * rescale * std::real(val), state);

            //<! Long-range ZZ random interaction
            for (int j = 0; j < this->system_size && j != i; j++) {
                auto [Sz_j, _] = operators::sigma_z(base_state, this->system_size, j);
                this->H(k, k) += this->_couplings(i, j) / std::sqrt(this->system_size) * std::real(Sz_i * Sz_j);
	        }
	    }
	}
}


/// @brief Method to create hamiltonian within the class
/// @param site site index where the local hamiltonian acts
/// @return the local hamiltonian at site site
typename pertIsing::sparse_matrix pertIsing::create_local_hamiltonian(int site)
{
    sparse_matrix H_local(dim, dim);
    
    return H_local;
}


//<! ------------------------------------------------------------------------------ OVVERRIDEN OPERATORS AND OPERATOR KERNELS
/// @brief Read model parameters from input stream
/// @tparam U1_sector U(1) symmetry sector as teamplate input 
/// @param os input stream to read parameters
std::istream& pertIsing::read(std::istream& os)
{
    
    return os;
}

/// @brief Write hamiltonian to stream as human readable
/// @param os input stream to read parameters
std::ostream& pertIsing::write(std::ostream& os) const
{
    printSeparated(os, "\t", 16, true, "Model:", "pertIsing spin chain");
    os << std::endl;
    printSeparated(os, "\t", 16, true, "Hamiltonian:", "H = 1/sqrt{L}\u03A3_ij J_ij S^z_iS^z_j + \u03A3_i (h_i S^z_i) + gV");
    printSeparated(os, "\t", 16, true, "Perturbation:", "V = 1/sqrt{L ln(L)}\u03A3_i S^x_i");
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
    printSeparated(os, "\t", 16, true, "Parameters:");
    printSeparated(os, "\t", 16, true, "L", this->system_size);
    printSeparated(os, "\t", 16, true, "g", this->_g);
    printSeparated(os, "\t", 16, true, "seed", this->_seed);
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");

    return os;
}
