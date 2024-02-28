#include "includes/config.hpp"
#include "../include/QHamSolver.h"
#include "includes/QREM.hpp"



//<! ---------------------------------------------------------------------------------------------------------------------------------------
//<! ------------------------------------------------------------------------------------------------------------------------ IMPLEMENTATION

//<! ------------------------------------------------------------------------------ CONSTRUCTORS

/// @brief Constructor for Quantum Random Energy Model class
/// @param L system size
/// @param g perturbation strength
QREM::QREM(unsigned int L, double g, const u64 seed)
{ 
    CONSTRUCTOR_CALL;

    this->system_size = L; 
    this->_g = g;
    
    init(); 
}

/// @brief Constructor from input stream
/// @param os input stream
QREM::QREM(std::istream& os)
    { os >> *this; }

//<! ------------------------------------------------------------------------------ HAMILTONIAN BUILDERS
/// @brief Set hamiltonian matrix element given with value and new index
/// @param k current basis state
/// @param value value of matrix element
/// @param new_idx new index to be found in hilbert space
void QREM::set_hamiltonian_elements(u64 k, double value, u64 new_idx)
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
void QREM::create_hamiltonian()
{
    this->H = sparse_matrix(this->dim, this->dim);
    
    this->_random_energies = this->disorder_generator.gaussian(this->dim, 0, std::sqrt(this->system_size / 2.0));
    #ifdef EXTRA_DEBUG
        std::cout << this->_disorder << std::endl;
    #endif

    const double rescale = 1. / (std::sqrt(this->system_size) * std::log(constants<double>::e * this->system_size));
    for (size_t k = 0; k < this->dim; k++) 
    {
        this->H(k, k) = this->_random_energies(k);
		size_t base_state = this->_hilbert_space(k);
	    for (int i = 0; i < this->system_size; i++) 
        {
            //<! perturbation (transverse for now)
            auto [val, state] = operators::sigma_x(base_state, this->system_size, i);
            this->set_hamiltonian_elements(k, this->_g * rescale * std::real(val), state);
	    }
	}
}


/// @brief Method to create hamiltonian within the class
/// @param site site index where the local hamiltonian acts
/// @return the local hamiltonian at site site
typename QREM::sparse_matrix QREM::create_local_hamiltonian(int site)
{
    sparse_matrix H_local(dim, dim);
    
    return H_local;
}


//<! ------------------------------------------------------------------------------ OVVERRIDEN OPERATORS AND OPERATOR KERNELS
/// @brief Read model parameters from input stream
/// @tparam U1_sector U(1) symmetry sector as teamplate input 
/// @param os input stream to read parameters
std::istream& QREM::read(std::istream& os)
{
    
    return os;
}

/// @brief Write hamiltonian to stream as human readable
/// @param os input stream to read parameters
std::ostream& QREM::write(std::ostream& os) const
{
    printSeparated(os, "\t", 16, true, "Model:", "pertIsing spin chain");
    os << std::endl;
    printSeparated(os, "\t", 16, true, "Hamiltonian:", "H = \u03A3_k=1^D E_n|k><k| + gV");
    printSeparated(os, "\t", 16, true, "Perturbation:", "V = 1/sqrt{L ln(L)}\u03A3_i S^x_i");
    printSeparated(os, "\t", 16, true, "random energies", "E_n is gaussian with sigma^2=L/2");
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
    printSeparated(os, "\t", 16, true, "Parameters:");
    printSeparated(os, "\t", 16, true, "L", this->system_size);
    printSeparated(os, "\t", 16, true, "g", this->_g);
    printSeparated(os, "\t", 16, true, "seed", this->_seed);
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");

    return os;
}
