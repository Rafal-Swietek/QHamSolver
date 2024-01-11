
#include "includes/config.hpp"
#include "../include/QHamSolver.h"
#include "includes/RandomMatrix.hpp"

//<! ---------------------------------------------------------------------------------------------------------------------------------------
//<! ------------------------------------------------------------------------------------------------------------------------ IMPLEMENTATION

//<! ------------------------------------------------------------------------------ CONSTRUCTORS
/// @brief Constructor of Quantum Sun model
/// @param dim matrix size
/// @param b 
/// @param alfa
/// @param gamma
/// @param seed random seed
RandomMatrix::RandomMatrix(u64 dim, double b, double alfa, double gamma, const u64 seed)
{ 
    CONSTRUCTOR_CALL;

    this->dim = dim;
    this->_b = b;
    this->_alfa = alfa;
    this->_gamma = gamma;
    init(); 
}

/// @brief Constructor from input stream
/// @param os input stream
RandomMatrix::RandomMatrix(std::istream& os)
    { os >> *this; }

//<! ------------------------------------------------------------------------------ HAMILTONIAN BUILDERS


/// @brief Method to create hamiltonian within the class
void RandomMatrix::create_hamiltonian()
{
    
	this->H = this->generator.generate_matrix(this->dim);
    // if(this->_norm_grain)
    //     H_grain /= std::sqrt(ULLPOW(this->grain_size) + 1);
}


/// @brief Method to create hamiltonian within the class
/// @param site site index where the local hamiltonian acts
/// @return the local hamiltonian at site site
typename RandomMatrix::sparse_matrix RandomMatrix::create_local_hamiltonian(int site)
{
    sparse_matrix H_local(this->dim, this->dim);
    
    return H_local;
}


//<! ------------------------------------------------------------------------------ OVVERRIDEN OPERATORS AND OPERATOR KERNELS
/// @brief Read model parameters from input stream
/// @tparam U1_sector U(1) symmetry sector as teamplate input 
/// @param os input stream to read parameters
std::istream& RandomMatrix::read(std::istream& os)
{
    
    return os;
}

/// @brief Write hamiltonian to stream as human readable
/// @param os input stream to read parameters
std::ostream& RandomMatrix::write(std::ostream& os) const
{
    printSeparated(os, "\t", 16, true, "Model:", "Quantum Sun model - O-dimensional EBT toy model");
    os << std::endl;
    printSeparated(os, "\t", 16, true, "Hamiltonian:", "H = Random Matrix", stringize(ENSEMBLE));
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
    printSeparated(os, "\t", 16, true, "Parameters:");
    printSeparated(os, "\t", 16, true, "dim", this->dim);

    printSeparated(os, "\t", 16, true, "b", this->_b);
    printSeparated(os, "\t", 16, true, "\u03B1", this->_alfa);
    printSeparated(os, "\t", 16, true, "\u03B3", this->_gamma);

    printSeparated(os, "\t", 16, true, "    seed", this->_seed);
    
    return os;
}
