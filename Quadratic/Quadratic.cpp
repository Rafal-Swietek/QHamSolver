
#include "includes/config.hpp"
#include "../include/QHamSolver.h"
#include "includes/Quadratic.hpp"

//<! ---------------------------------------------------------------------------------------------------------------------------------------
//<! ------------------------------------------------------------------------------------------------------------------------ IMPLEMENTATION

//<! ------------------------------------------------------------------------------ CONSTRUCTORS
/// @brief Constructor of Quantum Sun model
/// @param L system size (L = L_loc + M)
/// @param J coupling of grain to localized spins
/// @param alfa regulates decay of coupling to furthest spins
/// @param w bandwidth of disorder on localized spins
/// @param hz uniform magnetic field
/// @param seed random seed
/// @param N size of ergodic grain
/// @param zeta random positions for coupling
/// @param initiate_avalanche boolean value if initiate avalanche by hand (put fisrt coupling without decay)
/// @param normalize_grain normalize grain to unit hilbert-schmidt norm?
Quadratic::Quadratic(int L, double J, double w, const u64 seed)
{ 
    CONSTRUCTOR_CALL;

    this->system_size = L;

    this->_J = J;

    //<! disorder terms
    this->_w = w;
    this->_seed = seed;

    init(); 
}

/// @brief Constructor from input stream
/// @param os input stream
Quadratic::Quadratic(std::istream& os)
    { os >> *this; }

//<! ------------------------------------------------------------------------------ HAMILTONIAN BUILDERS



/// @brief Method to create hamiltonian within the class
void Quadratic::create_hamiltonian()
{
    this->H = sparse_matrix(this->dim, this->dim);
    #ifdef ANDERSON
        this->_disorder = this->disorder_generator.uniform(this->dim, this->_w / 2.);
        for(long int j = 0; j < this->dim; j++){
        
            this->H(j, j) = this->_disorder(j);
            auto neis = lattice.get_neighbours(j);
        
            for(auto& nei : neis){
                if(nei > 0){
                    this->H(j, nei) = this->_J;
                    this->H(nei, j) = this->_J;
                }
            }
        }
    #elif defined(SYK)
        this->H = this->random_matrix.generate_matrix(this->dim);

    #elif defined(FREE_FERMIONS)
        for(long int j = 0; j < this->dim; j++){
            auto neis = lattice.get_neighbours(j);
            for(auto& nei : neis){
                if(nei > 0){
                    this->H(j, nei) = this->_J;
                    this->H(nei, j) = this->_J;
                }
            }
        }
    #else
        #pragma message("No model chosen!!! Leaving empty matrix")
    #endif
}


//<! ------------------------------------------------------------------------------ OVVERRIDEN OPERATORS AND OPERATOR KERNELS
/// @brief Read model parameters from input stream
/// @tparam U1_sector U(1) symmetry sector as teamplate input 
/// @param os input stream to read parameters
std::istream& Quadratic::read(std::istream& os)
{
    
    return os;
}

/// @brief Write hamiltonian to stream as human readable
/// @param os input stream to read parameters
std::ostream& Quadratic::write(std::ostream& os) const
{
    printSeparated(os, "\t", 16, true, "Model:", "Quantum Sun model - O-dimensional EBT toy model");
    os << std::endl;
    printSeparated(os, "\t", 16, true, "Hamiltonian:", "H = J\u03A3_<i,j> c^+_i c_j + h.c + \u03A3_i h_i n_i");
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
    printSeparated(os, "\t", 16, true, "Parameters:");
    printSeparated(os, "\t", 16, true, "L", this->system_size);

    printSeparated(os, "\t", 16, true, "J", this->_J);
    printSeparated(os, "\t", 16, true, "w", this->_w);
    printSeparated(os, "\t", 16, true, "seed", this->_seed);
    
    return os;
}
