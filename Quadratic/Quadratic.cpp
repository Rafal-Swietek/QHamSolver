
#include "includes/config.hpp"
#include "../include/QHamSolver.h"
#include "includes/Quadratic.hpp"

//<! ---------------------------------------------------------------------------------------------------------------------------------------
//<! ------------------------------------------------------------------------------------------------------------------------ IMPLEMENTATION

//<! ------------------------------------------------------------------------------ CONSTRUCTORS
/// @brief Constructor of Quadratic models
/// @param L system size (volume is L^d)
/// @param J coupling (hopping))
/// @param w stregth of potential: disorder (Anderson), periodic (Aubey-Andre), ...
/// @param seed random seed
/// @param g periodicity for Aubry-Andre model
Quadratic::Quadratic(int L, double J, double w, const u64 seed, double g, bool _bound_cond)
{ 
    CONSTRUCTOR_CALL;

    this->system_size = L;
    this->_boundary_condition = _bound_cond;
    
    this->_J = J;

    //<! disorder terms
    this->_w = w;
    this->_seed = seed;

    //<! other terms
    this->_g = g;

    init(); 
}

// /// @brief Constructor from input stream
// /// @param os input stream
// Quadratic::Quadratic(std::istream& os)
//     { os >> *this; }

//<! ------------------------------------------------------------------------------ HAMILTONIAN BUILDERS



/// @brief Method to create hamiltonian within the class
void Quadratic::create_hamiltonian()
{
    this->H = sparse_matrix(this->dim, this->dim);
    #ifdef ANDERSON
        this->_disorder = this->disorder_generator.uniform(this->dim, this->_w / 2.);
        for(long int j = 0; j < this->dim; j++){
        
            this->H(j, j) = this->_disorder(j);
            auto neis = this->_lattice.get_neighbours(j);
        
            for(auto& nei : neis){
                if(nei > 0){
                    this->H(j, nei) = this->_J;
                    this->H(nei, j) = this->_J;
                }
            }
        }
    #elif defined(SYK)
        this->H = this->random_matrix.generate_matrix(this->dim);
    
    #elif defined(PLRB)
        this->H = this->random_matrix.generate_matrix(this->dim);
        for(long int i = 0; i < this->dim; i++)
            for(long int j = 0; j < this->dim; j++)
                this->H(i, j) = this->H(i, j) / std::pow(1 + std::abs(i - j) / this->_J, this->_g / 2.);

    #elif defined(AUBRY_ANDRE)
        double phase = this->disorder_generator.uniform_dist<double>(0, two_pi);
        for(long int j = 0; j < this->dim; j++){
            this->H(j, j) = this->_w * std::cos(two_pi * j * this->_g + phase);

            auto neis = this->_lattice.get_neighbours(j);
            for(auto& nei : neis){
                if(nei > 0){
                    this->H(j, nei) = -this->_J;
                    this->H(nei, j) = -this->_J;
                }
            }
        }
    #elif defined(FREE_FERMIONS)
        for(long int j = 0; j < this->dim; j++){
            auto neis = this->_lattice.get_neighbours(j);
            for(auto& nei : neis){
                if(nei > 0){
                    this->H(j, nei) = this->_J;
                    this->H(nei, j) = this->_J;
                }
            }
        }
    #elif defined(RP)
        arma::vec H0 = this->disorder_generator.gaussian(this->dim, 0, 1);
        this->H = arma::diagmat(H0) + this->random_matrix.generate_matrix(this->dim) / std::pow(this->dim, this->_g / 2.0);
    #else
        #pragma message("No model chosen!!! Leaving empty matrix")
    #endif

    // std::cout << arma::mat(this->H) << std::endl;
}


//<! ------------------------------------------------------------------------------ OVVERRIDEN OPERATORS AND OPERATOR KERNELS
/// @brief Read model parameters from input stream
/// @param os input stream to read parameters
std::istream& Quadratic::read(std::istream& os)
{
    
    return os;
}

/// @brief Write hamiltonian to stream as human readable
/// @param os input stream to read parameters
std::ostream& Quadratic::write(std::ostream& os) const
{
    printSeparated(os, "\t", 16, true, "Model:", "Quadratic model - d-dimensional");
    os << std::endl;
    printSeparated(os, "\t", 16, true, "Hamiltonian:", "H = J\u03A3_<i,j> A_ij c^+_i c_j + h.c + \u03A3_i h_i n_i");
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
    printSeparated(os, "\t", 16, true, "Parameters:");
    printSeparated(os, "\t", 16, true, "L", this->system_size);
    printSeparated(os, "\t", 16, true, "BC", this->_boundary_condition);

    printSeparated(os, "\t", 16, true, "J", this->_J);
    printSeparated(os, "\t", 16, true, "w", this->_w);
    printSeparated(os, "\t", 16, true, "seed", this->_seed);
    
    return os;
}
