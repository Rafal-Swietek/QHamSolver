
#include "includes/config.hpp"
#include "../include/QHamSolver.h"
#include "includes/QuadQSun.hpp"

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
QuadQSun::QuadQSun(int L, double J, double alfa, double gamma,
            double w, double hz, const u64 seed, int N, double zeta, bool initiate_avalanche, bool normalize_grain )
{ 
    CONSTRUCTOR_CALL;

    this->num_of_spins = L; 
    this->grain_size = N;
    this->system_size = this->num_of_spins + this->grain_size;

    this->_J = J;
    this->_alfa = alfa;
    this->_zeta = zeta;
    this->_gamma = gamma;
    
    this->_hz = hz;
    
    //<! disorder terms
    this->_w = w;
    
    this->_seed = seed;

    this->_initiate_avalanche = initiate_avalanche;
    this->_norm_grain = normalize_grain;
    init();
}

/// @brief Constructor from input stream
/// @param os input stream
QuadQSun::QuadQSun(std::istream& os)
    { os >> *this; }

//<! ------------------------------------------------------------------------------ HAMILTONIAN BUILDERS
/// @brief Set hamiltonian matrix element given with value and new index
/// @param k current basis state
/// @param value value of matrix element
/// @param new_idx new index to be found in hilbert space
void QuadQSun::set_hamiltonian_elements(u64 k, double value, u64 new_idx)
{
}


/// @brief Method to create hamiltonian within the class
void QuadQSun::create_hamiltonian()
{
    // this->_seed = std::abs(2 * (long)this->_seed - 10000) % ULONG_MAX;
    // disorder_generator = disorder<double>(this->_seed);

    const size_t dim_loc = ( (this->num_of_spins) );
	const size_t dim_erg = ( (this->grain_size) );

    this->H = sparse_matrix(this->dim, this->dim);
    this->_disorder = disorder_generator.uniform(this->num_of_spins, this->_hz - this->_w, this->_hz + this->_w);
    // std::cout << "AAAAA: " << this->_disorder.t() << std::endl;
	
    /* Create random neighbours for coupling hamiltonian */
    this->random_neigh = this->neighbor_generator.uniform(this->num_of_spins, 0, this->grain_size - 1);

	/* Create GOE Matrix */
	this->H_grain = this->_gamma * this->grain.generate_matrix(dim_erg);
    // if(this->_norm_grain)
    // this->H_grain /= std::sqrt((this->grain_size) + 1);
    this->H_grain /= arma::trace(this->H_grain * this->H_grain) / double(dim_erg);
    
    try_realloc_matrix(this->H_grain, dim, dim);
    /* Create random couplings */
    this->_long_range_couplings = arma::vec(this->num_of_spins, arma::fill::zeros);
    if(this->_alfa > 0){
        
        if( std::abs(this->_alfa - 0) < 1e-10){
            this->_long_range_couplings = arma::vec(this->num_of_spins, arma::fill::ones);
            //this->_disorder = arma::sort(this->_disorder, "ascend");
            {
                auto permut = sort_permutation(this->_disorder, [](const double a, const double b)
                                    { return std::abs(a) < std::abs(b); });
                apply_permutation(this->_disorder, permut);
            }
        } else {
            double u_j = 1 + disorder_generator.uniform_dist<double>(-this->_zeta, this->_zeta);
            this->_long_range_couplings(0) = this->_initiate_avalanche? 1.0 : std::pow(this->_alfa, u_j);
            for (int j = 1; j < this->num_of_spins; j++){
                int pos = j + 1 - (int)this->_initiate_avalanche; // if initiate avalanche next coupling alfa, not alfa^2
                double u_j = pos + disorder_generator.uniform_dist<double>(-this->_zeta, this->_zeta);
                // this->_long_range_couplings(j) = std::pow(this->_alfa, u_j);
                this->_long_range_couplings(j) = 1. / std::pow(u_j, this->_alfa);
            }
        }
    }
    _extra_debug(
	    std::cout << "disorder: \t\t" << this->_disorder.t() << std::endl;   
	    std::cout << "couplings: \t\t" << this->_long_range_couplings.t() << std::endl;
	    std::cout << "random_neigh: \t\t" << random_neigh.t() << std::endl;
        std::cout << "Grain matrix: \t\t" << H_grain << std::endl;
    )

    /* Generate coupling and spin hamiltonian */
    clk::time_point start = std::chrono::system_clock::now();
    for (int j = this->grain_size; j < this->system_size; j++)  // sum over spin d.o.f
    {
        const int pos_in_array = j - this->grain_size;                // array index of localised spin

        /* disorder on localised spins */
        this->H(j, j) += this->_disorder(pos_in_array);

        /* hopping on localized sites? */

        /* coupling of localised spins to GOE grain */
        int nei = random_neigh(pos_in_array);
        this->H(nei, j) += this->_J * this->_long_range_couplings(pos_in_array);
        this->H(j, nei) += this->_J * this->_long_range_couplings(pos_in_array);
    }
    std::cout << " - - - - - - finished Hamiltonian in : " << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
    
	this->H = this->H + arma::sp_mat(this->H_grain);
    // std::cout << "\n" << arma::mat(this->H) << std::endl;
}


/// @brief Method to create hamiltonian within the class
/// @param site site index where the local hamiltonian acts
/// @return the local hamiltonian at site site
typename QuadQSun::sparse_matrix QuadQSun::create_local_hamiltonian(int site)
{
    sparse_matrix H_local(dim, dim);
    
    return H_local;
}


//<! ------------------------------------------------------------------------------ OVVERRIDEN OPERATORS AND OPERATOR KERNELS
/// @brief Read model parameters from input stream
/// @tparam U1_sector U(1) symmetry sector as teamplate input 
/// @param os input stream to read parameters
std::istream& QuadQSun::read(std::istream& os)
{
    
    return os;
}

/// @brief Write hamiltonian to stream as human readable
/// @param os input stream to read parameters
std::ostream& QuadQSun::write(std::ostream& os) const
{
    printSeparated(os, "\t", 16, true, "Model:", "Quantum Sun model - O-dimensional EBT toy model");
    os << std::endl;
    printSeparated(os, "\t", 16, true, "Hamiltonian:", "H = \u03A3_{ij \in grain} A_{ij} c_i^+ c_j + J\u03A3_i \u03B1^{u_i} [ c_i^+ c_{ni} + h.c. ] + \u03A3_i h_i  c_i^+ c_i\t\t u_j in [j - \u03B6, j + \u03B6]");
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
    printSeparated(os, "\t", 16, true, "Parameters:");
    printSeparated(os, "\t", 16, true, "L,", this->system_size);
    printSeparated(os, "\t", 16, true, "grain size,", this->grain_size);

    printSeparated(os, "\t", 16, true, "J,", this->_J);
    printSeparated(os, "\t", 16, true, "\u03B1,", this->_alfa);
    printSeparated(os, "\t", 16, true, "\u03B3,", this->_gamma);
    printSeparated(os, "\t", 16, true, "w,", this->_w);
    printSeparated(os, "\t", 16, true, "hz,", this->_hz);
    printSeparated(os, "\t", 16, true, "\u03B6,", this->_zeta);
    //printSeparated(os, "\t", 16, true, "disorder", this->_disorder.t());

    printSeparated(os, "\t", 16, true, "    seed", this->_seed);
    
    return os;
}
