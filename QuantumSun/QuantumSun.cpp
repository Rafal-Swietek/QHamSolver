
#include "includes/config.hpp"
#include "../include/QHamSolver.h"
#include "includes/QuantumSun.hpp"

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
QuantumSun::QuantumSun(int L, double J, double alfa, double gamma,
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
    if constexpr (scaled_disorder == 1)
        this->_w = w * this->num_of_spins / 2.0;
    else
        this->_w = w;
    
    this->_seed = seed;

    this->_initiate_avalanche = initiate_avalanche;
    this->_norm_grain = normalize_grain;
    init(); 
}

/// @brief Constructor from input stream
/// @param os input stream
QuantumSun::QuantumSun(std::istream& os)
    { os >> *this; }

//<! ------------------------------------------------------------------------------ HAMILTONIAN BUILDERS
/// @brief Set hamiltonian matrix element given with value and new index
/// @param k current basis state
/// @param value value of matrix element
/// @param new_idx new index to be found in hilbert space
void QuantumSun::set_hamiltonian_elements(u64 k, double value, u64 new_idx)
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
void QuantumSun::create_hamiltonian()
{
    // this->_seed = std::abs(2 * (long)this->_seed - 10000) % ULONG_MAX;
    // disorder_generator = disorder<double>(this->_seed);

    const size_t dim_loc = ULLPOW( (this->num_of_spins) );
	const size_t dim_erg = ULLPOW( (this->grain_size) );

    this->H = sparse_matrix(this->dim, this->dim);
    if constexpr (conf_disorder == 1)
        this->_disorder = disorder_generator.uniform(dim_loc, this->_hz - this->_w, this->_hz + this->_w);
    else
        this->_disorder = disorder_generator.uniform(this->num_of_spins, this->_hz - this->_w, this->_hz + this->_w);
    
	
    /* Create random neighbours for coupling hamiltonian */
    auto random_neigh = this->neighbor_generator.uniform(this->num_of_spins, 0, this->grain_size - 1);

	/* Create GOE Matrix */
	arma::mat H_grain = this->_gamma * this->grain.generate_matrix(dim_erg);
    // if(this->_norm_grain)
    H_grain /= std::sqrt(ULLPOW(this->grain_size) + 1);

    /* Create random couplings */
    this->_long_range_couplings = arma::vec(this->num_of_spins, arma::fill::zeros);
    if(this->_alfa > 0){
        if(this->_alfa < 1.0){
            double u_j = 1 + disorder_generator.uniform_dist<double>(-this->_zeta, this->_zeta);
            this->_long_range_couplings(0) = this->_initiate_avalanche? 1.0 : std::pow(this->_alfa, u_j);
            for (int j = 1; j < this->num_of_spins; j++){
                int pos = j + 1 - (int)this->_initiate_avalanche; // if initiate avalanche next coupling alfa, not alfa^2
                double u_j = pos + disorder_generator.uniform_dist<double>(-this->_zeta, this->_zeta);
                this->_long_range_couplings(j) = std::pow(this->_alfa, u_j);
            }
        } else {
            this->_long_range_couplings = arma::vec(this->num_of_spins, arma::fill::ones);
            //this->_disorder = arma::sort(this->_disorder, "ascend");
            if constexpr (conf_disorder == 0){
                auto permut = sort_permutation(this->_disorder, [](const double a, const double b)
                                    { return std::abs(a) < std::abs(b); });
                apply_permutation(this->_disorder, permut);
            }
        }
    }
    #ifdef EXTRA_DEBUG
	    std::cout << "disorder: \t\t" << this->_disorder.t() << std::endl;   
	    std::cout << "couplings: \t\t" << this->_long_range_couplings.t() << std::endl;
	    std::cout << "random_neigh: \t\t" << random_neigh.t() << std::endl;
        std::cout << "Grain matrix: \t\t" << H_grain << std::endl;
    #endif

    /* Generate coupling and spin hamiltonian */
    for (u64 k = 0; k < this->dim; k++) {
		u64 base_state = this->_hilbert_space(k);
		for (int j = this->grain_size; j < this->system_size; j++)  // sum over spin d.o.f
        {
            const int pos_in_array = j - this->grain_size;                // array index of localised spin

			/* disorder on localised spins */
            if constexpr (conf_disorder == 0){
                auto [val, Sz_k] = operators::sigma_z(base_state, this->system_size, j);
			    this->set_hamiltonian_elements(k, this->_disorder(pos_in_array) * real(val), Sz_k);
            }

			/* coupling of localised spins to GOE grain */
			int nei = random_neigh(pos_in_array);
		    auto [val1, Sx_k] = operators::sigma_x(base_state, this->system_size, j);
		    auto [val2, SxSx_k] = operators::sigma_x(Sx_k, this->system_size, nei);
			this->set_hamiltonian_elements(k, this->_J * this->_long_range_couplings(pos_in_array) * real(val1 * val2), SxSx_k);
		}
	}
    if constexpr (conf_disorder == 1){
        arma::mat H_loc = arma::kron(arma::eye(dim_erg, dim_erg), arma::mat(arma::diagmat(this->_disorder)));
        this->H = this->H + H_loc;
    }
	this->H = this->H + arma::kron(H_grain, arma::eye(dim_loc, dim_loc));
}


/// @brief Method to create hamiltonian within the class
/// @param site site index where the local hamiltonian acts
/// @return the local hamiltonian at site site
typename QuantumSun::sparse_matrix QuantumSun::create_local_hamiltonian(int site)
{
    sparse_matrix H_local(dim, dim);
    
    return H_local;
}


//<! ------------------------------------------------------------------------------ OVVERRIDEN OPERATORS AND OPERATOR KERNELS
/// @brief Read model parameters from input stream
/// @tparam U1_sector U(1) symmetry sector as teamplate input 
/// @param os input stream to read parameters
std::istream& QuantumSun::read(std::istream& os)
{
    
    return os;
}

/// @brief Write hamiltonian to stream as human readable
/// @param os input stream to read parameters
std::ostream& QuantumSun::write(std::ostream& os) const
{
    printSeparated(os, "\t", 16, true, "Model:", "Quantum Sun model - O-dimensional EBT toy model");
    os << std::endl;
    printSeparated(os, "\t", 16, true, "Hamiltonian:", "H = R + J\u03A3_i \u03B1^{u_j} S^x_i S^x_i+1 + \u03A3_i h_i S^z_i\t\t u_j in [j - \u03B6, j + \u03B6]");
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
