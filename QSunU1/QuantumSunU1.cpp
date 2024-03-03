
#include "includes/config.hpp"
#include "../include/QHamSolver.h"
#include "includes/QuantumSunU1.hpp"

//<! ---------------------------------------------------------------------------------------------------------------------------------------
//<! ------------------------------------------------------------------------------------------------------------------------ IMPLEMENTATION

//<! ------------------------------------------------------------------------------ CONSTRUCTORS
/// @brief Constructor of Quantum Sun model
/// @param L system size (L = L_loc + M)
/// @param J coupling of grain to localized spins
/// @param alfa regulates decay of coupling to furthest spins
/// @param w bandwidth of disorder on localized spins
/// @param hz uniform magnetic field
/// @param _Sz magnetization sector
/// @param seed random seed
/// @param N size of ergodic grain
/// @param zeta random positions for coupling
/// @param initiate_avalanche boolean value if initiate avalanche by hand (put fisrt coupling without decay)
/// @param normalize_grain normalize grain to unit hilbert-schmidt norm?
QuantumSunU1::QuantumSunU1(int L, double J, double alfa, double gamma,
            double w, double hz, double _Sz, const u64 seed, int N, double zeta, bool initiate_avalanche, bool normalize_grain )
{ 
    CONSTRUCTOR_CALL;

    this->num_of_spins = L; 
    this->grain_size = N;
    this->system_size = this->num_of_spins + this->grain_size;
    this->Sz = _Sz;

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
QuantumSunU1::QuantumSunU1(std::istream& os)
    { os >> *this; }

//<! ------------------------------------------------------------------------------ HAMILTONIAN BUILDERS
/// @brief Set hamiltonian matrix element given with value and new index
/// @param k current basis state
/// @param value value of matrix element
/// @param new_idx new index to be found in hilbert space
void QuantumSunU1::set_hamiltonian_elements(u64 k, double value, u64 new_idx)
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
void QuantumSunU1::create_hamiltonian()
{
    this->H = sparse_matrix(this->dim, this->dim);
    this->_disorder = disorder_generator.uniform(this->num_of_spins, this->_hz - this->_w, this->_hz + this->_w);
    
	
    /* Create random neighbours for coupling hamiltonian */
    auto random_neigh = this->neighbor_generator.uniform(this->num_of_spins, 0, this->grain_size - 1);

	/* Create GOE Matrix */
    bool normalized = false;
    const size_t dim_erg = ULLPOW( (this->grain_size) );
    arma::sp_mat H_grain(dim_erg, dim_erg);
    
    if(normalized){
        _assert_(false, "Not implemented normalized U(1) grain.. :'("); 
    } else
    {
        arma::mat H_grain_full = this->_gamma * this->grain.generate_matrix(dim_erg);
        H_grain_full = 0.3 * H_grain_full / std::sqrt(2);
        _extra_debug( std::cout << "Full GOE grain:\n" << H_grain_full << std::endl << arma::trace(H_grain_full * H_grain_full) / dim_erg << std::endl; )

        for(int M = 0; M <= this->grain_size; M++)
        {
            auto _grain_hilbert_space = U1Hilbert(this->grain_size, double(M) - this->grain_size / 2.0);
            for(int i = 0; i < _grain_hilbert_space.get_hilbert_space_size(); i++)
            {
                u64 state1 = _grain_hilbert_space(i);
                for(int j = 0; j < _grain_hilbert_space.get_hilbert_space_size(); j++)
                {
                    u64 state2 = _grain_hilbert_space(j);
                    H_grain(state1, state2) = H_grain_full(state1, state2);
                }
            }
        }
        _extra_debug( std::cout << "U1 block-symmetric GOE grain:\n" << arma::mat(H_grain) << std::endl << arma::trace(H_grain * H_grain) / dim_erg << std::endl; )
    }

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
            {
                auto permut = sort_permutation(this->_disorder, [](const double a, const double b)
                                    { return std::abs(a) < std::abs(b); });
                apply_permutation(this->_disorder, permut);
            }
        }
    }
    _extra_debug(
	    std::cout << "disorder: \t\t" << this->_disorder.t() << std::endl;   
	    std::cout << "couplings: \t\t" << this->_long_range_couplings.t() << std::endl;
	    std::cout << "random_neigh: \t\t" << random_neigh.t() << std::endl;
    )

    /* Generate hamiltonian */
    u64 mask_grain = ULLPOW(this->num_of_spins) * (ULLPOW(this->grain_size) - 1);
    u64 mask_spins = (ULLPOW(this->num_of_spins) - 1);
    _extra_debug(
        std::cout << "Check masks:" << std::endl;
        std::cout << "Mask grain:\t" << mask_grain << "\t"; for(int j = 0; j < this->system_size; j++) std::cout << int(0.5 + std::real(std::get<0>( Z(mask_grain, this->system_size, j) )) ); std::cout << std::endl;
        std::cout << "Mask spins:\t" << mask_spins << "\t"; for(int j = 0; j < this->system_size; j++) std::cout << int(0.5 + std::real(std::get<0>( Z(mask_spins, this->system_size, j) )) ); std::cout << std::endl;
    )
    // auto check_spin = QOps::__builtins::get_digit(this->system_size);
    for (u64 k = 0; k < this->dim; k++) 
    {
		u64 base_state = this->_hilbert_space(k);
        /* U(1) grain */
        const u64 grain_state = (base_state & mask_grain) / ULLPOW(this->num_of_spins);
        const u64 spins_state = base_state & mask_spins;
        // cout << "----------------\nbase: "; for(int j = 0; j < this->system_size; j++) std::cout << int(0.5 + std::real(std::get<0>( Z(base_state, this->system_size, j) )) ); std::cout << std::endl;
        for(int idx = 0; idx < dim_erg; idx++)
        {
            double value = H_grain.col(grain_state)(idx);
            if( std::abs(value) > 1e-15)
            {
                auto row = ULLPOW(this->num_of_spins) * idx + spins_state;
                this->set_hamiltonian_elements(k, value, row);
            }
        }
        /* localised spins and interaction */
		for (int j = this->grain_size; j < this->system_size; j++)  // sum over spin d.o.f
        {
            const int pos_in_array = j - this->grain_size;                // array index of localised spin
			
            /* disorder on localised spins */
            auto [Sz_k, tmp1] = Z(base_state, this->system_size, j);
            this->H(k, k) += this->_disorder(pos_in_array) * std::real(Sz_k);
            

			/* coupling of localised spins to GOE grain */
			int nei = random_neigh(pos_in_array);
            auto [Sz_nei, tmp2] = Z(base_state, this->system_size, nei);
            if(std::real(Sz_k * Sz_nei) < 0)
            {
                auto [val1, Sx_k] = X(base_state, this->system_size, j);
                auto [val2, SxSx_k] = X(Sx_k, this->system_size, nei);
                this->set_hamiltonian_elements(k, this->_J * this->_long_range_couplings(pos_in_array) * std::real(0.5), SxSx_k); // you stupid bitch, not S^2 but 1/2
            }
		}
	}
    // std::cout << arma::mat(this->H) << std::endl;
	// this->H = this->H + arma::kron(H_grain, arma::eye(dim_loc, dim_loc));
}


/// @brief Method to create hamiltonian within the class
/// @param site site index where the local hamiltonian acts
/// @return the local hamiltonian at site site
typename QuantumSunU1::sparse_matrix QuantumSunU1::create_local_hamiltonian(int site)
{
    sparse_matrix H_local(dim, dim);
    
    return H_local;
}


//<! ------------------------------------------------------------------------------ OVVERRIDEN OPERATORS AND OPERATOR KERNELS
/// @brief Read model parameters from input stream
/// @tparam U1_sector U(1) symmetry sector as teamplate input 
/// @param os input stream to read parameters
std::istream& QuantumSunU1::read(std::istream& os)
{
    
    return os;
}

/// @brief Write hamiltonian to stream as human readable
/// @param os input stream to read parameters
std::ostream& QuantumSunU1::write(std::ostream& os) const
{
    printSeparated(os, "\t", 16, true, "Model:", "U1 Quantum Sun model - O-dimensional EBT toy model");
    os << std::endl;
    printSeparated(os, "\t", 16, true, "Hamiltonian:", "H = R_U1 + J\u03A3_i \u03B1^{u_j} S^x_n_i S^x_i + S^y_n_i S^y_i + \u03A3_i h_i S^z_i");
    printSeparated(os, "\t", 16, true, "Disorder:", "h_i in [hz - w, hz + w]\t u_j in [j - \u03B6, j + \u03B6]");
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




        // arma::mat H_grain(dim_erg, dim_erg, arma::fill::zeros);
        // int counter = 0;
        // for(int M = 0; M <= this->grain_size; M++)
        // {
        //     const size_t dim_erg_M = binom(this->grain_size, M);
        //     arma::mat H_grain_M = this->_gamma * this->grain.generate_matrix(dim_erg_M);
        //     H_grain_M /= std::sqrt(dim_erg_M + 1);
        //     std::cout << H_grain_M << std::endl << arma::trace(H_grain_M * H_grain_M) / dim_erg_M << std::endl;

        //     H_grain.submat(counter, counter, counter + dim_erg_M - 1, counter + dim_erg_M - 1) += H_grain_M;
        //     counter += dim_erg_M;
        // }
        // // H_grain /= std::sqrt(this->grain_size + 1);
        // H_grain /= std::sqrt(dim_erg + 1) / std::sqrt(this->grain_size + 1);
        // std::cout << H_grain << std::endl << arma::trace(H_grain * H_grain) << std::endl;

        
// const u64 grain_state = base_state & mask_grain;
        // const u64 spins_state = base_state & mask_spins;
        // arma::sp_mat::const_col_iterator it     = H_grain.begin_col(grain_state);
        // arma::sp_mat::const_col_iterator it_end = H_grain.end_col(grain_state);
        // cout << "base: "; for(int j = 0; j < this->system_size; j++) std::cout << int(0.5 + std::real(std::get<0>( Z(base_state, this->system_size, j) )) ); std::cout << std::endl;
        // for(; it != it_end; ++it)
        // {
        //     if( std::abs(*it) > 1e-15){
        //         auto row = ULLPOW(this->num_of_spins) * (it.row()) + spins_state;
        //         this->set_hamiltonian_elements(k, (*it), row);
        //         // this->H(row, base_state) += (*it);
        //         cout << "val: " << (*it)    << endl;
        //         cout << "row: " << it.row() << endl;
        //         cout << "col: " << it.col() << endl;
        //     }
        // }


                // std::cout << "row: " << row << std::endl;
                // std::cout << "col: " << grain_state << std::endl;
                // std::cout << "val: " << value    << std::endl;