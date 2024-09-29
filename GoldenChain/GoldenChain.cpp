#include "includes/config.hpp"
#include "../include/QHamSolver.h"
#include "includes/GoldenChain.hpp"


//<! ---------------------------------------------------------------------------------------------------------------------------------------
//<! ------------------------------------------------------------------------------------------------------------------------ IMPLEMENTATION

//<! ------------------------------------------------------------------------------ CONSTRUCTORS

/// @brief Initialize model dependencies (symetries, hilbert space, ...)
void GoldenChain::init()
{   
    // set symmetry generators
    this->set_symmetry_generators();

    // initialize hilbert space
    auto Translate = QOps::__builtins::translation(this->system_size, 1);
    auto flip = QOps::__builtins::spin_flip_x(this->system_size);
    auto some_kernel = [&Translate, &flip](u64 n){
        n = std::get<0>(flip(n));
        return !( (n) & std::get<0>( Translate(n) ) );
    };

    int parity_pos = (std::abs(this->_c) > 1e-14)? -1 : 0;
    std::cout << "Number of generators:\t" << this->symmetry_generators.size() << "\tuse syms=\t" << this->_use_symmetries << std::endl;

    auto _hilbert_GoldenChain = QHS::constrained_hilbert_space(this->system_size, std::move(some_kernel));
    auto _second_hilbert = this->_use_symmetries? QHS::point_symmetric( this->system_size, this->symmetry_generators, this->_boundary_condition, this->syms.k_sym, parity_pos) :\
                                                    QHS::point_symmetric( this->system_size, v_1d<QOps::genOp>(), 1, 0, -1);
    
    this->_hilbert_space = tensor(_second_hilbert, _hilbert_GoldenChain);
    // this->_hilbert_space = _second_hilbert;
    this->dim = this->_hilbert_space.get_hilbert_space_size();
    
    // create hamiltonian
    this->create_hamiltonian();
}

/// @brief Constructor for GoldenChain model class
/// @param _BC boundary condition
/// @param L system size
/// @param J model parameter
/// @param c coupling to Q5
/// @param ksym quasimomentum symmetry sector
/// @param rsym other Z_2 symmetry sector
/// @param zzsym spin flip in Z symmetry sector
/// @param use_syms use symmetric code?
GoldenChain::GoldenChain(int _BC, unsigned int L, double J, double c, int ksym, int psym, bool use_syms)
{ 
    CONSTRUCTOR_CALL;

    this->_boundary_condition = _BC;
    this->system_size = L; 
    this->_J = J;
    this->_c = c;
    
    this->_use_symmetries = use_syms;
    
    //<! symmetries
    this->syms.k_sym = ksym;
    this->syms.p_sym = psym;
    this->init(); 
}

/// @brief Constructor from input stream
/// @param os input stream
GoldenChain::GoldenChain(std::istream& os)
    { os >> *this; }

/// @brief Set symmetry generators (among spin flips if fields perpendicular to spin axis are 0)
void GoldenChain::set_symmetry_generators()
{   
    // parity symmetry
    if(this->_c == 0)
        this->symmetry_generators.emplace_back(QOps::_parity_symmetry(this->system_size, this->syms.p_sym));
}

//<! ------------------------------------------------------------------------------ HAMILTONIAN BUILDERS
/// @brief Set hamiltonian matrix element given with value and new index
/// @param k current basis state
/// @param value value of matrix element
/// @param new_idx new index to be found in hilbert space
void GoldenChain::set_hamiltonian_elements(u64 k, elem_ty value, u64 new_idx)
{   
    u64 state;
    elem_ty sym_eig;

    try {
        std::tie(state, sym_eig) = this->_hilbert_space.find_matrix_element(new_idx, this->_hilbert_space.get_norm(k));
        
        H(state, k) += value * std::conj(sym_eig);
    } 
    catch (const std::exception& err) {
        std::cout << "Exception:\t" << err.what() << "\n";
        std::cout << "SHit ehhh..." << std::endl;
        printSeparated(std::cout, "\t", 14, true, new_idx, this->_hilbert_space(k), value, sym_eig);
    }
}

/// @brief Method to create hamiltonian within the class
void GoldenChain::create_hamiltonian()
{
    this->H = sparse_matrix(this->dim, this->dim);
	for (int j = 0; j < this->system_size - 2 * int(this->_boundary_condition); j++) 
        this->H += this->create_local_hamiltonian(j);
}


/// @brief Method to create hamiltonian within the class
/// @param site site index where the local hamiltonian acts
/// @return the local hamiltonian at site site
typename GoldenChain::sparse_matrix GoldenChain::create_local_hamiltonian(int site)
{
    const double fi = (1.0 + std::sqrt(5.0)) / 2.0;
    const double par1 = 1. / std::sqrt(fi * fi * fi);
    const double par2 = 1. / (fi);
    const double par3 = 1. / (fi * fi);

    sparse_matrix H_local(dim, dim);
    auto set_loc_hamiltonian_elements = [this, &H_local](u64 k, elem_ty value, u64 new_idx){
        u64 state;
        elem_ty sym_eig;
        std::tie(state, sym_eig) = this->_hilbert_space.find_matrix_element(new_idx, this->_hilbert_space.get_norm(k) );
        // auto vec1 = boost::dynamic_bitset<>(this->system_size, new_idx);
        // auto vec2 = boost::dynamic_bitset<>(this->system_size, this->_hilbert_space(k));
        // printSeparated(std::cout, "\t", 14, true, vec1, vec2,  value, sym_eig);
        // if( std::abs(sym_eig) > 1e-10 && std::abs(value) > 1e-10 )
        H_local(state, k) += value * std::conj(sym_eig);
    };
    if(this->_boundary_condition && site >= this->system_size - 2){
        return sparse_matrix(dim, dim);
    } else {
        for (size_t k = 0; k < this->dim; k++) {
            int base_state = this->_hilbert_space(k);
            // auto vec1 = boost::dynamic_bitset<>(this->system_size, base_state);
                
            const double Sz_j  = -std::real( std::get<0>( Z(base_state, this->system_size, (site) % this->system_size) ) );
            const double Sz_j1 = -std::real( std::get<0>( Z(base_state, this->system_size, (site + 1) % this->system_size) ) );
            const double Sz_j2 = -std::real( std::get<0>( Z(base_state, this->system_size, (site + 2) % this->system_size) ) );
            const double Sz_j3 = -std::real( std::get<0>( Z(base_state, this->system_size, (site + 3) % this->system_size) ) );
            
            //<! ------------------------------------------------------------------------------ h_j - local hamiltonian
            //<! first term P_j X_j+1 P_j+2
            if(Sz_j < 0 && Sz_j2 < 0){
                auto [tmp1, X_state] = X(base_state, this->system_size, (site + 1) % this->system_size);
                set_loc_hamiltonian_elements(k, -par1 * tmp1, X_state);
            }
            //<! second term P_j P_j+1 P_j+2
            if(Sz_j < 0 && Sz_j1 < 0 && Sz_j2 < 0)
                H_local(k, k) -= par2;
            
            // //<! third term P_j N_j+1 P_j+2
            if(Sz_j < 0 && Sz_j1 > 0 && Sz_j2 < 0)
                H_local(k, k) -= par3;
            
            // //<! fourth term N_j P_j+1 N_j+2
            if(Sz_j > 0 && Sz_j1 < 0 && Sz_j2 > 0)
                H_local(k, k) -= 1.0;

            //<! ------------------------------------------------------------------------------ q4_j - local Q4 charge
            //<! first term P_j X_j+1 (P_j+1 + P_j+2 - 1) X_j+2 P_j+3
            if(Sz_j < 0 && (Sz_j1 * Sz_j2 > 0) && Sz_j3 < 0){
                auto [tmp1, X_state] = X(base_state, this->system_size, (site + 1) % this->system_size);
                auto [tmp2, XX_state] = X(X_state, this->system_size, (site + 2) % this->system_size);
                set_loc_hamiltonian_elements(k, 1i * this->_c * par1 * par1 * tmp1 * tmp2 * (1 - Sz_j1 - Sz_j2), XX_state);
            }
            //<! second term P_j X_j+1 Z_j+1 P_j+2 P_j+3
            if(Sz_j < 0 && Sz_j2 < 0 && Sz_j3 < 0){
                auto [tmp1, X_state] = X(base_state, this->system_size, (site + 1) % this->system_size);
                set_loc_hamiltonian_elements(k, -1i * this->_c * par1 * par2 * tmp1 * Sz_j1, X_state);
            }
            //<! third term P_j X_j+1 Z_j+1 P_j+2 N_j+3
            if(Sz_j < 0 && Sz_j2 < 0 && Sz_j3 > 0){
                auto [tmp1, X_state] = X(base_state, this->system_size, (site + 1) % this->system_size);
                set_loc_hamiltonian_elements(k, 1i * this->_c * par1 * tmp1 * Sz_j1, X_state);
            }
            //<! fourth term P_j P_j+1 X_j+2 Z_j+2 P_j+3
            if(Sz_j < 0 && Sz_j1 < 0 && Sz_j3 < 0){
                auto [tmp1, X_state] = X(base_state, this->system_size, (site + 2) % this->system_size);
                set_loc_hamiltonian_elements(k, 1i * this->_c * par1 * par2 * tmp1 * Sz_j2, X_state);
            }
            //<! fifth term N_j P_j+1 X_j+2 Z_j+2 P_j+3
            if(Sz_j > 0 && Sz_j1 < 0 && Sz_j3 < 0){
                auto [tmp1, X_state] = X(base_state, this->system_size, (site + 2) % this->system_size);
                set_loc_hamiltonian_elements(k, -1i * this->_c * par1 * tmp1 * Sz_j2, X_state);
            }
        }

        return H_local;
    }
}


//<! ------------------------------------------------------------------------------ OVVERRIDEN OPERATORS AND OPERATOR KERNELS
/// @brief Read model parameters from input stream
/// @tparam U1_sector U(1) symmetry sector as teamplate input 
/// @param os input stream to read parameters
std::istream& GoldenChain::read(std::istream& os)
{
    
    return os;
}

/// @brief Write hamiltonian to stream as human readable
/// @param os input stream to read parameters
std::ostream& GoldenChain::write(std::ostream& os) const
{
    printSeparated(os, "\t", 16, true, "Model:", "GoldenChain spin chain");
    os << std::endl;
    printSeparated(os, "\t", 16, true, "Hamiltonian:", "H = \u03A3_i [ X_i X_i+1 Z_i+2 + J^2 Z_i Y_i+1 Y_i+2 + J Z_i Z_i+2 ]");
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
    printSeparated(os, "\t", 16, true, "Parameters:");
    printSeparated(os, "\t", 16, true, "L", this->system_size);
    printSeparated(os, "\t", 16, true, "J", this->_J);
    printSeparated(os, "\t", 16, true, "c", this->_c);
    
    printSeparated(os, "\t", 16, true, "k", this->syms.k_sym);
    printSeparated(os, "\t", 16, true, "p", this->syms.p_sym);

    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");

    return os;
}



//<! ------------------------------------------------------------------------------ ADDITIONAL METHODS FOR SYMMETRIC HAMILTONIAN



