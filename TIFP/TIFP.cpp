#include "includes/config.hpp"
#include "../include/QHamSolver.h"
#include "includes/TIFP.hpp"


//<! ---------------------------------------------------------------------------------------------------------------------------------------
//<! ------------------------------------------------------------------------------------------------------------------------ IMPLEMENTATION

//<! ------------------------------------------------------------------------------ CONSTRUCTORS

/// @brief Initialize model dependencies (symetries, hilbert space, ...)
void TIFP::init()
{   
    // set symmetry generators
    this->set_symmetry_generators();

    // initialize hilbert space
    if(this->_use_symmetries){
        this->_hilbert_space = QHS::point_symmetric( this->system_size, 
                                            this->symmetry_generators, 
                                            this->_boundary_condition,
                                            this->syms.k_sym,
                                            -1
                                            );
    } else {
        this->_hilbert_space = QHS::point_symmetric( this->system_size, v_1d<QOps::genOp>(), 1, 0, 0 );
    }
    this->dim = this->_hilbert_space.get_hilbert_space_size();

    // create hamiltonian
    this->create_hamiltonian();
}

/// @brief Constructor for TIFP model class
/// @param _BC boundary condition
/// @param L system size
/// @param J model parameter
/// @param ksym quasimomentum symmetry sector
/// @param rsym other Z_2 symmetry sector
/// @param zzsym spin flip in Z symmetry sector
/// @param use_syms use symmetric code?
TIFP::TIFP(int _BC, unsigned int L, double J, int ksym, int rsym, int zzsym, bool use_syms)
{ 
    CONSTRUCTOR_CALL;

    this->_boundary_condition = _BC;
    this->system_size = L; 
    this->_J = J;
    
    this->_use_symmetries = use_syms;

    //<! symmetries
    this->syms.k_sym = ksym;
    this->syms.r_sym = rsym;
    this->syms.zz_sym = zzsym;
    #ifdef USE_REAL_SECTORS
        if(this->_boundary_condition == 0 && this->_use_symmetries){ // only for PBC
            bool is_k_sector_real = (std::abs(two_pi * ksym / this->system_size) < 1e-4) || (std::abs(two_pi * ksym / this->system_size - pi) < 1e-4);
            _assert_(is_k_sector_real, NOT_ALLOWED_SYM_SECTOR "\n\t\tMatrix type is real due to USE_REAL_SECTORS macro, but quasimomentum sector is complex, i.e. k != 0, pi");
        }
    #endif
    this->init(); 
}

/// @brief Constructor from input stream
/// @param os input stream
TIFP::TIFP(std::istream& os)
    { os >> *this; }

/// @brief Set symmetry generators (among spin flips if fields perpendicular to spin axis are 0)
void TIFP::set_symmetry_generators()
{   
    if( this->_use_symmetries)
    {

        // spin flips (only for even L both can be used)
        this->symmetry_generators.emplace_back(QOps::_spin_flip_z_symmetry(this->system_size, this->syms.zz_sym));

        // other Z_2 symmetry
        // if(this->system_size % 4 == 0){
        //     auto I = operators::sigma_0;
        //     auto X = operators::sigma_x;
        //     auto Y = operators::sigma_y;
        //     auto Z = operators::sigma_z;
        //     //<! for now vector with Lmax = 31 elements
        //     std::vector<decltype(X)> Quant_ops = { Z, Y, I, X, Z, Y, I, X, Z, Y, I, X, Z, Y, I, X, Z, Y, I, X, Z, Y, I, X, Z, Y, I, X, Z, Y, I};
        //     const int _L = this->system_size;
        //     auto op_kernel = [_L, Quant_ops](u64 state){
        //         cpx val = 1.0;
        //         cpx res = 1.0;
        //         u64 new_state = state;
        //         for(int i = _L - 1; i >= 0; i--){
        //             std::tie(res, new_state) = Quant_ops[ i ](new_state, _L, i);
        //             val *= res;
        //         }
        //         return std::make_pair(new_state, val);
        //     };
        //     this->symmetry_generators.emplace_back( QOps::generic_operator<>(this->system_size, op_kernel, this->syms.r_sym) );
        // }

    }
}

//<! ------------------------------------------------------------------------------ HAMILTONIAN BUILDERS
/// @brief Set hamiltonian matrix element given with value and new index
/// @param k current basis state
/// @param value value of matrix element
/// @param new_idx new index to be found in hilbert space
void TIFP::set_hamiltonian_elements(u64 k, elem_ty value, u64 new_idx)
{   
    u64 state;
    elem_ty sym_eig;

    try {
        std::tie(state, sym_eig) = this->_hilbert_space.find_matrix_element(new_idx, this->_hilbert_space.get_norm(k));
        #ifdef USE_REAL_SECTORS
            H(state, k) += std::real(value * sym_eig);
        #else
            H(state, k) += value * sym_eig;
        #endif
    } 
    catch (const std::exception& err) {
        std::cout << "Exception:\t" << err.what() << "\n";
        std::cout << "SHit ehhh..." << std::endl;
        printSeparated(std::cout, "\t", 14, true, new_idx, this->_hilbert_space(k), value, sym_eig);
    }
}

/// @brief Method to create hamiltonian within the class
void TIFP::create_hamiltonian()
{
    this->H = sparse_matrix(this->dim, this->dim);
    
    auto X = operators::sigma_x;
    auto Y = operators::sigma_y;
    auto Z = operators::sigma_z;

    for (size_t k = 0; k < this->dim; k++) {
		int base_state = this->_hilbert_space(k);
	    for (int j = 0; j < this->system_size - 2 * int(this->_boundary_condition); j++) {
            
            
            auto [Sz_j2, tmp] = Z(base_state, this->system_size, (j+2) % this->system_size);

            //<! first term X_j X_j+1 Z_j+2
            auto [tmp1, X_state] = X(base_state, this->system_size, (j+1) % this->system_size);
            auto [tmp2, XX_state] = X(X_state, this->system_size, j);
            this->set_hamiltonian_elements(k, std::real(Sz_j2), XX_state);

            //<! second term Z_j Y_j+1 Y_j+2
            auto [val1, Y_state] = Y(base_state, this->system_size, (j+2) % this->system_size);
            auto [val2, YY_state] = Y(Y_state, this->system_size, (j+1) % this->system_size);
            auto [Sz_jYY, tmp3] = Z(YY_state, this->system_size, j);
            this->set_hamiltonian_elements(k, this->_J * this->_J * std::real(Sz_jYY * val1 * val2), YY_state);
            
            //<! third term Z_j Z_j+2
            auto [Sz_j, tmp4] = Z(base_state, this->system_size, j);
            this->H(k, k) += this->_J * std::real(Sz_j * Sz_j2);

	    }
	}
}


/// @brief Method to create hamiltonian within the class
/// @param site site index where the local hamiltonian acts
/// @return the local hamiltonian at site site
typename TIFP::sparse_matrix TIFP::create_local_hamiltonian(int site)
{
    sparse_matrix H_local(dim, dim);
    
    return H_local;
}


//<! ------------------------------------------------------------------------------ OVVERRIDEN OPERATORS AND OPERATOR KERNELS
/// @brief Read model parameters from input stream
/// @tparam U1_sector U(1) symmetry sector as teamplate input 
/// @param os input stream to read parameters
std::istream& TIFP::read(std::istream& os)
{
    
    return os;
}

/// @brief Write hamiltonian to stream as human readable
/// @param os input stream to read parameters
std::ostream& TIFP::write(std::ostream& os) const
{
    printSeparated(os, "\t", 16, true, "Model:", "TIFP spin chain");
    os << std::endl;
    printSeparated(os, "\t", 16, true, "Hamiltonian:", "H = \u03A3_i [ X_i X_i+1 Z_i+2 + J^2 Z_i Y_i+1 Y_i+2 + J Z_i Z_i+2 ]");
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
    printSeparated(os, "\t", 16, true, "Parameters:");
    printSeparated(os, "\t", 16, true, "L", this->system_size);
    printSeparated(os, "\t", 16, true, "J", this->_J);
    
    printSeparated(os, "\t", 16, true, "k", this->syms.k_sym);
    printSeparated(os, "\t", 16, true, "r", this->syms.r_sym);
    printSeparated(os, "\t", 16, true, "zz", this->syms.zz_sym);

    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");

    return os;
}



//<! ------------------------------------------------------------------------------ ADDITIONAL METHODS FOR SYMMETRIC HAMILTONIAN



