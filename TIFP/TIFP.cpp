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
        this->_hilbert_space = QHS::point_symmetric( this->system_size, this->symmetry_generators, 1, 0, -1);
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
/// @param c coupling to Q5
/// @param ksym quasimomentum symmetry sector
/// @param rsym other Z_2 symmetry sector
/// @param zzsym spin flip in Z symmetry sector
/// @param use_syms use symmetric code?
TIFP::TIFP(int _BC, unsigned int L, double J, double c, int zzsym, int z1sym, int z2sym, bool use_syms)
{ 
    CONSTRUCTOR_CALL;

    this->_boundary_condition = _BC;
    this->system_size = L; 
    this->_J = J;
    this->_c = c;
    
    this->_use_symmetries = use_syms;

    //<! symmetries
    this->syms.z1_sym = z1sym;
    this->syms.z2_sym = z2sym;
    this->syms.zz_sym = zzsym;
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
        const int _L = this->system_size;
        if(this->system_size % 2 == 0)
        {
            auto C0C2 = [_L](u64 state)
                {
                    cpx val = 1.0;
                    for(int i = 0; i < _L; i += 2){
                        cpx res = 1.0;
                        std::tie(res, state) = X(state, _L, i);
                        val *= res;
                        if(i + 1 < _L) 
                            std::tie(res, state) = Y(state, _L, i + 1);
                        val *= res;
                    }
                    return std::make_pair(state, val);
                };
            auto C0C3 = [_L](u64 state)
                {
                    cpx val = 1.0;
                    for(int i = 0; i < _L; i++){
                        if( i%4 != 2 ){
                            cpx res = 1.0;
                            auto op = (i % 4 == 0)? Z : ( (i % 4 == 1)? Y : X);
                            std::tie(res, state) = op(state, _L, i);
                            val *= res;
                        }
                    }
                    return std::make_pair(state, val);
                };
            this->symmetry_generators.emplace_back( QOps::generic_operator<>(this->system_size, C0C2, this->syms.z1_sym) );
            if(this->system_size % 4 == 0)
                this->symmetry_generators.emplace_back( QOps::generic_operator<>(this->system_size, C0C3, this->syms.z2_sym) );
        }

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
        // #ifdef USE_REAL_SECTORS
        //     H(state, k) += std::real(value * sym_eig);
        // #else
        //     H(state, k) += value * sym_eig;
        // #endif
        H(state, k) += value * sym_eig;
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

	for (int j = 0; j < this->system_size - 2 * int(this->_boundary_condition); j++) 
    {
        //<! first of Q5 charge
        auto H_j = this->create_local_hamiltonian(j);
        auto H_j1 = this->create_local_hamiltonian(j+1);
        auto H_j2 = this->create_local_hamiltonian(j+2);
        this->H += this->_c * 1i * (H_j * (H_j1 + H_j2) - (H_j1 + H_j2) * H_j);

        for (size_t k = 0; k < this->dim; k++) {
            int base_state = this->_hilbert_space(k);
                
            auto [Sz_j2, tmp] = Z(base_state, this->system_size, (j+2) % this->system_size);

            //<! first term X_j X_j+1 Z_j+2
            auto [tmp1, X_state] = X(base_state, this->system_size, (j+1) % this->system_size);
            auto [tmp2, XX_state] = X(X_state, this->system_size, j);
            this->set_hamiltonian_elements(k, Sz_j2, XX_state);

            //<! second term Z_j Y_j+1 Y_j+2
            auto [Yval, Y_state] = Y(base_state, this->system_size, (j+2) % this->system_size);
            auto [YYval, YY_state] = Y(Y_state, this->system_size, (j+1) % this->system_size);
            auto [Sz_jYY, tmp3] = Z(YY_state, this->system_size, j);
            this->set_hamiltonian_elements(k, this->_J * this->_J * Sz_jYY * Yval * YYval, YY_state);
            
            //<! third term Z_j Z_j+2
            auto [Sz_j, tmp4] = Z(base_state, this->system_size, j);
            this->H(k, k) += this->_J * Sz_j * Sz_j2;

            //<! ----------------------- COUPLING TO Q5
            //<! first term Y_j X_j+1
            auto [Xval, _X_state] = X(base_state, this->system_size, (j + 1) % this->system_size);
            auto [YXval, YX_state] = Y(_X_state, this->system_size, (j) % this->system_size);
            this->set_hamiltonian_elements(k, 2 * this->_c * this->_J * (1.0 + this->_J * this->_J) * Xval * YXval, YX_state);
            
            //<! second term Y_j Z_j+1 X_j+2
            auto [_Xval, __X_state]  = X(base_state, this->system_size, (j + 2) % this->system_size);
            auto [ZXval, tmp5]      = Z(__X_state, this->system_size, (j + 1) % this->system_size);
            auto [YZXval, YZX_state] = Y(__X_state, this->system_size, (j) % this->system_size);
            this->set_hamiltonian_elements(k, 2 * this->_c * this->_J * this->_J * _Xval * ZXval * YZXval, YZX_state);
	    }
	}
}


/// @brief Method to create hamiltonian within the class
/// @param site site index where the local hamiltonian acts
/// @return the local hamiltonian at site site
typename TIFP::sparse_matrix TIFP::create_local_hamiltonian(int site)
{
    sparse_matrix H_local(dim, dim);
    auto set_loc_hamiltonian_elements = [this, &H_local](u64 k, elem_ty value, u64 new_idx){
        u64 state;
        elem_ty sym_eig;
        std::tie(state, sym_eig) = this->_hilbert_space.find_matrix_element(new_idx, this->_hilbert_space.get_norm(k));
        H_local(state, k) += value * sym_eig;
    };
    if(this->_boundary_condition && site >= this->system_size - 2){
        return sparse_matrix(dim, dim);
    } else {
        for (size_t k = 0; k < this->dim; k++) {
            int base_state = this->_hilbert_space(k);
                
            auto [Sz_j2, tmp] = Z(base_state, this->system_size, (site + 2) % this->system_size);

            //<! first term X_j X_j+1 Z_j+2
            auto [tmp1, X_state] = X(base_state, this->system_size, (site + 1) % this->system_size);
            auto [tmp2, XX_state] = X(X_state, this->system_size, site % this->system_size);
            set_loc_hamiltonian_elements(k, Sz_j2, XX_state);

            //<! second term Z_j Y_j+1 Y_j+2
            auto [Yval, Y_state] = Y(base_state, this->system_size, (site + 2) % this->system_size);
            auto [YYval, YY_state] = Y(Y_state, this->system_size, (site + 1) % this->system_size);
            auto [Sz_jYY, tmp3] = Z(YY_state, this->system_size, site % this->system_size);
            set_loc_hamiltonian_elements(k, this->_J * this->_J * Sz_jYY * Yval * YYval, YY_state);
            
            //<! third term Z_j Z_j+2
            auto [Sz_j, tmp4] = Z(base_state, this->system_size, site % this->system_size);
            H_local(k, k) += this->_J * Sz_j * Sz_j2;
        }

        return H_local;
    }
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
    printSeparated(os, "\t", 16, true, "c", this->_c);
    
    printSeparated(os, "\t", 16, true, "zz", this->syms.zz_sym);
    if(this->system_size % 2 == 0) printSeparated(os, "\t", 16, true, "z1", this->syms.z1_sym);
    if(this->system_size % 4 == 0) printSeparated(os, "\t", 16, true, "z2", this->syms.z2_sym);

    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");

    return os;
}



//<! ------------------------------------------------------------------------------ ADDITIONAL METHODS FOR SYMMETRIC HAMILTONIAN



