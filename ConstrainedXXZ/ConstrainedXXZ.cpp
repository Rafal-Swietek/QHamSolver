#include "includes/config.hpp"
#include "../include/QHamSolver.h"
#include "includes/ConstrainedXXZ.hpp"


//<! ---------------------------------------------------------------------------------------------------------------------------------------
//<! ------------------------------------------------------------------------------------------------------------------------ IMPLEMENTATION

//<! ------------------------------------------------------------------------------ CONSTRUCTORS

/// @brief Initialize model dependencies (symetries, hilbert space, ...)
void ConstrainedXXZ::init()
{   
    _debug_start( clk::time_point start = std::chrono::system_clock::now(); )
    // set symmetry generators
    this->set_symmetry_generators();

    // initialize hilbert space with PXP + U(1)
    auto Translate = QOps::__builtins::translation(this->system_size, 1);
    auto flip = QOps::__builtins::spin_flip_x(this->system_size);
    const int Lx = this->system_size;
    const float Sz = this->syms._Sz;
    auto some_kernel = [&Translate, &flip, Sz, Lx](u64 n){
        // n = std::get<0>(flip(n));
        int num_particles = Sz / _Spin + Lx / 2;
        // printSeparated(std::cout, "\t", 20, true, n, boost::dynamic_bitset<>(Lx, n), num_particles);
        return ( !( (n) & std::get<0>( Translate(n) ) ) ) && (__builtin_popcountll(n) == num_particles);
        // return !( (n) & std::get<0>( Translate(n) ) );
    };

    auto _hilbert_PXP    = QHS::constrained_hilbert_space(this->system_size, std::move(some_kernel));
    auto _second_hilbert = this->_use_symmetries? QHS::point_symmetric( this->system_size, this->symmetry_generators, this->_boundary_condition, this->syms.k_sym, 0) :\
                                                    QHS::point_symmetric( this->system_size, v_1d<QOps::genOp>(), 1, 0, -1);
    
    this->_hilbert_space = tensor(_second_hilbert, _hilbert_PXP);
    // this->_hilbert_space = _second_hilbert;
    this->dim = this->_hilbert_space.get_hilbert_space_size();
    _debug_end( std::cout << "\t\tFinished setting generating reduced basis (U(1) x point symmetries) with size:\t dim=" << this->dim << "\tin " << tim_s(start) << " seconds" << std::endl; )

    // create hamiltonian
    _debug_start( start = std::chrono::system_clock::now(); )
    this->create_hamiltonian();
    _debug_end( std::cout << "\t\tFinished generating Hamiltonian in " << tim_s(start) << " seconds" << std::endl; )
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
ConstrainedXXZ::ConstrainedXXZ(int _BC, unsigned int L, double J, double delta, int ksym, int psym, float Sz, bool use_syms)
{ 
    CONSTRUCTOR_CALL;

    this->_boundary_condition = _BC;
    this->system_size = L; 
    this->_J = J;
    this->_delta = delta;
    
    this->_use_symmetries = use_syms;
    
    //<! symmetries
    this->syms.k_sym = ksym;
    this->syms.p_sym = psym;
    this->syms._Sz = Sz;

    #ifdef USE_REAL_SECTORS
        if(this->_boundary_condition == 0){ // only for PBC
            bool is_k_sector_real = (std::abs(two_pi * ksym / this->system_size) < 1e-4) || (std::abs(two_pi * ksym / this->system_size - pi) < 1e-4);
            _assert_(is_k_sector_real, NOT_ALLOWED_SYM_SECTOR "\n\t\tMatrix type is real due to USE_REAL_SECTORS macro, but quasimomentum sector is complex, i.e. k != 0, pi");
        }
    #endif

    this->init(); 
}

/// @brief Constructor from input stream
/// @param os input stream
ConstrainedXXZ::ConstrainedXXZ(std::istream& os)
    { os >> *this; }

/// @brief Set symmetry generators (among spin flips if fields perpendicular to spin axis are 0)
void ConstrainedXXZ::set_symmetry_generators()
{   
    // parity symmetry
    this->symmetry_generators.emplace_back(QOps::_parity_symmetry(this->system_size, this->syms.p_sym));
}

//<! ------------------------------------------------------------------------------ HAMILTONIAN BUILDERS
/// @brief Set hamiltonian matrix element given with value and new index
/// @param k current basis state
/// @param value value of matrix element
/// @param new_idx new index to be found in hilbert space
void ConstrainedXXZ::set_hamiltonian_elements(u64 k, elem_ty value, u64 new_idx)
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
void ConstrainedXXZ::create_hamiltonian()
{
    this->H = sparse_matrix(this->dim, this->dim);
	for (int j = 0; j < this->system_size - 2 * int(this->_boundary_condition); j++) 
        this->H += this->create_local_hamiltonian(j);
}


/// @brief Method to create hamiltonian within the class
/// @param site site index where the local hamiltonian acts
/// @return the local hamiltonian at site site
typename ConstrainedXXZ::sparse_matrix ConstrainedXXZ::create_local_hamiltonian(int site)
{
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
    if(this->_boundary_condition && site >= this->system_size - 1){
        return sparse_matrix(dim, dim);
    } else {
        for (size_t k = 0; k < this->dim; k++) {
            int base_state = this->_hilbert_space(k);
            // auto vec1 = boost::dynamic_bitset<>(this->system_size, base_state);                
            const double Sz_j  = std::real( std::get<0>( Z(base_state, this->system_size, (site) % this->system_size) ) );
            const double Sz_j1 = std::real( std::get<0>( Z(base_state, this->system_size, (site + 1) % this->system_size) ) );
            
            //<! Interaction Z_j Z_j+1
            H_local(k, k) += this->_delta * Sz_j + Sz_j1;
            
            //<! Hopping X_j X_j+1 + Y_j Y_j+1
            if(Sz_j * Sz_j1 < 0){
                auto [tmp1, X_state] = X(base_state, this->system_size, (site) % this->system_size);
                auto [tmp2, XX_state] = X(X_state, this->system_size, (site + 1) % this->system_size);
                set_loc_hamiltonian_elements(k, this->_J * tmp1 * tmp2, XX_state);
            }
        }
        return H_local;
    }
}


//<! ------------------------------------------------------------------------------ OVVERRIDEN OPERATORS AND OPERATOR KERNELS
/// @brief Read model parameters from input stream
/// @tparam U1_sector U(1) symmetry sector as teamplate input 
/// @param os input stream to read parameters
std::istream& ConstrainedXXZ::read(std::istream& os)
{
    
    return os;
}

/// @brief Write hamiltonian to stream as human readable
/// @param os input stream to read parameters
std::ostream& ConstrainedXXZ::write(std::ostream& os) const
{
    printSeparated(os, "\t", 16, true, "Model:", "GoldenChain spin chain");
    os << std::endl;
    printSeparated(os, "\t", 16, true, "Hamiltonian:", "H = \u03A3_j P_j (X_j X_j+1 + Y_j Y_j+1 + \u0394 Z_j Z_j+1) P_j+1 ]");
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
    printSeparated(os, "\t", 16, true, "Parameters:");
    printSeparated(os, "\t", 16, true, "L", this->system_size);
    printSeparated(os, "\t", 16, true, "J", this->_J);
    printSeparated(os, "\t", 16, true, "\u0394", this->_delta);
    
    printSeparated(os, "\t", 16, true, "k", this->syms.k_sym);
    printSeparated(os, "\t", 16, true, "p", this->syms.p_sym);
    printSeparated(os, "\t", 16, true, "Sz", this->syms._Sz);

    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");

    return os;
}



//<! ------------------------------------------------------------------------------ ADDITIONAL METHODS FOR SYMMETRIC HAMILTONIAN



