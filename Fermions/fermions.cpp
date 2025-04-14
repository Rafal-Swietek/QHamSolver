#include "includes/config.hpp"
#include "../include/QHamSolver.h"
#include "includes/fermions.hpp"


//<! ---------------------------------------------------------------------------------------------------------------------------------------
//<! ------------------------------------------------------------------------------------------------------------------------ IMPLEMENTATION

//<! ------------------------------------------------------------------------------ CONSTRUCTORS

/// @brief Initialize model dependencies (symetries, hilbert space, ...)
void Fermions::init()
{   
    // set symmetry generator
    CONSTRUCTOR_CALL;
    _debug_start( clk::time_point start = std::chrono::system_clock::now(); )
    this->set_symmetry_generators();
    _debug_end( std::cout << "\t\tFinished setting generators in " << tim_s(start) << " seconds" << std::endl; )

    // initialize hilbert space
    _debug_start( start = std::chrono::system_clock::now(); )
    // this->_hilbert_space = tensor(
    //                         QHS::point_symmetric( this->system_size, 
    //                                         this->symmetry_generators, 
    //                                         this->_boundary_condition,
    //                                         this->syms.k_sym,
    //                                         0
    //                                         ),
    //                         U1Hilbert(this->system_size, this->syms.N)
    //                             );
    if(this->_use_symmetries){
        this->_hilbert_space = tensor(
                                QHS::point_symmetric<QOps::particle::fermion>( this->system_size, this->symmetry_generators, 
                                                this->_boundary_condition,this->syms.k_sym, 0),
                                U1Hilbert(this->system_size, this->syms.N)
                                    );
    } else {
        this->_hilbert_space = tensor(
                                QHS::point_symmetric<QOps::particle::fermion>( this->system_size, v_1d<QOps::genOp>(), 1, 0, 0 ),
                                U1Hilbert(this->system_size, this->syms.N)
                                    );
        // this->_hilbert_space = QHS::point_symmetric( this->system_size, v_1d<QOps::genOp>(), 1, 0, 0 );
    }
    this->dim = this->_hilbert_space.get_hilbert_space_size();
    _debug_end( std::cout << "\t\tFinished setting generating reduced basis (U(1) x point symmetries) with size:\t dim=" << this->dim << "\tin " << tim_s(start) << " seconds" << std::endl; )

    // create hamiltonian
    _debug_start( start = std::chrono::system_clock::now(); )
    this->create_hamiltonian();
    _debug_end( std::cout << "\t\tFinished generating Hamiltonian in " << tim_s(start) << " seconds" << std::endl; )
    // std::cout << "Mapping:\n" << this->_hilbert_space.get_mapping() << std::endl;
    // std::cout << "Hamiltonian:\n" << arma::Mat<elem_ty>(this->H) << std::endl;
}

/// @brief Constructor for Fermions model class
/// @param _BC boundary condition
/// @param L system size
/// @param N number of QOps::particles
/// @param t1 nearest nieghbour coupling
/// @param t2 next-nearest nieghbour coupling
/// @param V1 nearest nieghbour interaction
/// @param V2 next-nearest nieghbour interaction
/// @param mu chemical potential
/// @param ksym quasimomentum symmetry sector
/// @param psym parity symmetry sector
/// @param zxsym spin flip in X symemtry sector
/// @param use_syms use point symmetries or take entire U(1)n Hilbert space
Fermions::Fermions(int _BC, unsigned int L, unsigned int N, double t1, double t2, double V1, double V2, double mu, 
                        int ksym, int psym, int zxsym, bool use_syms)
{ 
    CONSTRUCTOR_CALL;

    this->_boundary_condition = _BC;
    this->system_size = L; 
    this->_t1 = t1;
    this->_t2 = t2;
    this->_V1 = V1;
    this->_V2 = V2;
    this->_mu = mu;

    //<! symmetries
    this->syms.k_sym = ksym;
    this->syms.p_sym = psym;
    this->syms.zx_sym = zxsym;
    this->syms.N = N;
    this->_use_symmetries = use_syms;

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
Fermions::Fermions(std::istream& os)
    { os >> *this; }

/// @brief Set symmetry generators (among spin flips if fields perpendicular to spin axis are 0)
void Fermions::set_symmetry_generators()
{   
    // parity symmetry
    this->symmetry_generators.emplace_back(QOps::_parity_symmetry<QOps::particle::fermion>(this->system_size, this->syms.p_sym));
    
    if(this->syms.N == this->system_size / 2 && this->_mu == 0)
        this->symmetry_generators.emplace_back(QOps::_spin_flip_x_symmetry<QOps::particle::fermion>(this->system_size, this->syms.zx_sym));
}

//<! ------------------------------------------------------------------------------ HAMILTONIAN BUILDERS
/// @brief Set hamiltonian matrix element given with value and new index
/// @param k current basis state
/// @param value value of matrix element
/// @param new_idx new index to be found in hilbert space
void Fermions::set_hamiltonian_elements(u64 k, elem_ty value, u64 new_idx)
{   
    u64 state, idx;
    elem_ty sym_eig;

    try {
        std::tie(state, sym_eig) = this->_hilbert_space.find_matrix_element(new_idx, this->_hilbert_space.get_norm(k));
        #ifdef USE_REAL_SECTORS
            H(state, k) += std::real(value * sym_eig);
            // H(k, state) += std::real(value * sym_eig);
        #else
            H(state, k) += value * sym_eig;
            // H(k, state) += value * sym_eig;
        #endif
    } 
    catch (const std::exception& err) {
        std::cout << "Exception:\t" << err.what() << "\n";
        std::cout << "SHit ehhh..." << std::endl;
        printSeparated(std::cout, "\t", 14, true, new_idx, idx, this->_hilbert_space(k), value, sym_eig);
    }
}

/// @brief Method to create hamiltonian within the class
void Fermions::create_hamiltonian()
{
    this->H = sparse_matrix(this->dim, this->dim);
    
    std::vector<double> coupling = {this->_t1, this->_t2};
    std::vector<double> interaction = {this->_V1, this->_V2};
    
    std::vector<int> neighbor_distance = {1, 2};
    auto check_spin = QOps::__builtins::get_digit(this->system_size);

    for (u64 k = 0; k < this->dim; k++) {
		double n_i, n_j;
		u64 base_state = this->_hilbert_space(k);
		for (int j = 0; j < this->system_size; j++) {
			n_i = check_spin(base_state, j) ? 1 : 0;				// true - spin up, false - spin down
            
            // Chemical potential shift
            this->H(k, k) += this->_mu * n_i;
            
			for(int a = 0; a < neighbor_distance.size(); a++){
                int r = neighbor_distance[a];
                int nei = j + r;
                if(nei >= this->system_size)
                    nei = (this->_boundary_condition)? -1 : nei % this->system_size;

                
                if (nei >= 0) //<! boundary conditions
                {
                    n_j = check_spin(base_state, nei) ? 1 : 0;
                    if(n_i == 0 && n_j == 1){
                        auto [val, state_tmp]   = operators::fermions::spinless::anihilate<elem_ty>(base_state, this->system_size, nei);
                        auto [val2, state]      = operators::fermions::spinless::create<elem_ty>(state_tmp, this->system_size, j);
                        
                        // 0.5 cause flip 0.5*(S+S- + S-S+)
                        this->set_hamiltonian_elements(k, coupling[a] * val * val2, state);
                    }
                    else if(n_i == 1 && n_j == 0){
                        auto [val, state_tmp]   = operators::fermions::spinless::anihilate<elem_ty>(base_state, this->system_size, j);
                        auto [val2, state]      = operators::fermions::spinless::create<elem_ty>(state_tmp, this->system_size, nei);
                        
                        // 0.5 cause flip 0.5*(S+S- + S-S+)
                        this->set_hamiltonian_elements(k, coupling[a] * val * val2, state);
                    }
                    
                    //<! Interaction (spin correlations) with neighbour at distance r
                    this->H(k, k) += interaction[a] * n_i * n_j;
                }
            }
		}
	}
}


/// @brief Method to create hamiltonian within the class
/// @param site site index where the local hamiltonian acts
/// @return the local hamiltonian at site site
typename Fermions::sparse_matrix Fermions::create_local_hamiltonian(int site)
{
    sparse_matrix H_local(dim, dim);
    
    return H_local;
}


//<! ------------------------------------------------------------------------------ OVVERRIDEN OPERATORS AND OPERATOR KERNELS
/// @brief Read model parameters from input stream
/// @tparam U1_sector U(1) symmetry sector as teamplate input 
/// @param os input stream to read parameters
std::istream& Fermions::read(std::istream& os)
{
    
    return os;
}

/// @brief Write hamiltonian to stream as human readable
/// @param os input stream to read parameters
std::ostream& Fermions::write(std::ostream& os) const
{
    printSeparated(os, "\t", 16, true, "Model:", "Fermions spin chain");
    os << std::endl;
    printSeparated(os, "\t", 16, true, "Hamiltonian:", "H = \u03A3_r \u03A3_i t_r[ c^+_i c_i+1 + h.c ] + \u0394_r n_i n_i+1");
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
    printSeparated(os, "\t", 16, true, "Parameters:");
    printSeparated(os, "\t", 16, true, "L", this->system_size);
    printSeparated(os, "\t", 16, true, "N", this->syms.N);
    printSeparated(os, "\t", 16, true, "t_1", this->_t1);
    printSeparated(os, "\t", 16, true, "V_1", this->_V1);

    printSeparated(os, "\t", 16, true, "t_2", this->_t2);
    printSeparated(os, "\t", 16, true, "V_2", this->_V2);

    printSeparated(os, "\t", 16, true, "k", this->syms.k_sym);
    printSeparated(os, "\t", 16, true, "p", this->syms.p_sym);
    if(this->syms.N == this->system_size / 2) 
        printSeparated(os, "\t", 16, true, "zx", this->syms.zx_sym);

    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");

    return os;
}



//<! ------------------------------------------------------------------------------ ADDITIONAL METHODS FOR SYMMETRIC HAMILTONIAN



