#include "includes/config.hpp"
#include "../include/QHamSolver.h"
#include "includes/XXZ_sym.hpp"


//<! ---------------------------------------------------------------------------------------------------------------------------------------
//<! ------------------------------------------------------------------------------------------------------------------------ IMPLEMENTATION

//<! ------------------------------------------------------------------------------ CONSTRUCTORS

/// @brief Initialize model dependencies (symetries, hilbert space, ...)
void XXZsym::init()
{   
    // set symmetry generator
    CONSTRUCTOR_CALL;
    _debug_start( clk::time_point start = std::chrono::system_clock::now(); )
    this->set_symmetry_generators();
    _debug_end( std::cout << "\t\tFinished setting generators in " << tim_s(start) << " seconds" << std::endl; )

    // initialize hilbert space
    _debug_start( start = std::chrono::system_clock::now(); )
    this->_hilbert_space = tensor(
                            QHS::point_symmetric( this->system_size, 
                                            this->symmetry_generators, 
                                            this->_boundary_condition,
                                            this->syms.k_sym,
                                            0
                                            ),
                            U1Hilbert(this->system_size, this->syms.Sz)
                                );
    this->dim = this->_hilbert_space.get_hilbert_space_size();
    _debug_end( std::cout << "\t\tFinished setting generating reduced basis (U(1) x point symmetries) with size:\t dim=" << this->dim << "\tin " << tim_s(start) << " seconds" << std::endl; )

    // create hamiltonian
    _debug_start( start = std::chrono::system_clock::now(); )
    this->create_hamiltonian();
    _debug_end( std::cout << "\t\tFinished generating Hamiltonian in " << tim_s(start) << " seconds" << std::endl; )
    // std::cout << "Mapping:\n" << this->_hilbert_space.get_mapping() << std::endl;
    // std::cout << "Hamiltonian:\n" << arma::Mat<elem_ty>(this->H) << std::endl;
}

/// @brief Constructor for XXZsym model class
/// @param _BC boundary condition
/// @param L system size
/// @param J1 nearest nieghbour coupling
/// @param J2 next-nearest nieghbour coupling
/// @param delta1 nearest nieghbour interaction
/// @param delta2 next-nearest nieghbour interaction
/// @param eta1 nearest nieghbour anisotropy
/// @param eta2 next-nearest nieghbour anisotropy
/// @param hx transverse magnetic field
/// @param hz longitudinal magnetic field
/// @param ksym quasimomentum symmetry sector
/// @param psym parity symmetry sector
/// @param zxsym spin flip in X symemtry sector
/// @param Sz magnetization sector
XXZsym::XXZsym(int _BC, unsigned int L, double J1, double J2, double delta1, double delta2, double hz, 
                int ksym, int psym, int zxsym, float Sz)
{ 
    CONSTRUCTOR_CALL;

    this->_boundary_condition = _BC;
    this->system_size = L; 
    this->_J1 = J1;
    this->_J2 = J2;
    this->_delta1 = delta1;
    this->_delta2 = delta2;
    
    this->_hz = hz;
    
    // if(this->_boundary_condition)   // only for OBC
    //     this->_add_edge_fields = add_edge_fields;

    //<! symmetries
    this->syms.k_sym = ksym;
    this->syms.p_sym = psym;
    this->syms.zx_sym = zxsym;
    this->syms.Sz = Sz;

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
XXZsym::XXZsym(std::istream& os)
    { os >> *this; }

/// @brief Set symmetry generators (among spin flips if fields perpendicular to spin axis are 0)
void XXZsym::set_symmetry_generators()
{   
    // parity symmetry
    this->symmetry_generators.emplace_back(op::_parity_symmetry(this->system_size, this->syms.p_sym));
    
    if(this->_hz == 0 && this->syms.Sz == 0.0)
        this->symmetry_generators.emplace_back(op::_spin_flip_x_symmetry(this->system_size, this->syms.zx_sym));
}

//<! ------------------------------------------------------------------------------ HAMILTONIAN BUILDERS
/// @brief Set hamiltonian matrix element given with value and new index
/// @param k current basis state
/// @param value value of matrix element
/// @param new_idx new index to be found in hilbert space
void XXZsym::set_hamiltonian_elements(u64 k, elem_ty value, u64 new_idx)
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
void XXZsym::create_hamiltonian()
{
    this->H = sparse_matrix(this->dim, this->dim);
    
    std::vector<double> coupling = {this->_J1, this->_J2};
    std::vector<double> interaction = {this->_delta1, this->_delta2};
    
    std::vector<int> neighbor_distance = {1, 2};
    auto check_spin = op::__builtins::get_digit(this->system_size);

    for (u64 k = 0; k < this->dim; k++) {
		double s_i, s_j;
		u64 base_state = this->_hilbert_space(k);
		for (int j = 0; j < this->system_size; j++) {
			s_i = check_spin(base_state, j) ? 0.5 : -0.5;				// true - spin up, false - spin down
			
            //<! longitudinal field with disorder
			this->H(k, k) += this->_hz * s_i;                            // diagonal elements setting

			for(int a = 0; a < neighbor_distance.size(); a++){
                int r = neighbor_distance[a];
                int nei = j + r;
                if(nei >= this->system_size)
                    nei = (this->_boundary_condition)? -1 : nei % this->system_size;

                
                if (nei >= 0) //<! boundary conditions
                {
                    s_j = check_spin(base_state, nei) ? 0.5 : -0.5;
                    if(s_i < 0 && s_j > 0){
                        // u64 new_idx =  flip(base_state, BinaryPowers[this->system_size - 1 - nei], this->system_size - 1 - nei);
                        // new_idx =  flip(new_idx, BinaryPowers[this->system_size - 1 - j], this->system_size - 1 - j);
                        auto [val, state_tmp]   = operators::sigma_minus(base_state, this->system_size, nei);
                        auto [val2, state]      = operators::sigma_plus(state_tmp, this->system_size, j);
                        
                        // 0.5 cause flip 0.5*(S+S- + S-S+)
                        this->set_hamiltonian_elements(k, 0.5 * coupling[a], state);
                    }
                    else if(s_i > 0 && s_j < 0){
                        // u64 new_idx =  flip(base_state, BinaryPowers[this->system_size - 1 - nei], this->system_size - 1 - nei);
                        // new_idx =  flip(new_idx, BinaryPowers[this->system_size - 1 - j], this->system_size - 1 - j);
                        auto [val, state_tmp]   = operators::sigma_minus(base_state, this->system_size, j);
                        auto [val2, state]      = operators::sigma_plus(state_tmp, this->system_size, nei);
                        
                        // 0.5 cause flip 0.5*(S+S- + S-S+)
                        this->set_hamiltonian_elements(k, 0.5 * coupling[a], state);
                    }
                    
                    //<! Interaction (spin correlations) with neighbour at distance r
                    this->H(k, k) += interaction[a] * s_i * s_j;
                }
            }
		}
		//std::cout << std::bitset<4>(base_state) << "\t";
	}
}


/// @brief Method to create hamiltonian within the class
/// @param site site index where the local hamiltonian acts
/// @return the local hamiltonian at site site
typename XXZsym::sparse_matrix XXZsym::create_local_hamiltonian(int site)
{
    sparse_matrix H_local(dim, dim);
    
    return H_local;
}


//<! ------------------------------------------------------------------------------ OVVERRIDEN OPERATORS AND OPERATOR KERNELS
/// @brief Read model parameters from input stream
/// @tparam U1_sector U(1) symmetry sector as teamplate input 
/// @param os input stream to read parameters
std::istream& XXZsym::read(std::istream& os)
{
    
    return os;
}

/// @brief Write hamiltonian to stream as human readable
/// @param os input stream to read parameters
std::ostream& XXZsym::write(std::ostream& os) const
{
    printSeparated(os, "\t", 16, true, "Model:", "XXZsym spin chain");
    os << std::endl;
    printSeparated(os, "\t", 16, true, "Hamiltonian:", "H = \u03A3_r \u03A3_i J_r[ S^x_i S^x_i+1 + S^y_i S^y_i+1 ] + \u0394_r S^z_iS^z_i+1 + h^z \u03A3_i S^z_i");
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
    printSeparated(os, "\t", 16, true, "Parameters:");
    printSeparated(os, "\t", 16, true, "L", this->system_size);
    printSeparated(os, "\t", 16, true, "J_1", this->_J1);
    printSeparated(os, "\t", 16, true, "\u0394_1", this->_delta1);
    printSeparated(os, "\t", 16, true, "hz", this->_hz);

    printSeparated(os, "\t", 16, true, "J_2", this->_J2);
    printSeparated(os, "\t", 16, true, "\u0394_2", this->_delta2);

    printSeparated(os, "\t", 16, true, "k", this->syms.k_sym);
    printSeparated(os, "\t", 16, true, "p", this->syms.p_sym);
    if(this->_hz == 0 && this->syms.Sz == 0.0) 
        printSeparated(os, "\t", 16, true, "zx", this->syms.zx_sym);
    printSeparated(os, "\t", 16, true, "Sz", this->syms.Sz);

    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");

    return os;
}



//<! ------------------------------------------------------------------------------ ADDITIONAL METHODS FOR SYMMETRIC HAMILTONIAN



