#include "includes/config.hpp"
#include "../include/QHamSolver.h"
#include "includes/XYZ_sym.hpp"


//<! ---------------------------------------------------------------------------------------------------------------------------------------
//<! ------------------------------------------------------------------------------------------------------------------------ IMPLEMENTATION

//<! ------------------------------------------------------------------------------ CONSTRUCTORS

/// @brief Initialize model dependencies (symetries, hilbert space, ...)
void XYZsym::init()
{   
    // set symmetry generators
    this->set_symmetry_generators();

    // initialize hilbert space
    this->_hilbert_space = point_symmetric( this->system_size, 
                                            this->symmetry_generators, 
                                            this->_boundary_condition,
                                            this->syms.k_sym,
                                            0
                                            );
    this->dim = this->_hilbert_space.get_hilbert_space_size();

    // create hamiltonian
    this->create_hamiltonian();
}

/// @brief Constructor for XYZsym model class
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
/// @param zzsym spin flip in Z symmetry sector
XYZsym::XYZsym(int _BC, unsigned int L, double J1, double J2, double delta1, double delta2, double eta1, double eta2,
            double hx, double hz, int ksym, int psym, int zzsym, int zxsym, bool add_edge_fields)
{ 
    CONSTRUCTOR_CALL;

    this->_boundary_condition = _BC;
    this->system_size = L; 
    this->_J1 = J1;
    this->_J2 = J2;
    this->_delta1 = delta1;
    this->_delta2 = delta2;
    this->_eta1 = eta1;
    this->_eta2 = eta2;
    
    this->_hz = hz;
    this->_hx = hx;
    
    if(this->_boundary_condition)   // only for OBC
        this->_add_edge_fields = add_edge_fields;


    //<! symmetries
    this->syms.k_sym = ksym;
    this->syms.p_sym = psym;
    this->syms.zx_sym = zxsym;
    this->syms.zz_sym = zzsym;
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
XYZsym::XYZsym(std::istream& os)
    { os >> *this; }

/// @brief Set symmetry generators (among spin flips if fields perpendicular to spin axis are 0)
void XYZsym::set_symmetry_generators()
{   
    // parity symmetry
    this->symmetry_generators.emplace_back(op::_parity_symmetry(this->system_size, this->syms.p_sym));

    // spin flips (only for even L both can be used)
    if(this->_hx == 0)
        this->symmetry_generators.emplace_back(op::_spin_flip_z_symmetry(this->system_size, this->syms.zz_sym));
    
    if(this->_hz == 0 && !this->_add_edge_fields)
        if(this->system_size % 2 == 0 || this->_hx != 0) // for odd system sizes enter only if previous symmetry not taken
            this->symmetry_generators.emplace_back(op::_spin_flip_x_symmetry(this->system_size, this->syms.zx_sym));
}

//<! ------------------------------------------------------------------------------ HAMILTONIAN BUILDERS
/// @brief Set hamiltonian matrix element given with value and new index
/// @param k current basis state
/// @param value value of matrix element
/// @param new_idx new index to be found in hilbert space
void XYZsym::set_hamiltonian_elements(u64 k, elem_ty value, u64 new_idx)
{   
    u64 state, idx;
    elem_ty sym_eig;

    try {
        // //<! Look for index in reduced basis (maybe its the SEC already)
        // idx = this->_hilbert_space.find(new_idx);
        // if (idx < dim)	std::tie(state, sym_eig) = std::make_pair(idx, this->_hilbert_space.get_norm(idx) / this->_hilbert_space.get_norm(k));
        
        // //<! find SEC for input state
        // auto [min, sym_eig] = this->_hilbert_space.find_SEC_representative(new_idx);
        // idx = this->_hilbert_space.find(min);
        // #ifndef USE_REAL_SECTORS
        //     sym_eig = std::conj(sym_eig);
        // #endif
        // if (idx < dim)	std::tie(state, sym_eig) = std::make_pair(idx, this->_hilbert_space.get_norm(idx) / this->_hilbert_space.get_norm(k) * sym_eig);
        // else			std::tie(state, sym_eig) = std::make_pair(0, 0.0);
        
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
        printSeparated(std::cout, "\t", 14, true, new_idx, idx, this->_hilbert_space(k), value, sym_eig);
    }
}

/// @brief Method to create hamiltonian within the class
void XYZsym::create_hamiltonian()
{
    this->H = sparse_matrix(this->dim, this->dim);
    
    double Jz = (this->_eta1 * this->_eta1 - 1) / 2.;
    std::vector<std::vector<double>> parameters = { { this->_J1 * (1 - this->_eta1), this->_J1 * (1 + this->_eta1), this->_J1 * this->_delta1},
                                                    { this->_J2 * (1 - this->_eta2), this->_J2 * (1 + this->_eta2), this->_J2 * this->_delta2}
                                                };
    for(auto& x : parameters)
        std::cout << x << std::endl;
    std::vector<op_type> XYZsymoperators = {operators::sigma_x, operators::sigma_y, operators::sigma_z };
    std::vector<int> neighbor_distance = {1, 2};

    for (size_t k = 0; k < this->dim; k++) {
		int base_state = this->_hilbert_space(k);
	    for (int j = 0; j < this->system_size; j++) {
            cpx val = 0.0;
            u64 op_k;

            double fieldZ = this->_hz;
            if(this->_add_edge_fields && (j == 0 || j == this->system_size - 1))
                fieldZ -= Jz / 2.0;
            std::tie(val, op_k) = operators::sigma_z(base_state, this->system_size, { j });
            this->set_hamiltonian_elements(k, fieldZ * real(val), op_k);
	    	
            std::tie(val, op_k) = operators::sigma_x(base_state, this->system_size, { j });			
            this->set_hamiltonian_elements(k, this->_hx * real(val), op_k);

            for(int a = 0; a < neighbor_distance.size(); a++){
                int r = neighbor_distance[a];
				int nei = j + r;
				if(nei >= this->system_size)
					nei = (this->_boundary_condition)? -1 : nei % this->system_size;

	    	    if (nei >= 0) {
                    for(int b = 0; b < XYZsymoperators.size(); b++){
                        op_type op = XYZsymoperators[b];
		                auto [val1, op_k] = op(base_state, this->system_size, { j });
		                auto [val2, opop_k] = op(op_k, this->system_size, { nei });
						this->set_hamiltonian_elements(k, parameters[a][b] * real(val1 * val2), opop_k);
                    }
	    	    }
            }
	    }
	}

    // add SUSY ground state energy (const shift) and invert (minus sign in front of hamiltonian)
    this->H = -this->H + this->_J1 * (this->system_size - int(this->_boundary_condition)) * (2 + Jz) / 4. * arma::eye(this->dim, this->dim);
    if(this->_boundary_condition)
        this->H = this->H + this->_J1 * (1 + 3 * this->_eta1 * this->_eta1) / 4.0 * arma::eye(this->dim, this->dim);
}


/// @brief Method to create hamiltonian within the class
/// @param site site index where the local hamiltonian acts
/// @return the local hamiltonian at site site
typename XYZsym::sparse_matrix XYZsym::create_local_hamiltonian(int site)
{
    sparse_matrix H_local(dim, dim);
    
    return H_local;
}


//<! ------------------------------------------------------------------------------ OVVERRIDEN OPERATORS AND OPERATOR KERNELS
/// @brief Read model parameters from input stream
/// @tparam U1_sector U(1) symmetry sector as teamplate input 
/// @param os input stream to read parameters
std::istream& XYZsym::read(std::istream& os)
{
    
    return os;
}

/// @brief Write hamiltonian to stream as human readable
/// @param os input stream to read parameters
std::ostream& XYZsym::write(std::ostream& os) const
{
    printSeparated(os, "\t", 16, true, "Model:", "XYZsym spin chain");
    os << std::endl;
    printSeparated(os, "\t", 16, true, "Hamiltonian:", "H = \u03A3_r J_r\u03A3_i [ (1-\u03B7_r) S^x_i S^x_i+1 + (1+\u03B7_r) S^y_i S^y_i+1) + \u0394_r S^z_iS^z_i+1] + \u03A3_i h_i S^z_i");
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
    printSeparated(os, "\t", 16, true, "Parameters:");
    printSeparated(os, "\t", 16, true, "L", this->system_size);
    printSeparated(os, "\t", 16, true, "J_1", this->_J1);
    printSeparated(os, "\t", 16, true, "\u0394_1", this->_delta1);
    printSeparated(os, "\t", 16, true, "\u03B7_1", this->_eta1);
    printSeparated(os, "\t", 16, true, "hz", this->_hz);
    printSeparated(os, "\t", 16, true, "hx", this->_hx);

    printSeparated(os, "\t", 16, true, "J_2", this->_J2);
    printSeparated(os, "\t", 16, true, "\u0394_2", this->_delta2);
    printSeparated(os, "\t", 16, true, "\u03B7_2", this->_eta2);

    printSeparated(os, "\t", 16, true, "k", this->syms.k_sym);
    printSeparated(os, "\t", 16, true, "p", this->syms.p_sym);
    if(this->_hz == 0) printSeparated(os, "\t", 16, true, "zx", this->syms.zx_sym);
    if(this->_hx == 0) printSeparated(os, "\t", 16, true, "zz", this->syms.zz_sym);

    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");

    return os;
}



//<! ------------------------------------------------------------------------------ ADDITIONAL METHODS FOR SYMMETRIC HAMILTONIAN



