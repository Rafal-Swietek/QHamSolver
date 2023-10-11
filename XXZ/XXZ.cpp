#include "includes/config.hpp"
#include "../include/QHamSolver.h"
#include "includes/XXZ.hpp"



//<! ---------------------------------------------------------------------------------------------------------------------------------------
//<! ------------------------------------------------------------------------------------------------------------------------ IMPLEMENTATION

//<! ------------------------------------------------------------------------------ CONSTRUCTORS

/// @brief Constructor for XXZ model class
/// @param _BC boundary condition
/// @param L system size
/// @param J1 nearest nieghbour coupling
/// @param J2 next-nearest nieghbour coupling
/// @param delta1 nearest nieghbour interaction
/// @param delta2 next-nearest nieghbour interaction
/// @param hz longitudinal magnetic field
/// @param add_parity_breaking add edge term to break parity (if no disorder present)
/// @param add_edge_fields add edge fields to keep supersymmetry for OBC
XXZ::XXZ(int _BC, unsigned int L, double J1, double J2, double delta1, double delta2, double hz, float Sz,  
            bool add_parity_breaking, double w, const u64 seed)
{ 
    CONSTRUCTOR_CALL;

    this->_boundary_condition = _BC;
    this->system_size = L; 
    this->_J1 = J1;
    this->_J2 = J2;
    this->_delta1 = delta1;
    this->_delta2 = delta2;
    
    this->_hz = hz;
    this->Sz = Sz;
    this->_add_parity_breaking = add_parity_breaking;

    // if(this->_boundary_condition)   // only for OBC
    //     this->_add_edge_fields = add_edge_fields;

    // if(this->_add_edge_fields)
    //     this->_add_parity_breaking = false;
    //<! disorder terms
    this->_w = w;
    if(std::abs(w) > 0){
        this->_use_disorder = true;
        this->_seed = seed;
        this->_add_parity_breaking = false;
    }
    init(); 
}

/// @brief Constructor from input stream
/// @param os input stream
XXZ::XXZ(std::istream& os)
    { os >> *this; }

//<! ------------------------------------------------------------------------------ HAMILTONIAN BUILDERS
/// @brief Set hamiltonian matrix element given with value and new index
/// @param k current basis state
/// @param value value of matrix element
/// @param new_idx new index to be found in hilbert space
void XXZ::set_hamiltonian_elements(u64 k, double value, u64 new_idx)
{
    u64 idx = this->_hilbert_space.find(new_idx);
    try {
        H(idx, k) += value;
        H(k, idx) += value;
    } 
    catch (const std::exception& err) {
        std::cout << "Exception:\t" << err.what() << "\n";
        std::cout << "SHit ehhh..." << std::endl;
        printSeparated(std::cout, "\t", 14, true, new_idx, idx, this->_hilbert_space(k), value);
    }
}


/// @brief Method to create hamiltonian within the class
void XXZ::create_hamiltonian()
{
    this->H = sparse_matrix(this->dim, this->dim);
    
    if(this->_use_disorder)
        this->_disorder = disorder_generator.uniform(system_size, this->_hz - this->_w, this->_hz + this->_w);
    else{
        if(this->_add_parity_breaking)
            this->_disorder(0) = this->_hz;
    }   
	std::cout << "disorder: \t\t" << this->_disorder.t() << std::endl;

    std::vector<double> coupling = {this->_J1, this->_J2};
    std::vector<double> interaction = {this->_delta1, this->_delta2};
    
    std::vector<int> neighbor_distance = {1, 2};
    auto check_spin = op::__builtins::get_digit(this->system_size);

    for (u64 k = 0; k < this->dim; k++) 
    {
		double s_i, s_j;
		u64 base_state = this->_hilbert_space(k);
		for (int j = 0; j < this->system_size; j++) 
        {
			s_i = check_spin(base_state, j) ? 0.5 : -0.5;				// true - spin up, false - spin down
			
            //<! longitudinal field with disorder
			this->H(k, k) += (this->_disorder(j)) * s_i;                            // diagonal elements setting

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
typename XXZ::sparse_matrix XXZ::create_local_hamiltonian(int site)
{
    sparse_matrix H_local(dim, dim);
    
    return H_local;
}


//<! ------------------------------------------------------------------------------ OVVERRIDEN OPERATORS AND OPERATOR KERNELS
/// @brief Read model parameters from input stream
/// @tparam U1_sector U(1) symmetry sector as teamplate input 
/// @param os input stream to read parameters
std::istream& XXZ::read(std::istream& os)
{
    
    return os;
}

/// @brief Write hamiltonian to stream as human readable
/// @param os input stream to read parameters
std::ostream& XXZ::write(std::ostream& os) const
{
    printSeparated(os, "\t", 16, true, "Model:", "XXZ spin chain");
    os << std::endl;
    printSeparated(os, "\t", 16, true, "Hamiltonian:", "H = \u03A3_r \u03A3_i J_r[ S^x_i S^x_i+1 + S^y_i S^y_i+1 ] + \u0394_r S^z_iS^z_i+1 + \u03A3_i h_i S^z_i");
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
    printSeparated(os, "\t", 16, true, "Parameters:");
    printSeparated(os, "\t", 16, true, "L", this->system_size);
    printSeparated(os, "\t", 16, true, "J_1", this->_J1);
    printSeparated(os, "\t", 16, true, "\u0394_1", this->_delta1);
    printSeparated(os, "\t", 16, true, "w", this->_w);
    printSeparated(os, "\t", 16, true, "hz", this->_hz);
    //printSeparated(os, "\t", 16, true, "disorder", this->_disorder.t());

    printSeparated(os, "\t", 16, true, "J_2", this->_J2);
    printSeparated(os, "\t", 16, true, "\u0394_2", this->_delta2);
    printSeparated(os, "\t", 16, true, "seed", this->_seed);
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");

    return os;
}
