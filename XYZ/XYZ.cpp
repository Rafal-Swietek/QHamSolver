#include "includes/config.hpp"
#include "../include/QHamSolver.h"
#include "includes/XYZ.hpp"



//<! ---------------------------------------------------------------------------------------------------------------------------------------
//<! ------------------------------------------------------------------------------------------------------------------------ IMPLEMENTATION

//<! ------------------------------------------------------------------------------ CONSTRUCTORS

/// @brief Constructor for XYZ model class
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
/// @param add_parity_breaking add edge term to break parity (if no disorder present)
/// @param add_edge_fields add edge fields to keep supersymmetry for OBC
XYZ::XYZ(int _BC, unsigned int L, double J1, double J2, double delta1, double delta2, double eta1, double eta2,
            double hx, double hz, bool add_parity_breaking, bool add_edge_fields, double w, const u64 seed)
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
    this->_add_parity_breaking = add_parity_breaking;

    if(this->_boundary_condition)   // only for OBC
        this->_add_edge_fields = add_edge_fields;

    if(this->_add_edge_fields)
        this->_add_parity_breaking = false;
    //<! disorder terms
    if(w > 0){
        this->_use_disorder = true;
        this->_w = w;
        this->_seed = seed;
        this->_add_parity_breaking = false;
    }
    init(); 
}

/// @brief Constructor from input stream
/// @param os input stream
XYZ::XYZ(std::istream& os)
    { os >> *this; }

//<! ------------------------------------------------------------------------------ HAMILTONIAN BUILDERS
/// @brief Set hamiltonian matrix element given with value and new index
/// @param k current basis state
/// @param value value of matrix element
/// @param new_idx new index to be found in hilbert space
void XYZ::set_hamiltonian_elements(u64 k, double value, u64 new_idx)
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
void XYZ::create_hamiltonian()
{
    this->H = sparse_matrix(this->dim, this->dim);
    
    if(this->_use_disorder)
        this->_disorder = disorder_generator.uniform(system_size, 0, two_pi);
    else{
        if(this->_add_parity_breaking && !this->_use_disorder)
            this->_disorder(0) = 5 * pi / 12.0;

        double Jz = (this->_eta1 * this->_eta1 - 1) / 2.;
        if(this->_add_edge_fields){
            this->_disorder(0)                      = -Jz / 2;
            this->_disorder(this->system_size - 1)  = -Jz / 2;
        }
    }
    std::vector<std::vector<double>> parameters = { { this->_J1 * (1 - this->_eta1), this->_J1 * (1 + this->_eta1), this->_J1 * this->_delta1},
                                                    { this->_J2 * (1 - this->_eta2), this->_J2 * (1 + this->_eta2), this->_J2 * this->_delta2}
                                                };
    for(auto& x : parameters)
        std::cout << x << std::endl;
    std::vector<op_type> XYZoperators = {operators::sigma_x, operators::sigma_y, operators::sigma_z };
    std::vector<int> neighbor_distance = {1, 2};

    for (size_t k = 0; k < this->dim; k++) {
		size_t base_state = this->_hilbert_space(k);
	    for (int j = 0; j < this->system_size; j++) {
            cpx val = 0.0;
            u64 op_k;
            std::tie(val, op_k) = operators::sigma_z(base_state, this->system_size, { j });
			double fieldZ = this->_w * std::cos(this->_disorder(j)) + this->_hz;
            this->set_hamiltonian_elements(k, fieldZ * real(val), op_k);
	    	
            std::tie(val, op_k) = operators::sigma_x(base_state, this->system_size, { j });			
            double fieldX = this->_w * std::sin(this->_disorder(j)) + this->_hx;
            this->set_hamiltonian_elements(k, fieldX * real(val), op_k);

            for(int a = 0; a < neighbor_distance.size(); a++){
                int r = neighbor_distance[a];
				int nei = j + r;
				if(nei >= this->system_size)
					nei = (this->_boundary_condition)? -1 : nei % this->system_size;

	    	    if (nei >= 0) {
                    for(int b = 0; b < XYZoperators.size(); b++){
                        op_type op = XYZoperators[b];
		                auto [val1, op_k] = op(base_state, this->system_size, { j });
		                auto [val2, opop_k] = op(op_k, this->system_size, { nei });
						this->set_hamiltonian_elements(k, parameters[a][b] * real(val1 * val2), opop_k);
                    }
	    	    }
            }
	    }
	}

    // add SUSY ground state energy (const shift) and invert (minus sign in front of hamiltonian)
    // this->H = -this->H + this->_J1 * (this->system_size - int(this->_boundary_condition)) * (2 + Jz) / 4. * arma::eye(this->dim, this->dim);
    // if(this->_boundary_condition)
    //     this->H = this->H + this->_J1 * (1 + 3 * this->_eta1 * this->_eta1) / 4.0 * arma::eye(this->dim, this->dim);
}


/// @brief Method to create hamiltonian within the class
/// @param site site index where the local hamiltonian acts
/// @return the local hamiltonian at site site
typename XYZ::sparse_matrix XYZ::create_local_hamiltonian(int site)
{
    sparse_matrix H_local(dim, dim);
    
    return H_local;
}


//<! ------------------------------------------------------------------------------ OVVERRIDEN OPERATORS AND OPERATOR KERNELS
/// @brief Read model parameters from input stream
/// @tparam U1_sector U(1) symmetry sector as teamplate input 
/// @param os input stream to read parameters
std::istream& XYZ::read(std::istream& os)
{
    
    return os;
}

/// @brief Write hamiltonian to stream as human readable
/// @param os input stream to read parameters
std::ostream& XYZ::write(std::ostream& os) const
{
    printSeparated(os, "\t", 16, true, "Model:", "XYZ spin chain");
    os << std::endl;
    printSeparated(os, "\t", 16, true, "Hamiltonian:", "H = \u03A3_r J_r\u03A3_i [ (1-\u03B7_r) S^x_i S^x_i+1 + (1+\u03B7_r) S^y_i S^y_i+1) + \u0394_r S^z_iS^z_i+1] + \u03A3_i h_i S^z_i");
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
    printSeparated(os, "\t", 16, true, "Parameters:");
    printSeparated(os, "\t", 16, true, "L", this->system_size);
    printSeparated(os, "\t", 16, true, "J_1", this->_J1);
    printSeparated(os, "\t", 16, true, "\u0394_1", this->_delta1);
    printSeparated(os, "\t", 16, true, "\u03B7_1", this->_eta1);
    printSeparated(os, "\t", 16, true, "w", this->_w);
    printSeparated(os, "\t", 16, true, "hz", this->_hz);
    printSeparated(os, "\t", 16, true, "hx", this->_hx);
    //printSeparated(os, "\t", 16, true, "disorder", this->_disorder.t());

    printSeparated(os, "\t", 16, true, "J_2", this->_J2);
    printSeparated(os, "\t", 16, true, "\u0394_2", this->_delta2);
    printSeparated(os, "\t", 16, true, "\u03B7_2", this->_eta2);
    printSeparated(os, "\t", 16, true, "seed", this->_seed);
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
    printSeparated(os, "\t", 16, true, "----------------------------------------------------------------------------------------------------");

    return os;
}
