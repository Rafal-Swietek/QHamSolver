#pragma once

#ifndef _HILBERT_BASE
    #include "_base.hpp"
#endif

class point_symmetric : public hilbert_space_base{
	v_1d<op::genOp> _symmetry_group;
	v_1d<int> _sectors;

	int k_sector = 0;		// quasimomentum symmetry ector
	int _boundary_cond = 1;	// 1-OBC, 0-PBC

	void generate_symmetry_goup(const v_1d<op::genOp>& sym_gen);

	/// @brief 
	virtual void init() override {

        this->create_basis();
    }
public:
	point_symmetric(int L, const v_1d<op::genOp>& sym_gen, int _BC = 1, int k_sector = 0);
};


//<! ---------------------------------------------------------------------------------------------------------------------------------------
//<! ------------------------------------------------------------------------------------------------------------------------ IMPLEMENTATION

/// @brief Constructor for point symmetric hilbert space with input generators (combinations are set within)
/// @param L system size
/// @param sym_gen list of symmetry generators (i.e. T, P, Z, ...)
/// @param _BC 
/// @param k_sec 
point_symmetric::point_symmetric(int L, const v_1d<op::genOp>& sym_gen, int _BC, int k_sec)
{
	this->system_size = L;
	this->_boundary_cond = _BC;
	if(this->_boundary_cond == 0)
		this->k_sector = k_sec;
	this->generate_symmetry_goup(sym_gen);
	this->init();
}

/// @brief 
/// @param sym_gen 
void point_symmetric::generate_symmetry_goup(const v_1d<op::genOp>& sym_gen)
{
	this->_symmetry_group = v_1d<op::genOp>();

	// set combinations of available symmetries
	const int NUM_OF_GENERATORS = sym_gen.size();
	for (int k = 0; k < NUM_OF_GENERATORS; k++) { // loop over all symmetries (no combinations)
		auto sym = sym_gen[k];
		for (int i = 0; i < NUM_OF_GENERATORS - k; i++) { // length of product
			auto sym_temp = sym;
			for (int j = k + 1; j < k + 1 + i; j++) // product off all possible lengths: 1,12,123,1234,... with no repetition starting at k
				sym_temp = sym_temp * sym_gen[j];
			this->_symmetry_group.push_back(sym_temp);
		}
	}

	//<! collect all sectors (Z2 sectors are +-1)
	for(auto& G : sym_gen)
		this->_sectors.emplace_back((int)std::real(chi(G)));
	
	// set combination of all syms with all translations
	if (this->_boundary_cond == 0) {
		op::genOp T = op::_translation_symmetry(this->system_size, this->k_sector);
		for (int l = 1; l < this->system_size; l++) {
			for (auto& G : this->_symmetry_group){
				op::genOp res = T * G;
				this->_symmetry_group.push_back(res);
			}
			T = T * T;
		}

		//<! append quasimomentum sector
		this->_sectors.emplace_back(this->k_sector);
	}

}