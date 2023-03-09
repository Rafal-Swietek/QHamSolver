#pragma once

#ifndef _HILBERT_BASE
    #include "_base.hpp"
#endif

#ifndef _HILBERT_SYM
#define _HILBERT_SYM
#include "../operators/symmetries.hpp"

#ifdef USE_REAL_SECTORS
    using elem_ty = double;
	#pragma message ("--> Using real symmetry transformation")
#else
    using elem_ty = cpx;
#endif

class point_symmetric : public hilbert_space_base{
	v_1d<op::genOp> _symmetry_group;
	v_1d<elem_ty> _normalisation;
	v_1d<int> _sectors;

	int k_sector = 0;			//<! quasimomentum symmetry ector
	int _boundary_cond = 1;		//<! 1-OBC, 0-PBC
	int _pos_of_parity = -1;	//<! position of parity generator in input symmetries (if not present put -1)
	bool _real_k_sector = 1;	//<! is the k_sector real or complex?, i.e. k = 0, pi
	
	void generate_symmetry_goup(const v_1d<op::genOp>& sym_gen);
	auto get_symmetry_normalization(u64 base_idx) const -> elem_ty;

	/// @brief 
	virtual void init() override 
		{ this->create_basis(); }
	
	typedef std::pair<u64, elem_ty> return_type;		// return type of operator, resulting state and value
public:
    point_symmetric() = default;
	point_symmetric(unsigned int L, const v_1d<op::genOp>& sym_gen, int _BC = 1, int k_sector = 0, int pos_of_parity = -1);
    
    virtual void create_basis() override;

	virtual u64 operator()(u64 idx) override;
	virtual u64 find(u64 element) override;

	return_type find_SEC_representative(u64 base_idx) const;

	auto get_symmetry_group() const { return this-> _symmetry_group; }
	auto get_normalisation()  const { return this-> _normalisation; }
	auto get_norm(u64 idx)	  const { return this-> _normalisation[idx]; }
	auto get_sectors() 		  const { return this-> _sectors; }

	arma::SpMat<elem_ty> symmetry_rotation() const;
};


//<! ---------------------------------------------------------------------------------------------------------------------------------------
//<! ------------------------------------------------------------------------------------------------------------------------ IMPLEMENTATION

/// @brief Constructor for point symmetric hilbert space with input generators (combinations are set within)
/// @param L system size
/// @param sym_gen list of symmetry generators (i.e. T, P, Z, ...)
/// @param _BC boundary condition (0-PBC, 1-OBC, ...) (default = 1)
/// @param k_sec quasimomentum symmetyr sector
/// @param pos_of_parity position of parity in sym_gen (if not present -> -1, by default -1)
inline
point_symmetric::point_symmetric(unsigned int L, const v_1d<op::genOp>& sym_gen, int _BC, int k_sec, int pos_of_parity)
{
	this->system_size = L;
	this->_boundary_cond = _BC;
	if(this->_boundary_cond == 0){ // if PBC
		this->k_sector = k_sec;
		this->_real_k_sector = (this->k_sector == 0) || (this->k_sector  == this->system_size / 2.);
	}
	
	// set position of parity symmetry in list of generators
	this->_pos_of_parity = pos_of_parity;

	this->generate_symmetry_goup(sym_gen);
	this->init();
}

//<! ------------------------------------------------------------------------------------------ SYMMETRY GROUP
/// @brief Generate symmetry group with all combinations of symmetry generators
/// @param sym_gen list of symmetry generators (shall not include translation! )
inline
void 
point_symmetric::generate_symmetry_goup(const v_1d<op::genOp>& sym_gen_in)
{
	this->_symmetry_group = v_1d<op::genOp>();
	v_1d<op::genOp> sym_gen = sym_gen_in;

	// remove parity for complex quasimomentum sectors
	if (!this->_real_k_sector && this->_pos_of_parity >= 0){
		std::cout << "Working in imaginary sector. Removing parity" << std::endl;
		sym_gen.erase(sym_gen.begin() + this->_pos_of_parity);
	}

	// add neutral element
	this->_symmetry_group.push_back(op::genOp(this->system_size));
	
	// set combinations of available symmetries
    const int NUM_OF_GENERATORS = sym_gen.size();
	for (int k = 1; k <= NUM_OF_GENERATORS; k++) {  // loop over all product length
        std::string bitmask(k, 1);              	// K leading 1's
        bitmask.resize(NUM_OF_GENERATORS, 0);   	// N - K trailing 0's
        
        do {
            op::genOp sym_temp(this->system_size);
            for (int i = 0; i < NUM_OF_GENERATORS; ++i) // [0..N-1] integers
                if (bitmask[i])
                    sym_temp %= sym_gen[i];
            
			this->_symmetry_group.push_back(sym_temp);
        } while (std::prev_permutation(bitmask.begin(), bitmask.end())); // loop over all combinations with bitmask
	}

	//<! collect all sectors (Z2 sectors are +-1)
	for(auto& G : sym_gen)
		this->_sectors.emplace_back((int)std::real(chi(G)));
	
	// set combination of all syms with all translations
	if (this->_boundary_cond == 0) {
		v_1d<op::genOp> sym_group_copy = this->_symmetry_group;
		op::genOp translation = op::_translation_symmetry(this->system_size, this->k_sector);
		for (int l = 1; l < this->system_size; l++) 
		{
			for (auto& G : sym_group_copy)
				this->_symmetry_group.push_back(translation % G);

			//this->_symmetry_group.push_back(translation);
			translation %= op::_translation_symmetry(this->system_size, this->k_sector);
		}

		//<! append quasimomentum sector
		this->_sectors.emplace_back(this->k_sector);
	}

}

/// @brief Find super-equivalent class (SEC) representative for given set of states related by symmetry transformations
/// @param base_idx find SEC for given input state
/// @return SEC
inline
point_symmetric::return_type 
point_symmetric::find_SEC_representative(u64 base_idx) const 
{
	u64 SEC = INT64_MAX;
	cpx return_val = 1.0;
	for( auto &G : this->_symmetry_group){
		auto [state, return_value] = G(base_idx);
		if (state < SEC) {
			SEC = state;
			return_val = return_value;
		}
	}
	#ifdef USE_REAL_SECTORS
		return std::make_pair(SEC, std::real(return_val));
	#else
		return std::make_pair(SEC, return_val);
	#endif
}

/// @brief Calculate normalisation for input state (sum off all symmetry eigenvalues for generators not changing input state)
/// @param base_idx input state
/// @return normalisation
inline
elem_ty 
point_symmetric::get_symmetry_normalization(u64 base_idx) const 
{
	elem_ty normalisation = 0.0;
	//for (unsigned int L = 0; l < this->_symmetry_group.size(); l++) {
	for( auto &G : this->_symmetry_group){
		auto [state, return_value] = G(base_idx);
		#ifdef USE_REAL_SECTORS
			if (state == base_idx)
				normalisation += std::real(return_value);
		#else
			if (state == base_idx)
				normalisation += return_value;
		#endif
	}
	return std::sqrt(normalisation);
}

/// @brief Generate Unitary transformation to full hilbert space from reduced basis
/// @return unitary transformation U
inline
arma::SpMat<elem_ty>
point_symmetric::symmetry_rotation() const
{
	const u64 dim_tot = ULLPOW(this->system_size);
	arma::SpMat<elem_ty> U(dim_tot, this->dim);
#pragma omp parallel for
	for (long int k = 0; k < this->dim; k++) {
		for (auto& G : this->_symmetry_group) {
			auto [idx, sym_eig] = G(this->mapping[k]);
		
		#ifdef USE_REAL_SECTORS
			if(idx < dim_tot) // only if exists in sector
				U(idx, k) += std::real(sym_eig / (this->_normalisation[k] * std::sqrt(this->_symmetry_group.size())));
		#else
			if(idx < dim_tot) // only if exists in sector
				U(idx, k) += std::conj(sym_eig / (this->_normalisation[k] * std::sqrt(this->_symmetry_group.size())));
		#endif
			// CONJUNGATE YOU MORON CAUSE YOU RETURN TO FULL STATE, I.E. INVERSE MAPPING!!!!!! 
		}
	}
	return U;
}

//<! ------------------------------------------------------------------------------------------ BASIS CONSTRUCTION
/// @brief Creates hilbert space basis with given point symmetries
inline
void point_symmetric::create_basis()
{
	//<! kernel for multithreaded mapping generation
	auto mapping_kernel = [this](u64 start, u64 stop, std::vector<u64>& map_threaded, std::vector<elem_ty>& norm_threaded)
	{
		for (u64 j = start; j < stop; j++){
			auto SEC = std::get<0>(find_SEC_representative(j));
			if (SEC == j) {
				elem_ty N = get_symmetry_normalization(j);					// normalisation condition -- check if state in basis
				if (std::abs(N) > 1e-6) {
					map_threaded.push_back(j);
					norm_threaded.push_back(N);
				}
			}
		}
		//std::cout << map_threaded << std::endl;
	};
	u64 start = 0, stop = ULLPOW(this->system_size);
	u64 _powL = BinaryPowers[this->system_size];		// maximal power (dimension without symmetries)
	if (num_of_threads == 1)
		mapping_kernel(start, stop, this->mapping, this->_normalisation);
	else {
		//Threaded
		v_2d<u64> map_threaded(num_of_threads);
		v_2d<elem_ty> norm_threaded(num_of_threads);
		std::vector<std::thread> threads;
		threads.reserve(num_of_threads);
		for (int t = 0; t < num_of_threads; t++) {
			start = (u64)(_powL / (double)num_of_threads * t);
			stop = ((t + 1) == num_of_threads ? _powL : u64(_powL / (double)num_of_threads * (double)(t + 1)));
			map_threaded[t] = v_1d<u64>();
			norm_threaded[t] = v_1d<elem_ty>();
			threads.emplace_back(mapping_kernel, start, stop, ref(map_threaded[t]), ref(norm_threaded[t]));
		}
		for (auto& t : threads) t.join();

		for (auto& t : map_threaded)
			this->mapping.insert(this->mapping.end(), std::make_move_iterator(t.begin()), std::make_move_iterator(t.end()));
		
		for (auto& t : norm_threaded)
			this->_normalisation.insert(this->_normalisation.end(), std::make_move_iterator(t.begin()), std::make_move_iterator(t.end()));

	}
	this->dim = this->mapping.size();
}


//<! ------------------------------------------------------------------------------------------ ACCESS TOOLS
/// @brief Overloaded operator to access elements in hilbert space
/// @param idx Index of element in hilbert space
/// @return Element of hilbert space at position 'index'
inline
u64 
point_symmetric::operator()(u64 idx)
{ 
	_assert_((idx < this->dim), OUT_OF_MAP);
    return this->mapping[idx]; 
}


/// @brief Find index of element in hilbert space
/// @param element element to find its index
/// @return index of element 'element'
inline
u64 
point_symmetric::find(u64 element)
	{ return binary_search(this->mapping, 0, this->dim - 1, element); }










#endif