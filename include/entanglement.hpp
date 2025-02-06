
//<! TODO: CLEAN UP AND MAKE USE OF ENUM CLASS SO EACH FUNCTION CAN ACCESS SCHMIDT DECOMPOSITION
enum class methods{
    schmidt_decomposition,
    reduced_density_matrix
};

namespace ReducedDensityMatrix{

    /// @brief Calculate the reduced density matrix of a subsystem with set size
    /// @tparam _ty template element type
    /// @param state input state to calculate reduced matrix for
    /// @param A_size subsystem size
    /// @param L system size
    /// @param permutation permutation from partition to bipartition (i.e. (ones are subsystem A), 00110101 -> p(i) = {2, 3, 5, 7, 0, 1, 4, 6} )
    /// @return Reduced density matrix for input state
    template <typename _ty>
    inline
    auto reduced_density_matrix(
        const arma::Col<_ty>& state,
        int A_size,
        unsigned int L,
        QOps::generic_operator<> permutation = QOps::generic_operator<>()
        ) 
        -> arma::Mat<_ty> 
        {
    	// set subsytsems size
    	const long long dimA = (ULLPOW( (block_size *      A_size ) ));
    	const long long dimB = (ULLPOW( (block_size * (L - A_size)) ));
        
        const long long N = dimA * dimB;
        arma::Mat<_ty> rho(dimA, dimA, arma::fill::zeros);
        for (long long n = 0; n < N; n++) {						// loop over configurational basis
            long long counter = 0;
            u64 p_n = std::get<0>(permutation(n));
            for (long long m = p_n % dimB; m < N; m += dimB) {		    // pick out state with same B side (last L-A_size bits)
                long idx = p_n / dimB;							        // find index of state with same B-side (by dividing the last bits are discarded)
                rho(idx, counter) += my_conjungate(state(p_n)) * state(m);
                counter++;										        // increase counter to move along reduced basis
            }
        }
    	return rho;	
    }

    /// @brief Calculate the eigenvalues of the reduced density matrix of a subsystem with set size
    /// @tparam _ty template element type
    /// @param state input state to calculate reduced matrix for
    /// @param A_size subsystem size
    /// @param L system size
    /// @param permutation permutation from partition to bipartition (i.e. (ones are subsystem A), 00110101 -> p(i) = {2, 3, 5, 7, 0, 1, 4, 6} )
    /// @return Reduced density matrix for input state
    template <typename _ty>
    inline
    auto reduced_density_matrix_eigvals(
        const arma::Col<_ty>& state,
        int A_size,
        unsigned int L,
        QOps::generic_operator<> permutation = QOps::generic_operator<>()
        ) 
        -> arma::vec
        {
        arma::Mat<_ty> rho = reduced_density_matrix(state, A_size, L, permutation);
        arma::vec eigvals = arma::eig_sym(rho);	
    	return eigvals;
    }

    /// @brief Calculates the Schmidt decomposition of a wavefunction and returns the Schmidt values
    /// @tparam _ty input state type
    /// @param state input state in full Hilbert space
    /// @param A_size subsystem size
    /// @param L system size
    /// @return entanglement entropy
    template <typename _ty>
    inline
    auto _schmidt_decomposition(
        const arma::Col<_ty>& state,
        int A_size,
        unsigned int L
        ) -> arma::vec
    {
    	const long long dimA = (ULLPOW( (block_size *      A_size ) ));
    	const long long dimB = (ULLPOW( (block_size * (L - A_size)) ));

        // reshape array to matrix
        arma::Mat<_ty> rho = arma::reshape(state, dimA, dimB);

        // get schmidt coefficients from singular-value-decomposition
        arma::vec schmidt_coeff = arma::svd(rho);
        return arma::square( arma::abs(schmidt_coeff) );
    }

    /// @brief Calculate eigenvalues the reduced density matrix of a subsystem with set size (different methods)
    /// @tparam _ty template element type
    /// @tparam _method_ enum to choose method to find eigevalues of reduced density matrix
    /// @param state input state to calculate reduced matrix for
    /// @param A_size subsystem size
    /// @param L system size
    /// @param permutation permutation from partition to bipartition (i.e. (ones are subsystem A), 00110101 -> p(i) = {2, 3, 5, 7, 0, 1, 4, 6} )
    /// @return Reduced density matrix for input state
    template <typename _ty, methods _method_ = methods::schmidt_decomposition>
    inline
    auto get_eigvals(
        const arma::Col<_ty>& state,
        int A_size,
        unsigned int L,
        QOps::generic_operator<> permutation = QOps::generic_operator<>()
        ) 
        -> arma::vec
        {
        
        arma::vec probabilities;
        if constexpr (_method_ == methods::reduced_density_matrix)
            probabilities = ReducedDensityMatrix::reduced_density_matrix_eigvals(state, A_size, L, permutation);
        else if constexpr (_method_ == methods::schmidt_decomposition)
            probabilities = ReducedDensityMatrix::_schmidt_decomposition(state, A_size, L);
        else
            _assert_(true, "\t\tNo method available. Only construction of RDM or schmidt decomposition available.");                                       
            //full_map.empty()?  : entaglement::reduced_density_matrix_sym(state, A_size, L, full_map);
    	
    	return probabilities;
    }

};

// enum class measure{
//     vonNeumann,
//     Shannon,
//     reyni
// };

//<! entropy calculations from the reduced density matrix
namespace entropy{
    
    //<! -------------------------------------------------- von Neumann ENTROPY --------------------------------------------------
    /// @brief von Neumann entropy for a set of probabilities
    /// @param probabilities vector of probabilties (eigenvalues of reduceed density matrix)
    /// @return von Neumann entanglement entropy
    inline 
    double vonNeumann(
        const arma::vec& probabilities
        ){
    	
    	double entropy = 0;
    	for (int i = 0; i < probabilities.size(); i++) {
    		auto value = probabilities(i);
    		entropy += (abs(value) > 0) ? -value * log(abs(value)) : 0;
    	}
    	//double entropy = -real(trace(rho * real(logmat(rho))));
    	return entropy;
    }

    /// @brief von Neumann entropy
    /// @tparam _ty input state type
    /// @tparam _method_ enum to choose method to find eigevalues of reduced density matrix
    /// @param state input state in full Hilbert space
    /// @param A_size subsystem size
    /// @param L system size
    /// @param permutation permutation from partition to bipartition (i.e. (ones are subsystem A), 00110101 -> p(i) = {2, 3, 5, 7, 0, 1, 4, 6} )
    /// @return von Neumann entanglement entropy
    template <typename _ty, methods _method_ = methods::schmidt_decomposition>
    inline 
    double vonNeumann(
        const arma::Col<_ty>& state,
        int A_size,
        unsigned int L,
        QOps::generic_operator<> permutation = QOps::generic_operator<>()
        ){
    	
        arma::vec probabilities = ReducedDensityMatrix::get_eigvals<_ty, _method_>(state, A_size, L, permutation);
    	return vonNeumann(probabilities);
    }
    
    //<! von Neumann entropy for each subsystem size

    /// @brief von Neumann entropy for all subsystem sizes
    /// @tparam _ty input state type
    /// @param state input state in full Hilbert space
    /// @param L system size
    /// @param permutation permutation from partition to bipartition (i.e. (ones are subsystem A), 00110101 -> p(i) = {2, 3, 5, 7, 0, 1, 4, 6} )
    /// @return entanglement entropy
    template <typename _ty>
    inline
    arma::vec vonNeumann(
        const arma::Col<_ty>& state,
        unsigned int L,
        QOps::generic_operator<> permutation
    ){
    	arma::vec _entropy(L - 1, arma::fill::zeros);
    //#pragma omp parallel for
    	for (int i = 0; i < L - 1; i++)
    		_entropy(i) = vonNeumann(state, i + 1, L, permutation);
    	return _entropy;
    }
    
    //<! -------------------------------------------------- REYNI ENTROPY --------------------------------------------------
    //<! Reyni entropy related to multifractality of the Hilbert space: S = 1 / (1 - a) * log2( Tr( q^a ) )

    /// @brief reyni entropy from input probabilities
    /// @param probabilities vector of probabilties (eigenvalues of reduceed density matrix)
    /// @param alfa reyni entropy order
    /// @return reyni entanglement entropy
    inline
    double reyni_entropy(
        const arma::vec& probabilities,
        int alfa
        ) {
        _assert_(alfa > 1, "Only alfa>=2 powers are possible");
    	
        arma::vec probabilities_alfa = arma::pow(probabilities, alfa);
        return std::log2(std::real(arma::sum(probabilities_alfa))) / (1.0 - alfa);
    }

    /// @brief reyni entropy related to multifractality of the Hilbert space: S = 1 / (1 - a) * log2( Tr( q^a ) )
    /// @tparam _ty input state type
    /// @tparam _method_ enum to choose method to find eigevalues of reduced density matrix
    /// @param state input state in full Hilbert space
    /// @param A_size subsystem size
    /// @param L system size
    /// @param alfa reyni entropy order
    /// @param permutation permutation from partition to bipartition (i.e. (ones are subsystem A), 00110101 -> p(i) = {2, 3, 5, 7, 0, 1, 4, 6} )
    /// @return entanglement entropy
    template <typename _ty, methods _method_ = methods::schmidt_decomposition>
    inline
    double reyni_entropy(
        const arma::Col<_ty>& state,
        int A_size,
        unsigned int L,
        int alfa,
        QOps::generic_operator<> permutation
        ) {
        _assert_(alfa > 1, "Only alfa>=2 powers are possible");
    	
        arma::vec probabilities = ReducedDensityMatrix::get_eigvals<_ty, _method_>(state, A_size, L, permutation);
        return reyni_entropy(probabilities, alfa);
    }
    
    //<! -------------------------------------------------- SHANNON ENTROPY --------------------------------------------------
    //<! Shannon entropy used in information theory
    //<!    q = \sum_n w_n |n><n|
    //<!    S = -\sum_n Tr( |w_n|^2 log2(|w_n|^2) )

    /// @brief Shannon entropy from input probabilities
    /// @param probabilities vector of probabilties (eigenvalues of reduceed density matrix)
    /// @return Shannon entanglement entropy
    template <typename _ty>
    inline
    double shannon_entropy(
        const arma::vec& probabilities
        ) {
        
        double _entropy = 0;
        for (int i = 0; i < probabilities.size(); i++) {
        	auto value = abs(probabilities(i) * probabilities(i));
        	_entropy += ((value) < 1e-10) ? 0 : -value * log2(value);
        }
        return _entropy;
    }

    /// @brief Shannon entropy used in information theory
    /// @tparam _ty input state type
    /// @tparam _method_ enum to choose method to find eigevalues of reduced density matrix
    /// @param state input state in full Hilbert space
    /// @param A_size subsystem size
    /// @param L system size
    /// @param permutation permutation from partition to bipartition (i.e. (ones are subsystem A), 00110101 -> p(i) = {2, 3, 5, 7, 0, 1, 4, 6} )
    /// @return entanglement entropy
    template <typename _ty, methods _method_ = methods::schmidt_decomposition>
    inline
    double shannon_entropy(
        const arma::Col<_ty>& state,
        int A_size,
        unsigned int L,
        QOps::generic_operator<> permutation
        ) {
        
    	arma::vec probabilities = ReducedDensityMatrix::get_eigvals<_ty, _method_>(state, A_size, L, permutation);
        double _entropy = 0;
    // #pragma omp parallel for reduction(+: _entropy)
        for (int i = 0; i < probabilities.size(); i++) {
        	auto value = abs(probabilities(i) * probabilities(i));
        	_entropy += ((value) < 1e-10) ? 0 : -value * log2(value);
        }
        return _entropy;
    }
    
    /// @brief Calculates the entropy using the Schmidt decomposition of a wavefunction
    /// @tparam _ty input state type
    /// @param state input state in full Hilbert space
    /// @param A_size subsystem size
    /// @param L system size
    /// @return entanglement entropy
    template <typename _ty>
    inline
    auto schmidt_decomposition(
        const arma::Col<_ty>& state,
        int A_size,
        unsigned int L
        )
    {
    	const long long dimA = (ULLPOW( (block_size *      A_size ) ));
    	const long long dimB = (ULLPOW( (block_size * (L - A_size)) ));

        // reshape array to matrix
        arma::Mat<_ty> rho = arma::reshape(state, dimA, dimB);

        // get schmidt coefficients from singular-value-decomposition
        arma::vec schmidt_coeff = arma::svd(rho);

        //calculate entropy
        double entropy = 0;
    // #pragma omp parallel for reduction(+: entropy)
    	for (int i = 0; i < schmidt_coeff.size(); i++) {
    		auto value = std::abs(schmidt_coeff(i)) * std::abs(schmidt_coeff(i));
    		entropy += (abs(value) > 0) ? -value * std::log(value) : 0;
    	}
        return entropy;
    }
};









    // /// @brief Calculate the reduced density matrix of a subsystem with set size for global symmetries (like U(1))
    // /// @tparam _ty template element type
    // /// @param state input state to calculate reduced matrix for
    // /// @param A_size subsystem size
    // /// @param L system size
    // /// @param full_map mapping to reduced hilbert space basis
    // /// @param permutation permutation from partition to bipartition (i.e. (ones are subsystem A), 00110101 -> p(i) = {2, 3, 5, 7, 0, 1, 4, 6} )
    // /// @return Reduced density matrix for input state
    // template <typename _ty>
    // inline
    // auto reduced_density_matrix_sym(
    //     const arma::Col<_ty>& state,
    //     int A_size,
    //     unsigned int L,
    //     const v_1d<u64>& full_map,
    //     std::string bit_mask = ""
    //     ) 
    //     -> arma::Mat<_ty>
    //     {
    // 	// set subsytsems size
    // 	const long long dimA = (ULLPOW( (block_size *      A_size ) ));
    // 	const long long dimB = (ULLPOW( (block_size * (L - A_size)) ));
    //     const long long full_dim = dimA * dimB;
    //     const long long N = full_map.size();
    //     auto find_index = [&](u64 index){   return binary_search(full_map, 0, N - 1, index);  };
    // 	arma::Mat<_ty> rho(dimA, dimA, arma::fill::zeros);
    // 	for (long long n = 0; n < N; n++) {						// loop over configurational basis
    // 		long long counter = 0;
    //         const u64 true_n = full_map[n];
    // 		for (long long j = true_n % dimB; j < full_dim; j += dimB) {	// pick out state with same B side (last L-A_size bits)
    // 			long idx = true_n / dimB;
    //             long long m = find_index(j);
    //             if(m >= 0)
    //                 rho(idx, counter) += my_conjungate(state(n)) * state(m);
    //             counter++;  // increase counter to move along reduced basis
    // 		}
    // 	}
    // 	return rho;	
    // }