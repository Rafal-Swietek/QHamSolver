#pragma once
#ifndef _POLFED_
#define _POLFED_

#ifndef _LANCZOS_PARAMS
    #include "../Lanczos/params.hpp"
#endif
#ifndef _LANCZOS
    #include "../Lanczos/_base.hpp"
#endif
#ifndef _BLOCK_LANCZOS
    #include "../LanczosBlock/_base.hpp"
#endif

#include "clenshaw.hpp"

namespace polfed {

	/// @brief Class for POLFED algorithm
	/// @tparam _ty type of input Hamiltonian (enforces type on input states)
	template <typename _ty, converge converge_type = converge::states>
	class POLFED {

		//<! class object types
		typedef hamiltonian_func_ptr<arma::Mat<_ty>> _ham_func_ptr;

		//<! private members
		const arma::SpMat<_ty>& H;					//<! original hamiltonian matrix -- change to operator instance
		arma::SpMat<_ty> P_H;						//<! hamiltonian matrix to tranform and work on -- change to operator instance
		// arma::SpMat<_ty> _dummy_H;					//<! dummy object to have H being reference to it when use_on_the_fly enabled
		arma::Col<_ty> coeff;			    		//<! coefficients of polynomial expansion
		u64 N;										//<! dimension of hilbert space
		
		_ham_func_ptr H_multiply;					//<! hamiltonian function pointer to hamiltonian-vector product
		_ham_func_ptr PH_multiply;					//<! hamiltonian function pointer to transfomred hamiltonian-vector product

		_ty sigma;										//<! target energy (for now set to mean energy)
		double tolerance 	= 1e-14;					//<! tolerance determining preciaion of lanczos iteration with convergence
		double cutoff 		= 0.17;						//<! cutoff value to estimate degree of polynomial
		long seed 			= std::random_device{}();   //<! seed for random generator
		int num_of_eigval 	= 200;						//<! number of lanczos iterations
		int bundle_size  	= 5;						//<! number of initial random vectors (number of columns in initial_bundle matrix)
		int K 				= -1;						//<! order of polynomial

		bool use_on_the_fly	= false;					//<! diagonalizing on-the-fly (not Hamiltonian as matrix is known)
		bool use_krylov 	= false;					//<! boolean value whether useing krylov matrix or not

		//! ----------------------------------------------------- PRIVATE BUILDERS / INITIALISERS
		void initialize();
		void set_poly_order(double Emin, double Emax);
		void transform_matrix();
		auto get_energy_bounds(int&) 	-> std::pair<double, double>;
		auto get_energy_bounds() 		-> std::pair<double, double>;
		// auto set_block_lanczos() 		-> lanczos::BlockLanczos<_ty>;

	public:

		//------------------------------------------------------------------------------------------------ CONSTRUCTOS
		~POLFED() { DESTRUCTOR_CALL; };
		POLFED() = delete;

		/// @brief Constructor of Lanczos class
		/// @tparam _ty type of input Hamiltonian (enforces type on Krylov basis)
		/// @param hamiltonian Input Hamiltonian matrix as sparse matrix
		/// @param Nev number of desired eigenstates
		/// @param s number of initial states (bundle size)
		/// @param K order of polynomial (maximal order of Chebyshev polynomial)
		/// @param p cutoff value to determine K
		/// @param tol tolerance for convergence of iterative procedure
		/// @param seed input seed for random computation
		/// @param use_reortho (boolean) use reorthogonalization scheme and keep all Krylov states?
		/// @param initial_states input random state
		explicit POLFED(
			const arma::SpMat<_ty>& hamiltonian, 
			int Nev, int s, int K, double tol = 1e-14, double p = 0.2,
			int seed_in = std::random_device{}(), 
			bool use_reortho = false
		) 
			: H(hamiltonian), num_of_eigval(Nev), bundle_size(s), K(K), seed(seed_in), 
			tolerance(tol), cutoff(p),
			use_krylov(use_reortho)
		{ initialize(); }

		// /// @brief Constructor of Lanczos class
		// /// @tparam _ty type of input Hamiltonian (enforces type on Krylov basis)
		// /// @param H_mult_state Function pointer for Hamiltonian-vector product
		// /// @param dimension dimension of Hilbert space for given hamiltonian (can't be established from the pointer)
		// /// @param Nev number of desired eigenstates
		// /// @param s number of initial states (bundle size)
		// /// @param K order of polynomial (maximal order of Chebyshev polynomial)
		// /// @param p cutoff value to determine K
		// /// @param tol tolerance for convergence of iterative procedure
		// /// @param seed input seed for random computation
		// /// @param use_reortho (boolean) use reorthogonalization scheme and keep all Krylov states?
		// /// @param initial_states input random state
		// explicit POLFED(
		// 	const _ham_func_ptr& H_mult_stat, u64 dimension, 
		// 	int Nev, int s, int K, double tol = 1e-14, double p = 0.2,
		// 	int seed_in = std::random_device{}(), 
		// 	bool use_reortho = false
		// ) 
		// 	: H_multiply(H_mult_stat), N(dimension),
		// 	num_of_eigval(Nev), bundle_size(s), K(K), seed(seed_in), 
		// 	tolerance(tol), cutoff(p),
		// 	use_krylov(use_reortho)
		// { 
		// 	this->use_on_the_fly = true;
		// 	this->initialize(); 
		// }
		// POLFED(const POLFED& input_model) = default;
		// POLFED(POLFED&& input_model) noexcept = default;
		// auto operator=(const POLFED& input_model){ return POLFED(input_model); };
		//auto operator=(BlockLanczos&& input_model) noexcept { return BlockLanczos(std::move(input_model)); };

		//------------------------------------------------------------------------------------------------ DIAGONALIZING MATRIX
		auto eig() -> std::pair<arma::vec, arma::Mat<_ty>>;
		void eig(arma::vec&, arma::Mat<_ty>&);
		void eig(arma::Mat<_ty>&);
		void eig(arma::vec&);

		//TODO: some methods with return values

		//------------------------------------------------------------------------------------------------ TOOLS
		// void convergence(std::string dir, std::string name);
		
		//------------------------------------------------------------------------------------------------ DYNAMICS
		// auto time_evolution_stationary(
		// 	const arma::Col<_ty>& input_state,
		// 	double time
		// ) -> arma::Col<_ty>;

		// auto time_evolution_non_stationary(
		// 	arma::Col<_ty>& prev_state,
		// 	double dt,
		// 	int lanczos_steps = 10
		// );
	};
};


#include "construct.hpp"		//<! constructors etc implemetation
#include "tools.hpp"			//<! tools implementation: i.e. energy bounds etc.
#include "preamble.hpp"			//<! preparation for Block-Lanczos
#include "eigs.hpp"				//<! finding eigenvalues and eigenstates

#endif