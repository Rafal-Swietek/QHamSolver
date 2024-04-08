#pragma once
#ifndef _BLOCK_LANCZOS
#define _BLOCK_LANCZOS


namespace lanczos {

	/// @brief Class for Block-Lanczos calculation with flags for number of steps and bundle size
	/// @tparam _ty type of input Hamiltonian (enforces type on Krylov basis) 
	/// @tparam converge_type enum type for convergence criterion (energies or states)
	template <typename _ty, converge converge_type = converge::states>
	class BlockLanczos {

		//<! class object types
		typedef hamiltonian_func_ptr<arma::Mat<_ty>> _ham_func_ptr;

		//<! private members
		arma::Mat<_ty> krylov_space;				//<! krylov matrix - basis transformation
		arma::Mat<_ty> H_lanczos;					//<! Lanczos matrix for tridiagonalization
		arma::Mat<_ty> eigenvectors;				//<! eigenvectors from diagonalizing lanczos matrix
		arma::vec eigenvalues;						//<! lanczos eignevalues

		_ham_func_ptr Hamiltonian;					//<! hamiltonian function pointer to hamiltonian-vector product
		
		arma::Mat<_ty> initial_bundle;			    //<! initial random vector
		// arma::Mat<_ty> randVec_inKrylovSpace;	//<! random vector written in lanczos basis (used in FTLM)

		disorder<_ty> generator;					//<! random variable generator -- uniform distribution
		u64 N;										//<! dimension of hilbert space

		double tolerance  = 1e-14;					//<! tolerance determining preciaion of lanczos iteration with convergence
		long _seed 		  = std::random_device{}(); //<! seed for random generator
		int maxiter 	  = -1;						//<! maxiteration (by default = dimension of Hilbert space)
		unsigned int lanczos_steps = 200;			//<! number of lanczos iterations (or number of required eigenstates)
		unsigned int random_steps  = 1;				//<! number of random vectors in FTLM
		unsigned int bundle_size   = 5;				//<! number of initial random vectors (number of columns in initial_bundle matrix)
		unsigned int matrix_size;					//<! size of lanczos matrix

		bool use_krylov				= true;			//<! boolean value whether useing krylov matrix or not
		bool use_full_convergence 	= true;			//<! don't test convergence of states

		//! ----------------------------------------------------- PRIVATE BUILDERS / INITIALISERS
		void initialize();
		void _build_lanczos();
		void _build_krylov();
		void _build_lanczos_converged();
		void _build_krylov_converged();

		double _calculate_convergence(arma::vec& Eprev, const arma::Mat<_ty>& beta);

		void orthogonalize(arma::Col<_ty>& vec_to_ortho, int j);
		void orthogonalize(arma::Mat<_ty>& mat_to_ortho, int j);

	public:
		auto get_eigenvalues() 							const { return this->eigenvalues; }
		auto get_eigenstate(int _id = 0) 				const { return conv_to_hilbert_space(_id); }
		auto get_eigenstates() -> arma::Mat<_ty>;
		auto get_krylov()								const { return this->krylov_space; }
		auto get_lanczos_matrix()						const { return this->H_lanczos; }
		auto get_lanczossteps()							const { return this->lanczos_steps; }

		//friend _returnTy FTLM(Lanczos&);
		//------------------------------------------------------------------------------------------------ CONSTRUCTOS
		~BlockLanczos() { DESTRUCTOR_CALL; };
		BlockLanczos() = delete;

		/// @brief Constructor of Lanczos class
		/// @tparam _ty type of input Hamiltonian (enforces type on Krylov basis)
		/// @param H Input Hamiltonian matrix as sparse matrix
		/// @param M number of lanczos steps (or number of required eigenstates)
		/// @param s number of initial states (bundle size)
		/// @param max_iter maximal number of lanczos steps
		/// @param tol tolerance for convergence of iterative procedure
		/// @param seed input seed for random computation
		/// @param use_reortho (boolean) use reorthogonalization scheme and keep all Krylov states?
		/// @param random_vec input random state
		explicit BlockLanczos(
			const arma::SpMat<_ty>& H, 
			int M, int s, int max_iter = -1, double tol = 1e-14,
			int seed = std::random_device{}(), 
			bool use_reortho = false, 
			const arma::Mat<_ty>& initial_states = arma::Mat<_ty>()
		) 
			: lanczos_steps(M), bundle_size(s), _seed(seed), maxiter(max_iter), tolerance(tol),
			use_krylov(use_reortho),
			initial_bundle(initial_states)
		{ 
			this->N = H.n_cols;
			this->Hamiltonian = _ham_func_ptr( [&H](const arma::Mat<_ty>& matrix) -> arma::Mat<_ty> { return H * matrix; } );
			initialize(); 
		}

		// /// @brief Constructor of Lanczos class
		// /// @tparam _ty type of input Hamiltonian (enforces type on Krylov basis)
		// /// @param hamiltonian Input Hamiltonian matrix as sparse matrix
		// /// @param M number of lanczos steps (or number of required eigenstates)
		// /// @param s number of initial states (bundle size)
		// /// @param max_iter maximal number of lanczos steps
		// /// @param tol tolerance for convergence of iterative procedure
		// /// @param seed input seed for random computation
		// /// @param use_reortho (boolean) use reorthogonalization scheme and keep all Krylov states?
		// /// @param mem_over_perf (boolean) use memory over performance (matrix-vector product on the fly), disables use_reortho
		// /// @param random_vec input random state
		// template <callable_type callable>
		// explicit BlockLanczos(
		// 	callable&& H_mult_state, 
		// 	int M, int s, int max_iter = -1, double tol = 1e-14,
		// 	int seed = std::random_device{}(), 
		// 	bool use_reortho = false, 
		// 	bool mem_over_perf = false,
		// 	const arma::Mat<_ty>& initial_states = arma::Mat<_ty>()
		// ) 
		// 	: Hmultiply(hamiltonian_func_ptr(std::move(H_mult_state))), lanczos_steps(M), bundle_size(s), _seed(seed), maxiter(max_iter), tolerance(tol),
		// 	use_reorthogonalization(use_reortho), memory_over_performance(mem_over_perf),
		// 	initial_bundle(initial_states)
		// { initialize(); }

		/// @brief Constructor of Lanczos class
		/// @tparam _ty type of input Hamiltonian (enforces type on Krylov basis)
		/// @param H_mult_state Function pointer for Hamiltonian-vector product
		/// @param dimension dimension of Hilbert space for given hamiltonian (can't be established from the pointer)
		/// @param M number of lanczos steps (or number of required eigenstates)
		/// @param s number of initial states (bundle size)
		/// @param max_iter maximal number of lanczos steps
		/// @param tol tolerance for convergence of iterative procedure
		/// @param seed input seed for random computation
		/// @param use_reortho (boolean) use reorthogonalization scheme and keep all Krylov states?
		/// @param random_vec input random state
		explicit BlockLanczos(
			const _ham_func_ptr& H_mult_state, u64 dimension,
			int M, int s, int max_iter = -1, double tol = 1e-14,
			int seed = std::random_device{}(), 
			bool use_reortho = false, 
			const arma::Mat<_ty>& initial_states = arma::Mat<_ty>()
		) 
			: Hamiltonian(std::move(H_mult_state)), N(dimension),
			lanczos_steps(M), bundle_size(s), _seed(seed), maxiter(max_iter), tolerance(tol),
			use_krylov(use_reortho),
			initial_bundle(initial_states)
		{ initialize(); }

		BlockLanczos(const BlockLanczos& input_model) = default;
		BlockLanczos(BlockLanczos&& input_model) noexcept = default;
		auto operator=(const BlockLanczos& input_model){ return BlockLanczos(input_model); };
		//auto operator=(BlockLanczos&& input_model) noexcept { return BlockLanczos(std::move(input_model)); };

		//------------------------------------------------------------------------------------------------ DIAGONALIZING MATRIX
		void build(const arma::Mat<_ty>& random_vec);
		void build();

		void diagonalization(const arma::Mat<_ty>& random_bundle);
		void diagonalization();
		//TODO: some methods with return values

		//------------------------------------------------------------------------------------------------ CAST STATES TO ORIGINAL HILBERT SPACE:
		template <base_type base>
		[[nodiscard]] auto conv_to(const arma::Col<_ty>& input) const -> arma::Col<_ty>;

		[[nodiscard]] auto conv_to_hilbert_space(int state_id) const -> arma::Col<_ty>;

		[[nodiscard]] auto conv_to_hilbert_space(const arma::Col<_ty>& input) const -> arma::Col<_ty>;
		[[nodiscard]] auto conv_to_krylov_space( const arma::Col<_ty>& input) const -> arma::Col<_ty>;
		// [[nodiscard]] auto conv_to_hilbert_space(const arma::vec&	 input)   const -> arma::Col<_ty>;
		// [[nodiscard]] auto conv_to_krylov_space( const arma::vec&	 input)   const -> arma::Col<_ty>;
		
		// //------------------------------------------------------------------------------------------------ TOOLS
		void convergence(std::string dir, std::string name);
		
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
#include "tools.hpp"			//<! lanczos tools implementation: i.e. convergence etc.
#include "build_converged.hpp"	//<! lanczos implementation with or without krylov space with testing of convergence
#include "build.hpp"			//<! lanczos implementation with or without krylov space
#include "converter.hpp"		//<! casting vectoes inbetween krylov and hilbert spaces
#include "eigs.hpp"				//<! diagonalization of lanczos matrix
// #include "dynamics.hpp"			//<! implementation of dynamic quantities

#endif