#pragma once
#ifndef _BLOCK_LANCZOS
#define _BLOCK_LANCZOS

namespace lanczos {
	
	template <typename _ty>
	class BlockLanczos;	//<! forward declaration for FTLM
};
//#include "FTLM.hpp"						//<! Finite-Temperature Lanczos Method

enum class base_type {
			hilbert,	//<! Hilbert basis, i.e. computational basis
			krylov		//<! Krylov basis build from random vector
		};
/// <summary>
/// BLOCK-LANCZOS CLASS
/// </summary>
namespace lanczos {

	/// @brief Class for Block-Lanczos calculation with flags for number of steps and bundle size
	/// @tparam _ty type of input Hamiltonian (enforces type on input states)
	template <typename _ty>
	class BlockLanczos {

		arma::Mat<_ty> krylov_space;				//<! krylov matrix - basis transformation
		arma::Mat<_ty> H_lanczos;					//<! Lanczos matrix for tridiagonalization
		arma::Mat<_ty> eigenvectors;				//<! eigenvectors from diagonalizing lanczos matrix
		arma::SpMat<_ty> H;							//<! hamiltonian matrix -- change to operator instance
		arma::vec eigenvalues;						//<! lanczos eignevalues
		
		arma::Mat<_ty> initial_bundle;			    //<! initial random vector
		// arma::Mat<_ty> randVec_inKrylovSpace;	//<! random vector written in lanczos basis (used in FTLM)

		disorder<_ty> generator;					//<! random variable generator -- uniform distribution
		u64 N;										//<! dimension of hilbert space

		long _seed = std::random_device{}();     	//<! seed for random generator
		int lanczos_steps = 200;					//<! number of lanczos iterations
		int random_steps  = 1;						//<! number of random vectors in FTLM
		int bundle_size  = 5;						//<! number of initial random vectors (number of columns in initial_bundle matrix)
		int matrix_size;							//<! size of lanczos matrix

		bool memory_over_performance = false;		//<! building hamiltonian as sparse (false) or diagonalizing on-the-fly (true)
		bool use_reorthogonalization = true;		//<! parameter to define whether use full reorthogonalization
		bool use_krylov;							//<! boolean value whether useing krylov matrix or not

		//! ----------------------------------------------------- PRIVATE BUILDERS / INITIALISERS
		void initialize();
		void build_lanczos();
		void build_krylov();

		void orthogonalize(arma::Col<_ty>& vec_to_ortho, int j);
		void orthogonalize(arma::Mat<_ty>& mat_to_ortho, int j);

	public:
		auto get_eigenvalues() 				const { return this->eigenvalues; }
		// auto get_eigenstate(int _id = 0) 	const { return conv_to_hilbert_space(_id); }
		auto get_krylov()					const { return this->krylov_space; }
		auto get_lanczos_matrix()			const { return this->H_lanczos; }
		//friend _returnTy FTLM(Lanczos&);
		//------------------------------------------------------------------------------------------------ CONSTRUCTOS
		~BlockLanczos() = default;
		BlockLanczos() = delete;

		/// @brief Constructor of Lanczos class
		/// @tparam _ty type of input Hamiltonian (enforces type on onput state)
		/// @param hamiltonian Input Hamiltonian matrix as sparse matrix
		/// @param M number of lanczos steps
		/// @param R number of random realizations of lanczos steps (for FTLM and dynamics)
		/// @param R number of initial states
		/// @param random_vec input random states (orthogonal matrix)
		/// @param seed input seed for random computation
		/// @param use_reortho (boolean) use reorthogonalization scheme and keep all Krylov states?
		/// @param mem_over_perf (boolean) use memory over performance (matrix-vector product on the fly), disables use_reortho
		explicit BlockLanczos(
			const arma::SpMat<_ty>& hamiltonian, 
			int M, int R, int s,
			int seed = std::random_device{}(), 
			bool use_reortho = false, 
			bool mem_over_perf = false,
			const arma::Mat<_ty>& initial_states = arma::Mat<_ty>()
		) 
			: H(hamiltonian), lanczos_steps(M), random_steps(R), bundle_size(s), _seed(seed),
			use_reorthogonalization(use_reortho), memory_over_performance(mem_over_perf),
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
		[[nodiscard]] 
		auto conv_to(const arma::Col<_ty>& input) const -> arma::Col<_ty>;

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

#include "construct_impl.hpp"	//<! constructors etc implemetation
#include "tools.hpp"			//<! lanczos tools implementation: i.e. convergence etc.
#include "build_impl.hpp"		//<! lanczos implementation with or without krylov space
#include "converter.hpp"		//<! casting vectoes inbetween krylov and hilbert spaces
#include "eigs.hpp"				//<! diagonalization of lanczos matrix
// #include "dynamics.hpp"			//<! implementation of dynamic quantities

#endif