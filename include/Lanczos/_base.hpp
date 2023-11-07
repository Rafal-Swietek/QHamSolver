#pragma once
#ifndef _LANCZOS
#define _LANCZOS

namespace lanczos {
	
	template <typename _ty>
	class Lanczos;	//<! forward declaration for FTLM
};
//#include "FTLM.hpp"						//<! Finite-Temperature Lanczos Method

/// <summary>
/// LANCZOS CLASS
/// </summary>
namespace lanczos {

	template <typename _ty>
	class Lanczos {

		arma::Mat<_ty> krylov_space;				//<! krylov matrix - basis transformation
		arma::Mat<_ty> H_lanczos;					//<! Lanczos matrix for tridiagonalization
		arma::Mat<_ty> eigenvectors;				//<! eigenvectors from diagonalizing lanczos matrix
		arma::SpMat<_ty> H;							//<! hamiltonian matrix -- change to operator instance
		arma::vec eigenvalues;						//<! lanczos eignevalues
		
		arma::Col<_ty> initial_random_vec;			//<! initial random vector
		arma::Col<_ty> randVec_inKrylovSpace;		//<! random vector written in lanczos basis (used in FTLM)

		disorder<_ty> generator;			//<! random variable generator -- uniform distribution
		u64 N;								//<! dimension of hilbert space

		long _seed = std::random_device{}();     	// seed for random generator
		int lanczos_steps = 200;					// number of lanczos iterations
		int random_steps  = 1;						// number of random vectors in FTLM
		bool memory_over_performance = false;		// building hamiltonian as sparse (false) or diagonalizing on-the-fly (true)
		bool use_reorthogonalization = true;		// parameter to define whether use full reorthogonalization
		bool use_krylov;							//<! boolean value whether useing krylov matrix or not

		//! ----------------------------------------------------- PRIVATE BUILDERS / INITIALISERS
		void initialize();
		void build_lanczos();
		void build_krylov();

		void orthogonalize(
			arma::Col<_ty>& vec_to_ortho,  //<! vector to orthogonalize
			int j							 //<! current dimension of Krylov space
		);

	public:
		auto get_eigenvalues() 				const { return this->eigenvalues; }
		auto get_eigenstate(int _id = 0) 	const { return conv_to_hilbert_space(_id); }
		auto get_krylov()					const { return this->krylov_space; }
		auto get_lanczos_matrix()			const { return this->H_lanczos; }
		//friend _returnTy FTLM(Lanczos&);
		//------------------------------------------------------------------------------------------------ CONSTRUCTOS
		~Lanczos() = default;
		Lanczos() = delete;

		/// @brief Constructor of Lanczos class
		/// @param hamiltonian INput Hamiltonian matrix as sparse matrix
		/// @param M number of lanczos steps
		/// @param R number of random realizations of lanczos steps (for FTLM and dynamics)
		/// @param random_vec input random state
		/// @param seed input seed for random computation
		/// @param use_reortho (boolean) use reorthogonalization scheme and keep all Krylov states?
		/// @param mem_over_perf (boolean) use memory over performance (matrix-vector product on the fly), disables use_reortho
		explicit Lanczos(
			const arma::SpMat<_ty>& hamiltonian, 
			int M, int R, 
			int seed = std::random_device{}(), 
			bool use_reortho = false, 
			bool mem_over_perf = false,
			const arma::Col<_ty>& random_vec = arma::Col<_ty>()
		) 
			: H(hamiltonian), lanczos_steps(M), random_steps(R), _seed(seed),
			use_reorthogonalization(use_reortho), memory_over_performance(mem_over_perf),
			initial_random_vec(random_vec)
		{ initialize(); }

		Lanczos(const Lanczos& input_model) = default;
		Lanczos(Lanczos&& input_model) noexcept = default;
		auto operator=(const Lanczos& input_model){ return Lanczos(input_model); };
		//auto operator=(Lanczos&& input_model) noexcept { return Lanczos(std::move(input_model)); };
		//------------------------------------------------------------------------------------------------ DIAGONALIZING MATRIX
		void build(const arma::Col<_ty>& random_vec);
		void build();

		void diagonalization(const arma::Col<_ty>& random_vec);
		void diagonalization();
		//TODO: some methods with return values

		//------------------------------------------------------------------------------------------------ CAST STATES TO ORIGINAL HILBERT SPACE:
		//enum class base_type {
		//	hilbert,	//<! Hilbert basis, i.e. computational basis
		//	krylov		//<! Krylov basis build from random vector
		//};
		[[nodiscard]] auto conv_to_hilbert_space(int state_id) const -> arma::Col<_ty>;

		[[nodiscard]] auto conv_to_hilbert_space(const arma::Col<_ty>& input) const -> arma::Col<_ty>;
		[[nodiscard]] auto conv_to_krylov_space( const arma::Col<_ty>& input) const -> arma::Col<_ty>;
		// [[nodiscard]] auto conv_to_hilbert_space(const arma::vec&	 input)   const -> arma::Col<_ty>;
		// [[nodiscard]] auto conv_to_krylov_space( const arma::vec&	 input)   const -> arma::Col<_ty>;
		
		//------------------------------------------------------------------------------------------------ TOOLS
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
#include "dynamics.hpp"			//<! implementation of dynamic quantities

#endif