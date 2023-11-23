#pragma once
#ifndef _LANCZOS_PARAMS
#define _LANCZOS_PARAMS

enum class base_type {
		hilbert,	//<! Hilbert basis, i.e. computational basis
		krylov		//<! Krylov basis build from random vector
	};

enum class converge {
		energies,	//<! converge using eigenenergies (faster, but states not fully converged)
		states,		//<! converge using eigenstates
		none		//<! no convergence
	};
	
namespace lanczos{

	///@brief settings for lanczos procedure
	struct lanczosParams {

		long _seed = std::random_device{}();     	// seed for random generator
		int lanczos_steps = 200;					// number of lanczos iterations
		int random_steps  = 1;						// number of random vectors in FTLM
		bool memory_over_performance = false;		// building hamiltonian as sparse (false) or diagonalizing on-the-fly (true)
		bool use_reorthogonalization = true;		// parameter to define whether use full reorthogonalization

		lanczosParams(
			int M, int R, int seed = std::random_device{}(), 
			bool use_reortho = false, 
			bool mem_over_perf = false
		)
			:
			lanczos_steps(M),
			random_steps(R),
			_seed(seed),
			use_reorthogonalization(use_reortho),
			memory_over_performance(mem_over_perf)
		{};
		lanczosParams()											 = default;
		lanczosParams(const lanczosParams& input)				 = default;
		lanczosParams(lanczosParams&& input)			noexcept = default;
		lanczosParams& operator=(lanczosParams&& input) noexcept = default;
	};
}
#endif
