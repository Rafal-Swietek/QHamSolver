#pragma once

namespace lanczos {

	template <typename _ty, converge converge_type>
	inline void Lanczos<_ty, converge_type>::initialize() {
		CONSTRUCTOR_CALL;
		this->N = this->H.n_rows;
		this->use_krylov = 
				this->use_reorthogonalization 
			&& !this->memory_over_performance;
		this->generator = disorder<_ty>(this->_seed);


		//<! CHECK ESSENTIAL CLASS ELEMENTS
		if(this->initial_random_vec.empty())
			this->initial_random_vec = this->generator.uniform(this->N, _ty(1.0));
		this->initial_random_vec = arma::normalise(this->initial_random_vec);
		
		if(this->maxiter < 0)
			this->use_full_convergence = false;
		else if(this->maxiter < this->lanczos_steps)
				this->maxiter += this->lanczos_steps;
		else if(this->maxiter > this->N)
				this->maxiter = this->N;

		if(this->lanczos_steps > this->N)
			this->lanczos_steps = this->N;

		if(this->tolerance < 0)
			this->tolerance = std::abs(tolerance);

		#ifdef EXTRA_DEBUG
			// std::cout << this->initial_random_vec.t() << std::endl;
			std::cout
				<< "Model transfered to Lanczos wrapper with:\n"
				<< this->lanczos_steps << " lanczos steps\n"
				<< this->maxiter << " maximal iterations\n"
				<< this->tolerance << " tolerance" << std::endl;
		#endif
	}
};