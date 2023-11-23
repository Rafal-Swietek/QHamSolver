#pragma once

namespace lanczos {

	/// @brief Initialize the BlockLanczos instance with set class members
	/// @tparam _ty type of input Hamiltonian (enforces type on Krylov basis) 
	/// @tparam converge_type enum type for convergence criterion (energies or states)
	template <typename _ty, converge converge_type>
	inline void BlockLanczos<_ty, converge_type>::initialize() {
		CONSTRUCTOR_CALL;
		
		this->generator = disorder<_ty>(this->_seed);

		//<! CHECK ESSENTIAL CLASS ELEMENTS
		if(this->maxiter < 0){
			this->use_full_convergence = false;
		} else if(this->maxiter < this->lanczos_steps){
			this->maxiter += this->lanczos_steps * this->bundle_size;
		} else if(this->maxiter > this->N){
			this->maxiter = this->N;
		}

		if(this->lanczos_steps * this->bundle_size > this->N){
			if(this->use_full_convergence){
				this->bundle_size = 1;
				this->lanczos_steps = this->N;
				this->maxiter = this->N;
			} else {
				this->bundle_size = getClosestFactor(this->bundle_size, this->N);
				this->lanczos_steps = this->N / this->bundle_size;
				this->maxiter = -1;
			}
		}

		if(this->initial_bundle.empty() || this->initial_bundle.n_cols != this->bundle_size){
			this->initial_bundle = arma::Mat<_ty>(this->N, this->bundle_size);
			for(int s = 0; s < this->bundle_size; s++){
				// this->initial_bundle(s, s) = 1.0;
				this->initial_bundle.col(s) = arma::normalise(this->generator.uniform(this->N, _ty(1.0)));
			}
		} else { 
			for(int s = 0; s < this->bundle_size; s++)
				this->initial_bundle.col(s) = arma::normalise(this->initial_bundle.col(s));
		}
		if(this->tolerance < 0)
			this->tolerance = std::abs(tolerance);

		//<! Orthogonalize input matrix
		#ifdef EXTRA_DEBUG
			std::cout << "V1 Before orthogonalization:\n" << this->initial_bundle.t() * this->initial_bundle << std::endl;
		#endif
		arma::Mat<_ty> dummy1, dummy2;
		arma::qr_econ(dummy1, dummy2, this->initial_bundle);
		this->initial_bundle = dummy1;
		#ifdef EXTRA_DEBUG
			std::cout << "V1 After orthogonalization:\n" << this->initial_bundle.t() * this->initial_bundle << std::endl;
		#endif

		// if(this->use_full_convergence)	this->matrix_size = this->bundle_size * this->maxiter;
		// else							this->matrix_size = this->bundle_size * this->lanczos_steps;

		this->matrix_size = this->bundle_size * this->lanczos_steps;
		
		#ifdef EXTRA_DEBUG
			std::cout
				<< "Model transfered to Lanczos wrapper with:\n"
				<< this->lanczos_steps << " lanczos steps\n"
				<< this->bundle_size << " initial random vectors\n"
				<< this->maxiter << " maximal iterations\n" 
				<< this->matrix_size << " matrix size" << std::endl;
		#endif
	}

};