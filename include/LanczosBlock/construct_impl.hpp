#pragma once

namespace lanczos {

	/// @brief Initialize the BlockLanczos instance with set class members
	/// @tparam _ty type of input Hamiltonian (enforces type on onput state)
	template <typename _ty>
	inline void BlockLanczos<_ty>::initialize() {
		this->N = this->H.n_rows;
		this->use_krylov = 
				this->use_reorthogonalization 
			&& !this->memory_over_performance;
		this->generator = disorder<_ty>(this->_seed);

		if(this->initial_bundle.empty()){
			this->initial_bundle = arma::Mat<_ty>(this->N, this->bundle_size);
			for(int s = 0; s < this->bundle_size; s++)
				this->initial_bundle.col(s) = arma::normalise(this->generator.uniform(this->N, _ty(1.0)));
		} else {
			if(this->initial_bundle.n_cols != this->bundle_size){
				this->bundle_size = this->initial_bundle.n_cols;
				std::cout << "Input matrix different than set bundle_size. Changed to new value.";
			}
			for(int s = 0; s < this->bundle_size; s++)
				this->initial_bundle.col(s) = arma::normalise(this->initial_bundle.col(s));
		}

		//<! Orthogonalize input matrix
		#ifdef EXTRA_DEBUG
			std::cout << "V1 Before orthogonalization:\n" << this->initial_bundle.t() * this->initial_bundle << std::endl;
		#endif
		arma::Mat<_ty> dummy1, dummy2;
		arma::qr_econ(dummy1, dummy2, this->initial_bundle);
		this->initial_bundle = dummy1;

		
		this->matrix_size = this->bundle_size * this->lanczos_steps;
		#ifdef EXTRA_DEBUG
			std::cout
				<< "Model transfered to Lanczos wrapper with:\n"
				<< this->lanczos_steps << " lanczos steps\n"
				<< this->bundle_size << " initial random vectors\n"
				<< this->random_steps << " realizations in FTLM" << std::endl;
		#endif
		//try {
		//	this->H = arma::SpMat<_type>(this->N, this->N);
		//}
		//catch (const std::bad_alloc& e) {
		//	std::cout << "Memory exceeded" << e.what() << "\n";
		//	assert(false);
		//}
		//if (this->memory_over_performance)
		//	std::cout << "Hamiltonian matrix not-generated.\n Diagonalization on-the-fly!" << std::endl;
		//else {
		//	this->model->hamiltonian();
		//	std::cout << "Hamiltonian matrix generated as sparse.\n Model successfully built!; elapsed time: "
		//		<< tim_s(this->model->start) << "s" << std::endl;
		//}
	}

};