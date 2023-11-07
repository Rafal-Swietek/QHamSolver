#pragma once

namespace lanczos {

	template <typename _ty>
	inline void Lanczos<_ty>::initialize() {
		this->N = this->H.n_rows;
		this->use_krylov = 
				this->use_reorthogonalization 
			&& !this->memory_over_performance;
		this->generator = disorder<_ty>(this->_seed);

		if(this->initial_random_vec.empty())
			this->initial_random_vec = this->generator.uniform(this->N, _ty(1.0));
		this->initial_random_vec = arma::normalise(this->initial_random_vec);
		
		// std::cout << this->initial_random_vec.t() << std::endl;
		#ifdef EXTRA_DEBUG
			std::cout
				<< "Model transfered to Lanczos wrapper with:\n"
				<< this->lanczos_steps << " lanczos steps\n"
				<< this->random_steps << " random vectors" << std::endl;
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