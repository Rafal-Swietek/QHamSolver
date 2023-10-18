#pragma once

namespace lanczos {

	template <typename _ty>
	inline void Lanczos<_ty>::initialize() {
		this->N = this->H.n_rows;
		this->use_krylov = 
				this->params.use_reorthogonalization 
			&& !this->params.memory_over_performance;
		this->generator = disorder<_ty>(this->params._seed);

		if(this->initial_random_vec.empty())
			this->initial_random_vec = arma::normalise(this->generator.uniform(N, 1.0));
		
		// std::cout << this->initial_random_vec.t() << std::endl;
		std::cout
			<< "Model transfered to Lanczos wrapper with:\n"
			<< this->params.lanczos_steps << " lanczos steps\n"
			<< this->params.random_steps << " random vectors" << std::endl;
		//try {
		//	this->H = arma::SpMat<_type>(this->N, this->N);
		//}
		//catch (const std::bad_alloc& e) {
		//	std::cout << "Memory exceeded" << e.what() << "\n";
		//	assert(false);
		//}
		//if (this->params.memory_over_performance)
		//	std::cout << "Hamiltonian matrix not-generated.\n Diagonalization on-the-fly!" << std::endl;
		//else {
		//	this->model->hamiltonian();
		//	std::cout << "Hamiltonian matrix generated as sparse.\n Model successfully built!; elapsed time: "
		//		<< tim_s(this->model->start) << "s" << std::endl;
		//}
	}

};