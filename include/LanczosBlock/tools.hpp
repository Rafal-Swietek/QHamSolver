#pragma once

namespace lanczos {

	/// @brief orthogonalizes input vector to the krylov space spanned by the first j bundles
	/// @tparam _ty type of input Hamiltonian (enforces type on Krylov basis) 
	/// @tparam converge_type enum type for convergence criterion (energies or states)
	/// @param vec_to_ortho Input state (in Hilbert basis) to orthogonalize in Krylov basis
	/// @param j dimension of Krylov basis
	template <typename _ty, converge converge_type>
	inline
	void BlockLanczos<_ty, converge_type>::orthogonalize(
			arma::Col<_ty>& vec_to_ortho,			//<! vector to orthogonalize
			int j									//<! current dimension of Krylov space
		) {
		arma::Col<_ty> temporary(this->N, arma::fill::zeros);
		for (int k = 0; k < j * this->bundle_size; k++)
			temporary += arma::cdot(this->krylov_space.col(k), vec_to_ortho) * this->krylov_space.col(k);

		vec_to_ortho = vec_to_ortho - temporary;
	};

	/// @brief orthogonalizes input matrix (set of vectors) to the krylov space spanned by the first j bundles
	/// @tparam _ty type of input Hamiltonian (enforces type on Krylov basis) 
	/// @tparam converge_type enum type for convergence criterion (energies or states)
	/// @param mat_to_ortho Input matrix (bundle of states in Hilbert basis) to orthogonalize in Krylov basis
	/// @param j dimension of Krylov basis (lanczos step)
	template <typename _ty, converge converge_type>
	inline
	void BlockLanczos<_ty, converge_type>::orthogonalize(
			arma::Mat<_ty>& mat_to_ortho,			//<! vector to orthogonalize
			int k									//<! current dimension of Krylov space
		) {
		// auto Krylov = this->krylov_space.cols(0, (k+1) * this->bundle_size - 1);
		mat_to_ortho -= this->krylov_space * (this->krylov_space.t() * mat_to_ortho);
	};
	
	/// @brief testing convergence of lanczos procedure
	/// @tparam _ty type of input Hamiltonian (enforces type on Krylov basis) 
	/// @tparam converge_type enum type for convergence criterion (energies or states)
	/// @param dir saving directory
	/// @param name name of file
	/// @param num_state_out number of states taken for convergence
	template <typename _ty, converge converge_type>
	inline
	void BlockLanczos<_ty, converge_type>::convergence(std::string dir, std::string name) 
	{
		std::string folder = dir + "BLOCK_LANCZOS" + kPSep + "Convergence" + kPSep;
		if (fs::create_directories(folder) || fs::is_directory(folder)) // creating the directory for saving the files with results
			std::cout << "Created LANCZOS/Convergence directory" << std::endl << std::endl;
		
		name += ",lM=" + std::to_string(this->lanczos_steps)\
			  + ",s=" + std::to_string(this->bundle_size)\
			  + ",ortho=" + std::to_string(this->use_krylov);
		
		const int M = this->lanczos_steps;
		arma::Mat<double> energies(this->lanczos_steps, this->matrix_size, arma::fill::value(1.0 / 0.0));
		arma::Mat<double> state_conv(this->lanczos_steps, this->matrix_size, arma::fill::value(1.0 / 0.0));
		for (int j = 1; j <= M; j++) {
			this->lanczos_steps = j;
			this->matrix_size = j * this->bundle_size;
			this->diagonalization();

			for (int k = 0; k < j * this->bundle_size; k++) {
				arma::Col<_ty> eigenState = conv_to_hilbert_space(k);
				double error = std::abs(arma::norm(this->eigenvalues(k) * eigenState - Hamiltonian(eigenState) ));
				state_conv(j-1, k) = error;
				energies(j-1, k) = this->eigenvalues(k);
			}
		}
		this->lanczos_steps = M;
		this->matrix_size = this->lanczos_steps * this->bundle_size;

		energies.save(arma::hdf5_name(folder + name + ".hdf5", "energies"));
		state_conv.save(arma::hdf5_name(folder + name + ".hdf5", "state error", arma::hdf5_opts::append));
	}


}