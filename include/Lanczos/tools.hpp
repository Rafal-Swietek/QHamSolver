#pragma once

namespace lanczos {

	/// @brief testing convergence of lanczos procedure
	/// @tparam _ty template for element type
	/// @param dir saving directory
	/// @param name name of file
	template <typename _ty>
	inline
	void Lanczos<_ty>::convergence(std::string dir, std::string name) 
	{
		std::string folder = dir + "LANCZOS" + kPSep + "Convergence" + kPSep;
		if (fs::create_directories(folder) || fs::is_directory(folder)) // creating the directory for saving the files with results
			std::cout << "Created LANCZOS/Convergence directory" << std::endl << std::endl;
		
		name += ",lM=" + std::to_string(this->lanczos_steps)\
			  + ",ortho=" + std::to_string(this->use_krylov);
		
		const int M = this->lanczos_steps;
		arma::Mat<double> energies(M, M, arma::fill::value(1.0 / 0.0));
		arma::Mat<double> state_conv(M, M, arma::fill::value(1.0 / 0.0));
		for (int j = 1; j <= M; j++) {
			this->lanczos_steps = j;

			this->diagonalization();
			
			for (int k = 0; k < j; k++) {
				arma::Col<_ty> eigenState = conv_to_hilbert_space(k);
				double error = arma::norm(this->eigenvalues(k) * eigenState - this->H * eigenState);
				state_conv(j-1, k) = error;
				energies(j-1, k) = this->eigenvalues(k);
			}
		}
		this->lanczos_steps = M;

		energies.save(arma::hdf5_name(folder + name + ".hdf5", "energies"));
		state_conv.save(arma::hdf5_name(folder + name + ".hdf5", "state error", arma::hdf5_opts::append));
	}


}