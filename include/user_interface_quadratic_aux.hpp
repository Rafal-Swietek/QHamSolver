#pragma once

/// @brief Calculate entanglement entropy in all eigenstates and all subsystem sizes using schmidt decomposition
/// @tparam Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_quadratic<Hamiltonian>::eigenstate_entanglement()
{
    clk::time_point start = std::chrono::system_clock::now();
	
	std::string dir = this->saving_dir + "Entropy" + kPSep + "Eigenstate" + kPSep;
	createDirs(dir);
	
	std::string info = this->set_info();
	std::string filename = info;// + "_subsize=" + std::to_string(VA);

	u64 num_states = this->num_of_points;//ULLPOW(14);

	arma::Col<int> subsystem_sizes = arma::conv_to<arma::Col<int>>::from(arma::linspace(0, this->V / 2 - 1, this->V / 2));
	std::cout << subsystem_sizes(0) << "...\t" << subsystem_sizes(subsystem_sizes.size() - 1) << std::endl;

	arma::vec energies(num_states, arma::fill::zeros);
	arma::vec entropies(subsystem_sizes.size(), arma::fill::zeros);
	arma::vec single_site_entropy(subsystem_sizes.size(), arma::fill::zeros);

	int counter = 0;

	
	disorder<double> random_generator(this->seed);

// #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
	for(int realis = 0; realis < this->realisations; realis++)
	{
		if(realis > 0)
			this->ptr_to_model->generate_hamiltonian();
		start = std::chrono::system_clock::now();
    #ifdef ARMA_USE_SUPERLU
        if(this->ch){
            this->ptr_to_model->hamiltonian();
            this->ptr_to_model->diag_sparse(true);
        } else
            this->ptr_to_model->diagonalization();
    
    #else
        this->ptr_to_model->diagonalization();
    #endif

		std::cout << " - - - - - - finished diagonalization in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simuVAtion end
		
		arma::vec single_particle_energy = this->ptr_to_model->get_eigenvalues();
		
		arma::cx_mat orbitals = arma::cx_mat(this->V, this->V, arma::fill::zeros);
		orbitals.set_real(this->ptr_to_model->get_eigenvectors());
        //<! Make general for complex matrices
        
		arma::vec S(subsystem_sizes.size(), arma::fill::zeros);
		arma::vec S_site(subsystem_sizes.size(), arma::fill::zeros);

        //<! add choosing flag
		// auto mb_states = single_particle::mb_config_all(this->V);
		auto mb_states = single_particle::mb_config(num_states, this->V, random_generator);
		// auto mb_states = single_particle::mb_config_free_fermion(num_states, this->V, random_generator);
		
		arma::vec E(num_states);
		// num_states = mb_states.size();
		
		
		std::cout << " - - - - - - finished many-body configurations in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl;
		std::cout << "Number of states = \t\t" << num_states << std::endl << std::endl; 
		outer_threads = this->thread_number;
		omp_set_num_threads(1);
		std::cout << outer_threads << "\t\t" << omp_get_num_threads() << std::endl;
		
		for(auto& VA : subsystem_sizes)
		{
			auto start_VA = std::chrono::system_clock::now();
			
			start_VA = std::chrono::system_clock::now();

			double entropy_single_site = 0;
			double entropy = 0;

		#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
			for(u64 n = 0; n < num_states; n++){
				E(n) = 0;
				double lambda = 0;
				arma::cx_mat J_m(VA, VA, arma::fill::zeros);
				int N = 0;
				for(int q = 0; q < this->V; q++){
					double n_q = int(mb_states[n][q]);
					double c_q = 2 * n_q - 1;
					lambda += c_q * std::abs(orbitals(q, VA) * orbitals(q, VA));
					
					E(n) += single_particle_energy(q) * n_q;
					if(VA > 0){
						arma::cx_vec orbital = orbitals.col(q).rows(0, VA - 1);
						J_m += c_q * orbital * orbital.t();
					}
					N += n_q;
				}
				auto lambdas = arma::eig_sym(J_m);
				
				#pragma omp critical
				{
					entropy 			+= single_particle::entanglement::vonNeumann(lambdas);
					entropy_single_site += single_particle::entanglement::vonNeumann_helper(lambda);
				} 
			}
			S(VA - subsystem_sizes(0)) 			= entropy / (double)num_states;					// entanglement of subsystem VA
			S_site(VA - subsystem_sizes(0)) 	= entropy_single_site / double(num_states);		// single site entanglement at site VA

    		std::cout << " - - - - - - finished entropy size VA: " << VA << " in time:" << tim_s(start_VA) << " s - - - - - - " << std::endl; // simuVAtion end
		}

		E = arma::sort(E);
		if(this->realisations > 1){
			std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
			createDirs(dir_realis);
			// E.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "energy"));
			S.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "entropy"));
			S_site.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "single_site_entropy", arma::hdf5_opts::append));
		}
		energies += E;
		entropies += S;
		single_site_entropy += S_site;
		
		counter++;
    	omp_set_num_threads(this->thread_number);

		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start) << " s - - - - - - " << std::endl; // simuVAtion end
	}
    
	energies /= double(counter);
	entropies /= double(counter);
	single_site_entropy /= double(counter);

	filename += "_jobid=" + std::to_string(this->jobid);
	energies.save(arma::hdf5_name(dir + filename + ".hdf5", "energy"));
	entropies.save(arma::hdf5_name(dir + filename + ".hdf5", "entropy", arma::hdf5_opts::append));
	single_site_entropy.save(arma::hdf5_name(dir + filename + ".hdf5", "single_site_entropy", arma::hdf5_opts::append));
    std::cout << " - - - - - - FINISHED ENTROPY CALCUVATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simuVAtion end
}


#pragma once

/// @brief Calculate entanglement entropy for randomly mixed many-body states
/// @tparam Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_quadratic<Hamiltonian>::eigenstate_entanglement_degenerate()
{
    clk::time_point start = std::chrono::system_clock::now();
	
	std::string dir = this->saving_dir + "Entropy" + kPSep + "Degeneracy" + kPSep;
	createDirs(dir);
	
	std::string info = this->set_info();
	std::string filename = info;// + "_subsize=" + std::to_string(VA);

	const int Gamma_max = this->num_of_points;
	u64 num_states = 100 * Gamma_max;//ULLPOW(14);

	// arma::Col<int> subsystem_sizes = arma::conv_to<arma::Col<int>>::from(arma::linspace(0, this->V / 2 - 1, this->V / 2));
	arma::Col<int> subsystem_sizes = arma::Col<int>({this->V / 2});
	std::cout << subsystem_sizes(0) << "...\t" << subsystem_sizes(subsystem_sizes.size() - 1) << std::endl;

	arma::mat entropies(Gamma_max, subsystem_sizes.size(), arma::fill::zeros);
	arma::mat single_site_entropy(Gamma_max, subsystem_sizes.size(), arma::fill::zeros);

	int counter = 0;

	disorder<double> random_generator(this->seed);
	CUE random_matrix(this->seed);
	//GUE
// #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
	for(int realis = 0; realis < this->realisations; realis++)
	{
		if(realis > 0)
			this->ptr_to_model->generate_hamiltonian();
		start = std::chrono::system_clock::now();
    #ifdef ARMA_USE_SUPERLU
        if(this->ch){
            this->ptr_to_model->hamiltonian();
            this->ptr_to_model->diag_sparse(true);
        } else
            this->ptr_to_model->diagonalization();
    
    #else
        this->ptr_to_model->diagonalization();
    #endif

		std::cout << " - - - - - - finished diagonalization in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simuVAtion end
		
		arma::vec single_particle_energy = this->ptr_to_model->get_eigenvalues();
		
		arma::cx_mat orbitals = arma::cx_mat(this->V, this->V, arma::fill::zeros);
		orbitals.set_real(this->ptr_to_model->get_eigenvectors());
        //<! Make general for complex matrices
        
		arma::mat S(Gamma_max, subsystem_sizes.size(), arma::fill::zeros);
		arma::mat S_site(Gamma_max, subsystem_sizes.size(), arma::fill::zeros);

        //<! add choosing flag
		// auto mb_states = single_particle::mb_config_all(this->V);
		auto mb_states = single_particle::mb_config(num_states, this->V, random_generator);
		// auto mb_states = single_particle::mb_config_free_fermion(num_states, this->V, random_generator);
		
		num_states = mb_states.size();
		
		
		std::cout << " - - - - - - finished many-body configurations in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl;
		std::cout << "Number of states = \t\t" << num_states << std::endl << std::endl; 
		outer_threads = this->thread_number;
		omp_set_num_threads(1);
		std::cout << outer_threads << "\t\t" << omp_get_num_threads() << std::endl;
		
		
		for(auto& VA : subsystem_sizes)
		{
			auto start_VA = std::chrono::system_clock::now();
			
			start_VA = std::chrono::system_clock::now();


		#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
			for(int gamma_a = 1; gamma_a <= Gamma_max; gamma_a++)
			{
				int counter_states = 0;

				double entropy_single_site = 0;
				double entropy = 0;
				arma::cx_mat U = random_matrix.generate_matrix(gamma_a);
				// realisations to draw states randomly
				for(u64 id = 0; id < std::min(50, gamma_a); id++)
				{
					arma::cx_mat J_m(VA, VA, arma::fill::zeros);
					arma::Col<int> indices = random_generator.create_random_vec<int>(gamma_a, 0, num_states - 1);
					cpx lambda = 0;
					for(int n = 0; n < gamma_a; n++)
					{
						auto state_n = mb_states[indices(n)];
						for(int m = 0; m < gamma_a; m++)
						{
							auto prefactor = std::conj(U(id, m)) * U(id, n);
							auto state_m = mb_states[indices(m)];
							if(n == m)
							{
								// <n|f+_q f_q|n>
								arma::cx_mat J_m_tmp(VA, VA);
								double lambda_tmp;
								std::tie(J_m_tmp, lambda_tmp) = single_particle::correlators::one_body(orbitals, state_n, VA);
								if(VA > 0)	J_m += prefactor * J_m_tmp;
								lambda += prefactor * lambda_tmp;
							}
							else{
								// <m|f+_q1 f_q2|n>
								arma::cx_mat J_m_tmp(VA, VA, arma::fill::zeros);
								auto x = state_n ^ state_m;
								if(x.count() == 2){
									for(int q1 = 0; q1 < this->V; q1++){
										for(int q2 = 0; q2 < this->V; q2++){
											// if states are equal up to q1 and q2 single particle states, then differ accordingly
											if((x[q1] && x[q2]) && state_n[q2] && !state_n[q1] && state_m[q1] && !state_m[q2]){
												lambda += prefactor * std::abs(orbitals(q2, VA) * std::conj(orbitals(q1, VA)));
												
												if(VA > 0){
													arma::cx_vec orbital1 = orbitals.col(q1).rows(0, VA - 1);
													arma::cx_vec orbital2 = orbitals.col(q2).rows(0, VA - 1);
													J_m_tmp += prefactor * orbital2 * orbital1.t();
												}
											}
										}
									}
									if(VA > 0){
										// std::cout << state_n << "\t\t" << state_m << "\t\t" << x << std::endl;
										// if(!J_m_tmp.is_hermitian() && !J_m_tmp.is_symmetric()) 
										// 	std::cout << J_m_tmp << std::endl;
										J_m += J_m_tmp;
									}
								}
							}
						}
					}
					J_m = 2.0 * J_m - arma::eye(VA, VA);
					auto lambdas = arma::eig_sym(J_m);
					entropy 			+= single_particle::entanglement::vonNeumann(lambdas);
					entropy_single_site += single_particle::entanglement::vonNeumann_helper(2.0 * std::real(lambda) - 1.0);
					counter_states++;
				}
				S(gamma_a-1, VA - subsystem_sizes(0)) 		= entropy / (double)counter_states;				// entanglement of subsystem VA
				S_site(gamma_a-1, VA - subsystem_sizes(0)) 	= entropy_single_site / double(counter_states);	// single site entanglement at site VA
			}
    		std::cout << " - - - - - - finished entropy size VA: " << VA << " in time:" << tim_s(start_VA) << " s - - - - - - " << std::endl; // simuVAtion end
		}

		if(this->realisations > 1){
			std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
			createDirs(dir_realis);
			S.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "entropy"));
			S_site.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "single_site_entropy", arma::hdf5_opts::append));
		}
		entropies += S;
		single_site_entropy += S_site;
		
		counter++;
    	omp_set_num_threads(this->thread_number);

		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start) << " s - - - - - - " << std::endl; // simuVAtion end
	}
    
	entropies /= double(counter);
	single_site_entropy /= double(counter);

	filename += "_jobid=" + std::to_string(this->jobid);
	entropies.save(arma::hdf5_name(dir + filename + ".hdf5", "entropy"));
	single_site_entropy.save(arma::hdf5_name(dir + filename + ".hdf5", "single_site_entropy", arma::hdf5_opts::append));
    std::cout << " - - - - - - FINISHED ENTROPY CALCUVATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simuVAtion end
}


// ---------------------------------------------------------------------------------------------------------------- IMPLEMENTATION OF UI

/// @brief Sets model parameters from values in command line
/// @tparam Hamiltonian Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_quadratic<Hamiltonian>::parse_cmd_options(int argc, std::vector<std::string> argv)
{
	//<! set all general UI parameters
    user_interface_dis<Hamiltonian>::parse_cmd_options(argc, argv);

	std::string choosen_option = "";																// current choosen option

	//---------- SIMULATION PARAMETERS
	
}


/// @brief Sets all UI parameters to default values
/// @tparam Hamiltonian Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_quadratic<Hamiltonian>::set_default(){
	
	user_interface_dis<Hamiltonian>::set_default();
	
	this->V = this->L;
}


/// @brief Prints all general UI option values
/// @tparam Hamiltonian Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_quadratic<Hamiltonian>::printAllOptions() const {
	
	user_interface_dis<Hamiltonian>::printAllOptions();

	std::cout << "---------------------------------------CHOSEN MODEL:" << std::endl;
}

