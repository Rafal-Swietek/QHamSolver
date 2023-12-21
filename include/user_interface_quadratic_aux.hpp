#pragma once

/// @brief Calculate entanglement entropy in all eigenstates and all subsystem sizes using schmidt decomposition
/// @tparam Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_quadratic<Hamiltonian>::eigenstate_entanglement()
{
    clk::time_point start = std::chrono::system_clock::now();
	
	std::string dir = this->saving_dir + "Entropy" + kPSep + "Eigenstate" + kPSep;
	#ifdef FREE_FERMIONS
		if(this->op == 0) 		dir += "E=0,Q=0" + kPSep;
		else if(this->op == 2)	dir += "AllStates" + kPSep;
		else 					dir += "RandomChoice" + kPSep;
	#else
		if(this->op == 2)	dir += "AllStates" + kPSep;
		else 				dir += "RandomChoice" + kPSep;
	#endif
	
	createDirs(dir);
	
	std::string info = this->set_info();
	std::string filename = info;// + "_subsize=" + std::to_string(VA);


	// arma::Col<int> subsystem_sizes = arma::conv_to<arma::Col<int>>::from(arma::linspace(0, this->V / 2, this->V / 2 + 1));
	arma::Col<int> small_subsystem = arma::linspace<arma::Col<int>>(0, min(this->V / 2 - 1, 50), min(this->V / 2 - 1, 50) + 1);
	arma::Col<int> subsystem_sizes = arma::join_cols(small_subsystem, arma::Col<int>({this->V / 2}));
	std::cout << subsystem_sizes(0) << "...\t" << subsystem_sizes(subsystem_sizes.size() - 1) << std::endl;

	arma::vec entropies(subsystem_sizes.size(), arma::fill::zeros);
	arma::vec single_site_entropy(subsystem_sizes.size(), arma::fill::zeros);

	int counter = 0;

	double filling = 0.5;
	const long N = int(filling * this->V);
	disorder<double> random_generator(this->seed);
	
	// int time_end = (int)std::ceil(std::log10(5*this->V));
	arma::vec times = arma::logspace(log10(1.0 / (this->V))-1, 1, 5000);
	arma::vec sff(times.size(), arma::fill::zeros);
	double Z = 0;

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
		
        //<! Make general for complex matrices
		arma::vec S(subsystem_sizes.size(), arma::fill::zeros);
		arma::vec S_site(subsystem_sizes.size(), arma::fill::zeros);
		arma::vec sff_r(times.size(), arma::fill::zeros);
		double Z_r = 0.0;

		u64 num_states = this->num_of_points;//ULLPOW(14);
		std::vector<boost::dynamic_bitset<>> mb_states;
		#ifdef FREE_FERMIONS
			if(this->op == 1)		mb_states = QHS::single_particle::mb_config(num_states, this->V, random_generator, N);
			else if(this->op == 2) 	mb_states = QHS::single_particle::mb_config_all(this->V, N);
			else					mb_states = QHS::single_particle::mb_config_free_fermion(this->V, N);

			for(int k = 0; k < this->V; k++){
				single_particle_energy(k) = 2.0 * std::cos(two_pi * double(k) / double(this->V));
				for(int ell = 0; ell < this->V; ell++)
					orbitals(ell, k) = std::exp(-1.0i * two_pi * double(k) / double(this->V) * double(ell)) / std::sqrt(this->V);
			}
		#else
			orbitals.set_real(this->ptr_to_model->get_eigenvectors());
			if(this->op == 2) 	mb_states = QHS::single_particle::mb_config_all(this->V, N);
			else			 	mb_states = QHS::single_particle::mb_config(num_states, this->V, random_generator, N);
		#endif
		// std::cout << orbitals << std::endl;
		// for(auto& state : mb_states){
		// 	double E = 0;
		// 	int Q = 0;
		// 	int N = 0;
		// 	for(int q = 0; q < this->V; q++){
		// 		double n_q = int(state[q]);
		// 		E +=  n_q * (-2*std::cos(two_pi * q / double(this->V)));
		// 		if( n_q ){
		// 			Q += q;
		// 			N++;
		// 		}
		// 	}
		// 	printSeparated(std::cout, "\t", 20, true, state, N, (Q % this->V), E);
		// }
		num_states = mb_states.size();
		
		arma::vec E(num_states, arma::fill::zeros);
		arma::vec gap_ratio(num_states, arma::fill::zeros);

		std::cout << " - - - - - - finished many-body configurations in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl;
		std::cout << "Number of states = \t\t" << num_states << std::endl << std::endl; 
		// outer_threads = this->thread_number;
		// omp_set_num_threads(1);
		// std::cout << outer_threads << "\t\t" << omp_get_num_threads() << std::endl;
		
	// #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
		for(int VA_idx = 0; VA_idx < subsystem_sizes.size(); VA_idx++)
		{
			auto VA = subsystem_sizes(VA_idx);
			auto start_VA = std::chrono::system_clock::now();
			
			start_VA = std::chrono::system_clock::now();

			double entropy_single_site = 0;
			double entropy = 0;

		// #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
			for(u64 n = 0; n < num_states; n++){
				auto state_n = mb_states[n];
				E(n) = 0;
				for(long i = 0; i < state_n.size(); i++){
					if(state_n[i])
						E(n) += single_particle_energy(i);
				}
				//<! Generate ope-body density matrix rho -> then do correlator J
				arma::cx_mat J_m(VA, VA, arma::fill::zeros);
				cpx lambda = 0.0;
				QHS::single_particle::correlators::one_body(orbitals, state_n, VA, J_m, lambda, 1.0);
				J_m = 2.0 * J_m - arma::eye(VA, VA);

				arma::vec lambdas = arma::eig_sym(J_m);
				double S_temp = QHS::single_particle::entanglement::vonNeumann(lambdas);
				
				//<! Entanglement Hamiltonian eigenvalues
				if(VA == this->V / 2){
					arma::vec E_ent = (lambdas + 1.0) / 2.0;
					E_ent = arma::log( (1 - E_ent) / E_ent);
					arma::uvec X = arma::find_nan(E_ent);
					if( X.size() == 0){
						u64 E_av_idx = spectrals::get_mean_energy_index(E_ent);
						const u64 num = E_ent.size() < 1000? 0.25 * E_ent.size() : 0.5 * E_ent.size();
						double r_tmp = 0.0;
						double wH = 0.0;
						int countttt = 0;
						for(int i = (E_av_idx - num / 2); i < (E_av_idx + num / 2); i++){
							const double gap1 = E_ent(i) - E_ent(i - 1);
							wH += gap1;

							const double gap2 = E_ent(i + 1) - E_ent(i);
							const double min = std::min(gap1, gap2);
							const double max = std::max(gap1, gap2);
							if (abs(gap1) <= 1e-15 || abs(gap2) <= 1e-15){ 
								std::cout << "Index: " << i << std::endl;
								_assert_(false, "Found degeneracy, while doing r-statistics!\n");
							}
							r_tmp += min / max;
							countttt++;
						}
						gap_ratio(n) = r_tmp / double(countttt);		// Unbounded spectrum -> problems in gap ratio?
						// double wH = statistics::typical_level_spacing(E_ent) / two_pi;
						wH = wH / double(countttt) / two_pi;
						E_ent /= wH;
						auto [sff_tmp, Z_tmp] = statistics::spectral_form_factor(E_ent, times, 0.0, -1.0);
						X = arma::find_nan(sff_tmp);
						if( X.size() == 0){
							sff_r += sff_tmp;
							Z_r += Z_tmp;
						}
					}
				}
				// #pragma omp critical
				{
					entropy 			+= S_temp;
					entropy_single_site += QHS::single_particle::entanglement::vonNeumann_helper(2.0 * std::real(lambda) - 1.0);

				}
				// if( std::abs(entropyyy - entropy_test) > 1e-14)
				// printSeparated(std::cout, "\t", 20, true, VA, mb_states[n], S_temp, entropy_test, entropy_test - S_temp, entropy_test / S_temp);
			}
			S(VA_idx) 		= entropy / (double)num_states;					// entanglement of subsystem VA
			S_site(VA_idx) 	= entropy_single_site / double(num_states);		// single site entanglement at site VA
			if(VA == this->V / 2){
				sff_r /= double(num_states);
				Z_r /= double(num_states);
			}
    		std::cout << " - - - - - - finished entropy size VA: " << VA << " in time:" << tim_s(start_VA) << " s - - - - - - " << std::endl; // simuVAtion end
		}

		entropies += S;
		single_site_entropy += S_site;
		sff += sff_r;
		Z += Z_r;
		// for(int VA_idx = 0; VA_idx < subsystem_sizes.size(); VA_idx++)
		// 	sff_r.col(VA_idx) /= Z_r(VA_idx);

		// if(this->realisations > 1)
		{
			std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
			createDirs(dir_realis);
			S.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "entropy"));
			S_site.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "single_site_entropy", arma::hdf5_opts::append));
			arma::vec({Z_r}).save(arma::hdf5_name(dir_realis + filename + ".hdf5", "Z", arma::hdf5_opts::append));
			sff_r.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "sff", arma::hdf5_opts::append));
			gap_ratio.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "gap ratio", arma::hdf5_opts::append));
			subsystem_sizes.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "VA", arma::hdf5_opts::append));
			E.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "energy", arma::hdf5_opts::append));
		}
		
		counter++;
    	// omp_set_num_threads(this->thread_number);

		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start) << " s - - - - - - " << std::endl; // simuVAtion end
	}
    
	entropies /= double(counter);
	single_site_entropy /= double(counter);
	sff /= double(counter);
	Z /= double(counter);
	sff /= Z;
	
	#ifdef MY_MAC
		filename += "_jobid=" + std::to_string(this->jobid);
		entropies.save(arma::hdf5_name(dir + filename + ".hdf5", "entropy"));
		single_site_entropy.save(arma::hdf5_name(dir + filename + ".hdf5", "single_site_entropy", arma::hdf5_opts::append));
		sff.save(arma::hdf5_name(dir + filename + ".hdf5", "sff", arma::hdf5_opts::append));
		subsystem_sizes.save(arma::hdf5_name(dir + filename + ".hdf5", "VA", arma::hdf5_opts::append));
	#endif
    std::cout << " - - - - - - FINISHED ENTROPY CALCUVATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simuVAtion end
}

/// @brief Calculate entanglement entropy for randomly mixed many-body states
/// @tparam Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_quadratic<Hamiltonian>::eigenstate_entanglement_degenerate()
{
    clk::time_point start = std::chrono::system_clock::now();
	
	std::string dir = this->saving_dir + "Entropy" + kPSep + "Degeneracy" + kPSep;
	#ifdef FREE_FERMIONS
		if(this->op == 0) 		dir += "E=0,Q=0" + kPSep;
		else if(this->op == 2)	dir += "AllStates" + kPSep;
		else 					dir += "RandomChoice" + kPSep;
	#else
		if(this->op == 2)	dir += "AllStates" + kPSep;
		else 				dir += "RandomChoice" + kPSep;
	#endif
	
	createDirs(dir);
	
	std::string info = this->set_info();
	std::string filename = info;// + "_subsize=" + std::to_string(VA);

	const int Gamma_max = this->num_of_points;
	u64 num_states = 1e4;//500 * Gamma_max;//ULLPOW(14);

	// arma::Col<int> subsystem_sizes = arma::conv_to<arma::Col<int>>::from(arma::linspace(0, this->V / 2, this->V / 2 + 1));
	arma::Col<int> subsystem_sizes = arma::Col<int>({this->V / 2});
	// arma::Col<int> subsystem_sizes = arma::Col<int>({this->V / 6, this->V / 4, this->V / 2, this->V / 2});

	std::cout << subsystem_sizes(0) << "...\t" << subsystem_sizes(subsystem_sizes.size() - 1) << std::endl;

	arma::mat entropies(Gamma_max, subsystem_sizes.size(), arma::fill::zeros);
	arma::mat single_site_entropy(Gamma_max, subsystem_sizes.size(), arma::fill::zeros);

	int counter = 0;


	double filling = 0.5;
	const long N = int(filling * this->V);
    auto _hilbert_space = QHS::U1_hilbert_space<QHS::U1::charge, true>(this->V, N);
	size_t dim = _hilbert_space.get_hilbert_space_size();

	disorder<double> random_generator(this->seed);
	CUE random_matrix(this->seed);
	//GUE

	printSeparated(std::cout, "\t", 20, true, "VA", "ManyBody state", "S_opdm", "S_schmidt", "S_opdm - S_schmidt");
#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
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
		start = std::chrono::system_clock::now();

		arma::vec single_particle_energy = this->ptr_to_model->get_eigenvalues();
		
		arma::cx_mat orbitals = arma::cx_mat(this->V, this->V, arma::fill::zeros);
        //<! Make general for complex matrices
        
		arma::mat S(Gamma_max, subsystem_sizes.size(), arma::fill::zeros);
		arma::mat S_site(Gamma_max, subsystem_sizes.size(), arma::fill::zeros);

		std::vector<boost::dynamic_bitset<>> mb_states;
		#ifdef FREE_FERMIONS
			if(this->op == 1)		mb_states = QHS::single_particle::mb_config(num_states, this->V, random_generator, N);
			else if(this->op == 2) 	mb_states = QHS::single_particle::mb_config_all(this->V, N);
			else					mb_states = QHS::single_particle::mb_config_free_fermion(this->V, N);

			for(int k = 0; k < this->V; k++){
				single_particle_energy(k) = 2.0 * std::cos(two_pi * double(k) / double(this->V));
				for(int ell = 0; ell < this->V; ell++)
					orbitals(ell, k) = std::exp(-1.0i * two_pi * double(k) / double(this->V) * double(ell)) / std::sqrt(this->V);
			}
		#else
			orbitals.set_real(this->ptr_to_model->get_eigenvectors());
			if(this->op == 2) 	mb_states = QHS::single_particle::mb_config_all(this->V, N);
			else			 	mb_states = QHS::single_particle::mb_config(num_states, this->V, random_generator, N);
		#endif
		// std::cout << orbitals << std::endl;

		// printSeparated(std::cout, "\t", 20, true, "many-body state", "N", "Q", "E");
		// for(auto& state : mb_states){
		// 	double E = 0;
		// 	int Q = 0;
		// 	int N = 0;
		// 	for(int q = 0; q < this->V; q++){
		// 		double n_q = int(state[q]);
		// 		E +=  n_q * (-2*std::cos(two_pi * q / double(this->V)));
		// 		if( n_q ){
		// 			Q += q;
		// 			N++;
		// 		}
		// 	}
		// 	printSeparated(std::cout, "\t", 20, true, state, N, (Q % this->V), E);
		// }
		
		num_states = mb_states.size();
		
		
		std::cout << " - - - - - - finished many-body configurations in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl;
		std::cout << "Number of states = \t\t" << num_states << std::endl << std::endl;
		start = std::chrono::system_clock::now();

		QHS::single_particle::slater::ManyBodyState<cpx> SlaterConverter(orbitals, _hilbert_space);

		std::cout << " - - - - - - finished setting slater converter in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl;
		// for(auto& VA : subsystem_sizes)
		for(int VA_idx = 0; VA_idx < subsystem_sizes.size(); VA_idx++)
		{
			auto VA = subsystem_sizes(VA_idx);
			auto start_VA = std::chrono::system_clock::now();
			
			start_VA = std::chrono::system_clock::now();

			for(int gamma_a = 1; gamma_a <= Gamma_max; gamma_a++)
			{
				int counter_states = 0;

				double entropy_single_site = 0;
				double entropy = 0;
				double entropy_test = 0;
			// #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
				for(u64 unused = 0; unused < 50; unused++)
				{
					arma::cx_mat U = random_matrix.generate_matrix(gamma_a);
					arma::cx_mat J_m(VA, VA, arma::fill::zeros);
					
					arma::Col<int> indices = random_generator.create_random_vec<int>(gamma_a, 0, num_states - 1);
					int id = random_generator.random_uni<int>(0, gamma_a-1);
					cpx lambda = 0.0;
					
					arma::cx_vec coeff = U.col(id);
					coeff = arma::normalise(coeff);
					arma::cx_vec fullstate(ULLPOW(this->V), arma::fill::zeros);

					for(int n = 0; n < gamma_a; n++)
					{
						auto state_n = mb_states[indices(n)];

						// Fill state with appropriate values --------------------------------------------------
						SlaterConverter.convert(fullstate, state_n, coeff(n));
						// --------------------------------------------------------------------------------------
					}
					entropy += entropy::schmidt_decomposition(fullstate, VA, this->V);
					counter_states++;
				}

				// printSeparated(std::cout, "\t", 16, true, VA, gamma_a, entropy / (double)counter_states, entropy_test / (double)counter_states, entropy / (double)counter_states - entropy_test / (double)counter_states);
				
				S(gamma_a-1, VA_idx) 		= entropy / (double)counter_states;				// entanglement of subsystem VA using Slater determiniants
				// S_site(gamma_a-1, VA_idx) 	= entropy_single_site / double(counter_states);	// single site entanglement at site VA
			}
    		std::cout << "\n - - - - - - finished entropy size VA: " << VA << " in time:" << tim_s(start_VA) << " s - - - - - - " << std::endl; // simuVAtion end
		}

		// if(this->realisations > 1)
		{
			std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
			createDirs(dir_realis);
			S.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "entropy"));
			S_site.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "single_site_entropy", arma::hdf5_opts::append));
			single_particle_energy.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "single particle energy", arma::hdf5_opts::append));
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

/// @brief Calculate entanglement entropy in all eigenstates and all subsystem sizes using schmidt decomposition
/// @tparam Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_quadratic<Hamiltonian>::diagonal_matrix_elements()
{
    clk::time_point start = std::chrono::system_clock::now();
	
	#if DIM == 1
		auto lattice = lattice::lattice1D(this->L);
	#elif DIM == 2
		auto lattice = lattice::lattice2D(this->L);
	#else
		auto lattice = lattice::lattice3D(this->L);
	#endif
	
	start = std::chrono::system_clock::now();

	arma::ivec neighbours(this->V, arma::fill::value(-1));
	arma::ivec next_neighbours(this->V, arma::fill::value(-1));
	for(int ell = 0; ell < this->V; ell++){
		auto nei = lattice.get_nearest_neighbour(ell);
		neighbours(ell) = nei;

		nei = lattice.get_next_nearest_neighbour(ell);
		next_neighbours(ell) = nei;
	}
	std::cout << " - - - - - - set lattice and neighbours in : " << tim_s(start) << " s - - - - - - " << std::endl;

	std::string dir = this->saving_dir + "DiagonalMatrixElements" + kPSep + "ManyBody" + kPSep;
	#ifdef FREE_FERMIONS
		if(this->op == 0) 		dir += "E=0,Q=0" + kPSep;
		else if(this->op == 2)	dir += "AllStates" + kPSep;
		else 					dir += "RandomChoice" + kPSep;
	#else
		if(this->op == 2)	dir += "AllStates" + kPSep;
		else 				dir += "RandomChoice" + kPSep;
	#endif
	
	createDirs(dir);
	
	std::string info = this->set_info();
	std::string filename = info;
	
	disorder<double> random_generator(this->seed);
	double filling = 0.5;
	const int N = int(this->V * filling);


	//<! START REALISATION
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
		start = std::chrono::system_clock::now();

		arma::vec single_particle_energy = this->ptr_to_model->get_eigenvalues();
		
		arma::Mat<element_type> orbitals;
		
		u64 num_states = this->num_of_points;//ULLPOW(14);
		std::vector<boost::dynamic_bitset<>> mb_states;
		#ifdef FREE_FERMIONS
			if(this->op == 1)		mb_states = QHS::single_particle::mb_config(num_states, this->V, random_generator, N);
			else if(this->op == 2) 	mb_states = QHS::single_particle::mb_config_all(this->V, N);
			else					mb_states = QHS::single_particle::mb_config_free_fermion(this->V, N);

			for(int k = 0; k < this->V; k++){
				single_particle_energy(k) = 2.0 * std::cos(two_pi * double(k) / double(this->V));
				for(int ell = 0; ell < this->V; ell++)
					orbitals(ell, k) = std::cos(two_pi * double(k) / double(this->V) * double(ell)) / std::sqrt(this->V);
					// orbitals(ell, k) = std::exp(-1.0i * two_pi * double(k) / double(this->V) * double(ell)) / std::sqrt(this->V);
			}
		#else
			orbitals = this->ptr_to_model->get_eigenvectors();
			if(this->op == 2) 	mb_states = QHS::single_particle::mb_config_all(this->V, N);
			else			 	mb_states = QHS::single_particle::mb_config(num_states, this->V, random_generator, N);
		#endif
		
		std::cout << " - - - - - - finished many-body configurations in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simuVAtion end

		arma::vec energy(mb_states.size(), arma::fill::zeros);

		//<! 1-BODY OBSERVABLES
		arma::Col<element_type> m0(mb_states.size(), arma::fill::zeros);

		arma::Col<element_type> T_nn(mb_states.size(), arma::fill::zeros);
		arma::Col<element_type> T_nnn(mb_states.size(), arma::fill::zeros);
		arma::Col<element_type> T_nn_loc(mb_states.size(), arma::fill::zeros);
		arma::Col<element_type> T_nnn_loc(mb_states.size(), arma::fill::zeros);

		//<! 2-BODY OBSERVABLES
		arma::Col<element_type> U_nn(mb_states.size(), arma::fill::zeros);
		arma::Col<element_type> U_nnn(mb_states.size(), arma::fill::zeros);
		arma::Col<element_type> U_nn_loc(mb_states.size(), arma::fill::zeros);
		arma::Col<element_type> U_nnn_loc(mb_states.size(), arma::fill::zeros);

		arma::Mat<unsigned int> states(mb_states.size(), this->V, arma::fill::zeros);
	#pragma omp parallel for
		for(int idx = 0; idx < mb_states.size(); idx++)
		{
			auto state = mb_states[idx];	// get many-body-state
			auto set_q 	 = QHS::single_particle::slater::ManyBodyState<element_type>::set_indices(state, N);
			
			energy(idx) = 0.;
			for(u64 q : set_q){
				states(idx, q) = 1;
				energy(idx) += single_particle_energy(q);
			}

			// printSeparated(std::cout, "\t", 20, true, idx, state, ~state);
			//<! ----
			for(u64 ell = 0; ell < this->V; ell++)
			{
				u64 nei 		= neighbours(ell);
				u64 next_nei 	= next_neighbours(ell);
				element_type Al = 0., Al_1 = 0., Al_2 = 0., 
							 Bl_1 = 0., Bl_2 = 0.;
							//  Cl_1 = 0., Cl_2 = 0.;
				for(u64 q : set_q){
					Al +=   my_conjungate( orbitals(ell,      q) ) * orbitals(ell, 		q);
					Al_1 += my_conjungate( orbitals(nei,      q) ) * orbitals(nei, 		q);
					Al_2 += my_conjungate( orbitals(next_nei, q) ) * orbitals(next_nei, q);

					Bl_1 += my_conjungate( orbitals(ell, q) ) * orbitals(nei, 	   q);
					Bl_2 += my_conjungate( orbitals(ell, q) ) * orbitals(next_nei, q);

					// Cl_1 += my_conjungate( orbitals(ell, q) * orbitals(nei, 	 q) ) * orbitals(ell, q) * orbitals(nei, 	  q);
					// Cl_2 += my_conjungate( orbitals(ell, q) * orbitals(next_nei, q) ) * orbitals(ell, q) * orbitals(next_nei, q);
					
					for(u64 ell2 = 0; ell2 < ell; ell2++)
						m0(idx) += my_conjungate( orbitals(ell, q) ) * orbitals(ell2, q) + my_conjungate( orbitals(ell2, q) ) * orbitals(ell, q);
				}
				T_nn(idx)  += Bl_1 + my_conjungate(Bl_1);
				T_nnn(idx) += Bl_2 + my_conjungate(Bl_2);
				U_nn(idx)  += Al * Al_1 - Bl_1 * my_conjungate(Bl_1);// + Cl_1;
				U_nnn(idx) += Al * Al_2 - Bl_2 * my_conjungate(Bl_2);// + Cl_2;

				if(ell == 0){
					T_nn_loc(idx)  += Bl_1 + my_conjungate(Bl_1);
					T_nnn_loc(idx) += Bl_2 + my_conjungate(Bl_2);
					U_nn_loc(idx)  += Al * Al_1 - Bl_1 * my_conjungate(Bl_1);// + Cl_1;
					U_nnn_loc(idx) += Al * Al_2 - Bl_2 * my_conjungate(Bl_2);// + Cl_2;
				}
			}
			//<! ----
				
		}
		m0 /= double(this->V);
		T_nn /= std::sqrt(this->V);
		T_nnn /= std::sqrt(this->V);
		U_nn /= std::sqrt(this->V);
		U_nnn /= std::sqrt(this->V);


		std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
		createDirs(dir_realis);
		states.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "many-body states"));
		single_particle_energy.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "single particle energy", arma::hdf5_opts::append));
		energy.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "many-body energy", arma::hdf5_opts::append));
		m0.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "m0", arma::hdf5_opts::append));

		T_nn.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "T_nn", arma::hdf5_opts::append));
		T_nnn.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "T_nnn", arma::hdf5_opts::append));
		T_nn_loc.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "T_nn_loc", arma::hdf5_opts::append));
		T_nnn_loc.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "T_nnn_loc", arma::hdf5_opts::append));

		U_nn.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "U_nn", arma::hdf5_opts::append));
		U_nnn.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "U_nnn", arma::hdf5_opts::append));
		U_nn_loc.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "U_nn_loc", arma::hdf5_opts::append));
		U_nnn_loc.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "U_nnn_loc", arma::hdf5_opts::append));

		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start) << " s - - - - - - " << std::endl; // simuVAtion end
	}
    
    std::cout << " - - - - - - FINISHED DIAGONAL MATRIX ELEMENTS CALCUVATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simuVAtion end
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




// ---------------------------------------------------------------------------------------------------------------- OPDM for gaussian mixture
						// <n|f+_q f_q|n>
						// double pre = std::abs(coeff(n)) * std::abs(coeff(n));
						// QHS::single_particle::correlators::one_body(orbitals, state_n, VA, J_m, lambda, pre);
						
						// <m|f+_q1 f_q2|n>
					// 	for(int m = n + 1; m < gamma_a; m++)
					// 	{
					// 		auto state_m = mb_states[indices(m)];
						
					// 		// arma::cx_mat J_m_tmp(VA, VA, arma::fill::zeros);
					// 		auto x = state_n ^ state_m;
					// 		if(x.count() == 2){		// states differ only at two sites, q1 and q2
					// 			// std::cout << state_n << "\t\t" << state_m << std::endl;
					// 			std::vector<int> qs;
					// 			auto prefactor = std::conj(coeff(m)) * coeff(n);
					// 			for(int q = 0; q < this->V; q++)
					// 				if(x[q]) qs.push_back(q);
									
					// 			if(state_n[qs[0]] ^ state_n[qs[1]])	// state n and m differ at q1 and q2 to enable hopping, otherwise skip
					// 			{
					// 				for(auto& qss : v_2d<int>( { qs, v_1d<int>({qs[1], qs[0]}) } ) ){
					// 					int q1 = qss[0];
					// 					int q2 = qss[1];

					// 					cpx pre = prefactor;
					// 					if(state_n[q1])		// for one of the 2 cases do conjungation
					// 						pre = std::conj(prefactor);
										
					// 					lambda += pre * std::abs(orbitals(q2, VA) * std::conj(orbitals(q1, VA)));
										
					// 					if(VA > 0){
					// 						auto orbital1 = orbitals.col(q1).rows(0, VA - 1);
					// 						auto orbital2 = orbitals.col(q2).rows(0, VA - 1);
					// 						J_m += pre * orbital2 * orbital1.t();
					// 					}
					// 				}
					// 			}
					// 		}
					// 	}

					// fullstate = arma::normalise(fullstate);
					// J_m = 2.0 * J_m - arma::eye(VA, VA);
					// auto lambdas = arma::eig_sym(J_m);
						// entropy 			+= QHS::single_particle::entanglement::vonNeumann(lambdas);
						// entropy_single_site += QHS::single_particle::entanglement::vonNeumann_helper(2.0 * std::real(lambda) - 1.0);
					
