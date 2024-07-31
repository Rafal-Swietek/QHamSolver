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
		// arma::vec gap_ratio(num_states, arma::fill::zeros);

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
						double wH = statistics::typical_level_spacing(E_ent) / two_pi;
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
			// gap_ratio.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "gap ratio", arma::hdf5_opts::append));
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

	// const int Gamma_max = this->num_of_points;
	u64 num_states = 1e5;//500 * Gamma_max;//ULLPOW(14);
	
	arma::Col<int> Gammas = arma::Col<int>({1, 2, 4, this->V / 4, this->V / 2, this->V, this->V * this->V});
	const int Gamma_max = Gammas.size();
	arma::vec qs = arma::vec({0.5, 1, 2});

	// arma::Col<int> subsystem_sizes = arma::conv_to<arma::Col<int>>::from(arma::linspace(0, this->V / 2, this->V / 2 + 1));
	arma::Col<int> subsystem_sizes = arma::Col<int>({this->V / 2});
	// arma::Col<int> subsystem_sizes = arma::Col<int>({this->V / 6, this->V / 4, this->V / 2, this->V / 2});

	std::cout << subsystem_sizes(0) << "...\t" << subsystem_sizes(subsystem_sizes.size() - 1) << std::endl;

	arma::mat entropies(Gamma_max, subsystem_sizes.size(), arma::fill::zeros);
	arma::mat single_site_entropy(Gamma_max, subsystem_sizes.size(), arma::fill::zeros);
	arma::mat participation_ratios(Gamma_max, qs.size(), arma::fill::zeros);

	int counter = 0;


	double filling = 0.5;
	const long N = int(filling * this->V);
    auto _hilbert_space = QHS::U1_hilbert_space<QHS::U1::charge, true>(this->V, N);
	size_t dim = _hilbert_space.get_hilbert_space_size();

	disorder<double> random_generator(this->seed);
	disorder<int> random_integers(this->seed);
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
		arma::mat prs(Gamma_max, qs.size(), arma::fill::zeros);
		for(int VA_idx = 0; VA_idx < subsystem_sizes.size(); VA_idx++)
		{
			auto VA = subsystem_sizes(VA_idx);
			auto start_VA = std::chrono::system_clock::now();
			
			start_VA = std::chrono::system_clock::now();
			prs.zeros();
			// for(int gamma_a = 1; gamma_a <= Gamma_max; gamma_a++)
			for(int ii = 0; ii < Gammas.size(); ii++)
			{
				int gamma_a = Gammas(ii);
				int counter_states = 0;

				double entropy_single_site = 0;
				double entropy = 0;
				double entropy_test = 0;
			// #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
				for(u64 unused = 0; unused < 1; unused++)
				{
					arma::vec _prs_(qs.size(), arma::fill::zeros);
					arma::cx_mat U = random_matrix.generate_matrix(gamma_a);
					arma::cx_mat J_m(VA, VA, arma::fill::zeros);
					
					arma::Col<int> indices = random_integers.uniform(gamma_a, 0, num_states - 1);
					int id = random_integers.uniform_dist<int>(0, gamma_a-1);
					cpx lambda = 0.0;
					
					arma::cx_vec coeff = U.col(id);
					coeff = arma::normalise(coeff);
					arma::cx_vec fullstate(ULLPOW(this->V), arma::fill::zeros);

					// auto starta = std::chrono::system_clock::now();
					for(int n = 0; n < gamma_a; n++)
					{
						auto state_n = mb_states[indices(n)];

						// Fill state with appropriate values --------------------------------------------------
						SlaterConverter.convert(fullstate, state_n, coeff(n), qs, _prs_);
						// --------------------------------------------------------------------------------------
					}
					// std::cout << "\n - - - - - - finished Many Body state in time:" << tim_s(starta) << " s - - - - - - " << std::endl; // simuVAtion end
					// starta = std::chrono::system_clock::now();
					entropy += entropy::schmidt_decomposition(fullstate, VA, this->V);
					// std::cout << "\n - - - - - - finished entropy of Many Body state in time:" << tim_s(starta) << " s - - - - - - " << std::endl; // simuVAtion end

					prs.row(ii) += _prs_.t();
					counter_states++;
				}
				participation_ratios(ii) = prs(ii) / (double)counter_states;

				// printSeparated(std::cout, "\t", 16, true, VA, gamma_a, entropy / (double)counter_states, entropy_test / (double)counter_states, entropy / (double)counter_states - entropy_test / (double)counter_states);
				
				S(ii, VA_idx) 		= entropy / (double)counter_states;				// entanglement of subsystem VA using Slater determiniants
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
			subsystem_sizes.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "subsystem_sizes", arma::hdf5_opts::append));
			qs.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "qs", arma::hdf5_opts::append));
			prs.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "participation_ratio", arma::hdf5_opts::append));
			single_particle_energy.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "single particle energy", arma::hdf5_opts::append));
		}
		entropies += S;
		single_site_entropy += S_site;
		
		counter++;
    	omp_set_num_threads(this->thread_number);

		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start) << " s - - - - - - " << std::endl; // simuVAtion end
	}
    
	// entropies /= double(counter);
	// single_site_entropy /= double(counter);

	// filename += "_jobid=" + std::to_string(this->jobid);
	// entropies.save(arma::hdf5_name(dir + filename + ".hdf5", "entropy"));
	// single_site_entropy.save(arma::hdf5_name(dir + filename + ".hdf5", "single_site_entropy", arma::hdf5_opts::append));
    std::cout << " - - - - - - FINISHED ENTROPY CALCUVATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simuVAtion end
}


/// @brief Calculate entanglement entropy in all eigenstates and all subsystem sizes using schmidt decomposition
/// @tparam Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_quadratic<Hamiltonian>::eigenstate_entanglement_manybody()
{
    clk::time_point start = std::chrono::system_clock::now();
	
	std::string dir = this->saving_dir + "Entropy" + kPSep + "ManyBody" + kPSep;
	
	createDirs(dir);
	
	std::string info = this->set_info();
	std::string filename = info;// + "_subsize=" + std::to_string(VA);

	auto dim = this->ptr_to_model->get_hilbert_size();
	const size_t dim_cut = 100000;
    const size_t size = dim > dim_cut? this->l_steps : dim;

	// arma::Col<int> subsystem_sizes = arma::conv_to<arma::Col<int>>::from(arma::linspace(0, this->V / 2, this->V / 2 + 1));
	arma::Col<int> subsystem_sizes = arma::linspace<arma::Col<int>>(0, this->L, this->L + 1);
	// arma::Col<int> subsystem_sizes = arma::join_cols(small_subsystem, arma::Col<int>({this->V / 2}));
	std::cout << subsystem_sizes(0) << "...\t" << subsystem_sizes(subsystem_sizes.size() - 1) << std::endl;

	arma::vec E(size, arma::fill::zeros);
	arma::mat entropies(size, subsystem_sizes.size(), arma::fill::zeros);
	arma::mat single_site_entropy(size, subsystem_sizes.size(), arma::fill::zeros);

	int counter = 0;

	std::vector<QOps::genOp> permutation_op;
	for(int VA_idx = 0; VA_idx < subsystem_sizes.size() - 1; VA_idx++)
	{	
		int VA = subsystem_sizes[VA_idx];
		std::vector<int> p(this->L);
		p[VA % this->L] = 0;
		for(int l = 0; l < this->L; l++){
			if(l != VA % this->V){
				p[l] = (l < (VA % this->V) )? l + 1 : l;
			}
		}
		auto permutation = QOps::_permutation_generator(this->L, p);
		permutation_op.push_back(permutation);
	}
// #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
	for(int realis = 0; realis < this->realisations; realis++)
	{
		this->ptr_to_model->generate_hamiltonian();
		start = std::chrono::system_clock::now();
			
		if(dim > dim_cut){
			double error = this->ptr_to_model->diag_sparse(this->l_steps, this->l_bundle, this->tol, this->seed);
            if( error > 1e-10 ) { std::cout << "POLFED FAILED: Maximal Error = " << error << std::endl; continue; }
		}
		else{
        	this->ptr_to_model->diagonalization();
		}
		std::cout << " - - - - - - finished diagonalization in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simuVAtion end
		
		arma::vec energy = this->ptr_to_model->get_eigenvalues();
		
        //<! Make general for complex matrices
		arma::mat S(size, subsystem_sizes.size(), arma::fill::zeros);
		arma::mat S_site(size, subsystem_sizes.size(), arma::fill::zeros);

		// for(int VA_idx = 0; VA_idx < subsystem_sizes.size(); VA_idx++)
		// {
		// 	auto VA = subsystem_sizes(VA_idx);
		// 	auto start_VA = std::chrono::system_clock::now();
			
		// 	start_VA = std::chrono::system_clock::now();

		// 	double entropy_single_site = 0;
		// 	double entropy = 0;

		// 	for(u64 n = 0; n < num_states; n++){
		// 		auto state_n = mb_states[n];
		// 		E(n) = 0;
		// 		for(long i = 0; i < state_n.size(); i++){
		// 			if(state_n[i])
		// 				E(n) += single_particle_energy(i);
		// 		}
		// 		//<! Generate ope-body density matrix rho -> then do correlator J
		// 		arma::cx_mat J_m(VA, VA, arma::fill::zeros);
		// 		cpx lambda = 0.0;
		// 		QHS::single_particle::correlators::one_body(orbitals, state_n, VA, J_m, lambda, 1.0);
		// 		J_m = 2.0 * J_m - arma::eye(VA, VA);

		// 		arma::vec lambdas = arma::eig_sym(J_m);
		// 		double S_temp = QHS::single_particle::entanglement::vonNeumann(lambdas);
				
		// 		// #pragma omp critical
		// 		{
		// 			entropy 			+= S_temp;
		// 			entropy_single_site += QHS::single_particle::entanglement::vonNeumann_helper(2.0 * std::real(lambda) - 1.0);

		// 		}
		// 		// if( std::abs(entropyyy - entropy_test) > 1e-14)
		// 		// printSeparated(std::cout, "\t", 20, true, VA, mb_states[n], S_temp, entropy_test, entropy_test - S_temp, entropy_test / S_temp);
		// 	}
		// 	S(VA_idx) 		= entropy / (double)num_states;					// entanglement of subsystem VA
		// 	S_site(VA_idx) 	= entropy_single_site / double(num_states);		// single site entanglement at site VA
    	// 	std::cout << " - - - - - - finished entropy size VA: " << VA << " in time:" << tim_s(start_VA) << " s - - - - - - " << std::endl; // simuVAtion end
		// }


		// arma::mat S_mb(size, this->L + 1, arma::fill::zeros);
		// arma::mat S_site_mb = S_mb;

		#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
		for(int n = 0; n < size; n++){
			arma::Col<element_type> state = arma::normalise(this->ptr_to_model->get_eigenState(n));
			arma::Col<element_type> state2 = arma::normalise(this->ptr_to_model->get_eigenState(n));

			state = this->cast_state(state);

			for(int LA_idx = 0; LA_idx < subsystem_sizes.size() - 1; LA_idx++)
			{	
				int LA = subsystem_sizes[LA_idx];
				S(n, LA_idx) = entropy::schmidt_decomposition(state, this->L - LA, this->L);	// bipartite entanglement at subsystem size LA
				
				arma::vec permuted_state = arma::real(permutation_op[LA_idx].multiply(state2));
				S_site(n, LA_idx) = entropy::schmidt_decomposition(permuted_state, this->L - 1, this->L);	// single site entanglement at site LA
			}
		}

		entropies += S;
		single_site_entropy += S_site;
		E += energy;

		outer_threads = this->thread_number;
		omp_set_num_threads(1);


		// if(this->realisations > 1)
		{
			std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
			createDirs(dir_realis);
			S.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "entropy"));
			S_site.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "single_site_entropy", arma::hdf5_opts::append));
			subsystem_sizes.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "VA", arma::hdf5_opts::append));
			energy.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "energy", arma::hdf5_opts::append));
		}
		
		counter++;
    	omp_set_num_threads(this->thread_number);

		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start) << " s - - - - - - " << std::endl; // simuVAtion end
	}
    
	entropies /= double(counter);
	single_site_entropy /= double(counter);
	
	#ifdef MY_MAC
		filename += "_jobid=" + std::to_string(this->jobid);
		entropies.save(arma::hdf5_name(dir + filename + ".hdf5", "entropy"));
		single_site_entropy.save(arma::hdf5_name(dir + filename + ".hdf5", "single_site_entropy", arma::hdf5_opts::append));
		subsystem_sizes.save(arma::hdf5_name(dir + filename + ".hdf5", "VA", arma::hdf5_opts::append));
	#endif
    std::cout << " - - - - - - FINISHED ENTROPY CALCUVATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simuVAtion end
}

/// @brief Calculate entanglement entropy in all eigenstates and all subsystem sizes using schmidt decomposition
/// @tparam Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_quadratic<Hamiltonian>::diagonal_matrix_elements()
{
    clk::time_point start = std::chrono::system_clock::now();
	
	#if DIM == 1
		auto lattice = lattice::lattice1D(this->L, this->boundary_conditions);
	#elif DIM == 2
		auto lattice = lattice::lattice2D(this->L, this->boundary_conditions);
	#else
		auto lattice = lattice::lattice3D(this->L, this->boundary_conditions);
	#endif
	
	start = std::chrono::system_clock::now();

	arma::ivec neighbours(this->V, arma::fill::value(-1));
	arma::ivec next_neighbours(this->V, arma::fill::value(-1));
	arma::ivec next_next_neighbours(this->V, arma::fill::value(-1));
	for(int ell = 0; ell < this->V; ell++){
		auto nei = lattice.get_nearest_neighbour(ell);
		neighbours(ell) = nei;

		nei = lattice.get_next_nearest_neighbour(ell);
		next_neighbours(ell) = nei;

		nei = ell + 3 >= this->V? -1 : ell + 3;
		next_next_neighbours(ell) = nei;
	}
	// std::cout << neighbours << std::endl;
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

			orbitals = this->ptr_to_model->get_eigenvectors();
			// for(int k = 0; k < this->V; k++){
			// 	single_particle_energy(k) = 2.0 * std::cos(two_pi * double(k) / double(this->V));
			// 	for(int ell = 0; ell < this->V; ell++)
			// 		orbitals(ell, k) = std::cos(two_pi * double(k) / double(this->V) * double(ell)) / std::sqrt(this->V);
			// 		// orbitals(ell, k) = std::exp(-1.0i * two_pi * double(k) / double(this->V) * double(ell)) / std::sqrt(this->V);
			// }
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
		arma::Col<element_type> pair_hop(mb_states.size(), arma::fill::zeros);
		arma::Col<element_type> U_nn_loc(mb_states.size(), arma::fill::zeros);
		arma::Col<element_type> U_nnn_loc(mb_states.size(), arma::fill::zeros);
		arma::Col<element_type> pair_hop_loc(mb_states.size(), arma::fill::zeros);

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
				if(neighbours(ell) >= 0 && next_neighbours(ell) >= 0){
					long long nei 			= neighbours(ell);
					long long next_nei 		= next_neighbours(ell);
					long long next_next_nei = next_next_neighbours(ell);
					element_type Al = 0., Al_1 = 0., Al_2 = 0., 
								Bl_1 = 0., Bl_2 = 0., Bl_3 = 0.,
								Bl_21 = 0., Bl_23 = 0.;
								//  Cl_1 = 0., Cl_2 = 0.;
					for(u64 q : set_q){
						Al +=   my_conjungate( orbitals(ell,      q) ) * orbitals(ell, 		q);
						Al_1 += my_conjungate( orbitals(nei,      q) ) * orbitals(nei, 		q);
						Al_2 += my_conjungate( orbitals(next_nei, q) ) * orbitals(next_nei, q);

						Bl_1 += my_conjungate( orbitals(ell, q) ) * orbitals(nei, 	   q);
						Bl_2 += my_conjungate( orbitals(ell, q) ) * orbitals(next_nei, q);

						// Cl_1 += my_conjungate( orbitals(ell, q) * orbitals(nei, 	 q) ) * orbitals(ell, q) * orbitals(nei, 	  q);
						// Cl_2 += my_conjungate( orbitals(ell, q) * orbitals(next_nei, q) ) * orbitals(ell, q) * orbitals(next_nei, q);
						Bl_21 += my_conjungate( orbitals(next_nei, q) ) * orbitals(ell, q);
						if( next_next_nei > 0){
							Bl_3 += my_conjungate( orbitals(ell, q) ) * orbitals(next_next_nei, q);
							Bl_23 += my_conjungate( orbitals(next_nei, q) ) * orbitals(next_next_nei, q);
						}
						for(u64 ell2 = 0; ell2 < ell; ell2++)
							m0(idx) += my_conjungate( orbitals(ell, q) ) * orbitals(ell2, q) + my_conjungate( orbitals(ell2, q) ) * orbitals(ell, q);
					}
					T_nn(idx)  += Bl_1 + my_conjungate(Bl_1);
					T_nnn(idx) += Bl_2 + my_conjungate(Bl_2);
					U_nn(idx)  += Al * Al_1 - Bl_1 * my_conjungate(Bl_1);// + Cl_1;
					U_nnn(idx) += Al * Al_2 - Bl_2 * my_conjungate(Bl_2);// + Cl_2;
					
					element_type _pair_hop_tmp = Bl_1 * Bl_23 - Bl_3 * Bl_21;
					_pair_hop_tmp += my_conjungate(_pair_hop_tmp);
					pair_hop(idx) += _pair_hop_tmp;
					if(ell == this->V / 2 ){
						pair_hop_loc(idx) += _pair_hop_tmp;
						T_nn_loc(idx)  += Bl_1 + my_conjungate(Bl_1);
						T_nnn_loc(idx) += Bl_2 + my_conjungate(Bl_2);
						U_nn_loc(idx)  += Al * Al_1 - Bl_1 * my_conjungate(Bl_1);// + Cl_1;
						U_nnn_loc(idx) += Al * Al_2 - Bl_2 * my_conjungate(Bl_2);// + Cl_2;
					}
				}
			}
			//<! ----
				
		}
		m0 /= double(this->V);
		T_nn  /= std::sqrt(this->V - this->boundary_conditions);
		T_nnn /= std::sqrt(this->V - 2*this->boundary_conditions);
		U_nn  /= std::sqrt(this->V - this->boundary_conditions);
		U_nnn /= std::sqrt(this->V - 2*this->boundary_conditions);
		pair_hop /= std::sqrt(this->V - 3*this->boundary_conditions);


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

		pair_hop.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "pair_hop", arma::hdf5_opts::append));
		pair_hop_loc.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "pair_hop_loc", arma::hdf5_opts::append));

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
					
