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
	if(this->op)	dir += "RandomChoice" + kPSep + "SameHamiltonian" + kPSep;
	else 			dir += "RandomChoice" + kPSep + "DifferentHamiltonian" + kPSep;
	// #ifdef FREE_FERMIONS
	// 	if(this->op == 0) 		dir += "E=0,Q=0" + kPSep;
	// 	else if(this->op == 2)	dir += "AllStates" + kPSep;
	// 	else 					dir += "RandomChoice" + kPSep;
	// #else
	// 	if(this->op == 2)	dir += "AllStates" + kPSep;
	// 	else 				dir += "RandomChoice" + kPSep;
	// #endif
	
	createDirs(dir);
	
	std::string info = this->set_info();
	std::string filename = info;// + "_subsize=" + std::to_string(VA);

	// const int Gamma_max = this->num_of_points;
	u64 num_states = this->num_of_points;//500 * Gamma_max;//ULLPOW(14);
	

	arma::Col<int> subsystem_sizes = arma::conv_to<arma::Col<int>>::from(arma::linspace(1, this->V-1, this->V-1));
	arma::Col<int> subsystem_sizes_MB = subsystem_sizes; //arma::Col<int>({this->V / 2});
	
	arma::Col<int> Gammas = arma::Col<int>({1, 2, 4, 10, this->V / 2, this->V});
	if(this->V < 26){
		Gammas = arma::Col<int>({1, 2, 4, 10, this->V / 2, this->V, 2 * this->V, int(this->V * std::log(this->V))});
		// subsystem_sizes_MB = subsystem_sizes;
	}
	const int Gamma_max = Gammas.size();
	// arma::vec qs = arma::vec({0.5, 1, 2});

	// arma::Col<int> subsystem_sizes = arma::Col<int>({this->V / 6, this->V / 4, this->V / 2, this->V / 2});

	std::cout << subsystem_sizes(0) << "...\t" << subsystem_sizes(subsystem_sizes.size() - 1) << std::endl;

	int counter = 0;


	double filling = 0.5;
	const long N = int(filling * this->V);
    auto _hilbert_space = QHS::U1_hilbert_space<QHS::U1::charge, true>(this->V, N);
	size_t dim = _hilbert_space.get_hilbert_space_size();

	disorder<double> random_generator(this->seed);
	disorder<int> random_integers(this->seed);
	CUE random_matrix(this->seed);
	//GUE

	// printSeparated(std::cout, "\t", 20, true, "VA", "ManyBody state", "S_opdm", "S_schmidt", "S_opdm - S_schmidt");
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
		start = std::chrono::system_clock::now();

		arma::vec single_particle_energy = this->ptr_to_model->get_eigenvalues();
		
		arma::cx_mat orbitals = arma::cx_mat(this->V, this->V, arma::fill::zeros);
        orbitals.set_real(this->ptr_to_model->get_eigenvectors());
		//<! Make general for complex matrices
        

		arma::mat S(Gamma_max, subsystem_sizes_MB.size(), arma::fill::zeros);
		// arma::vec S(Gamma_max, arma::fill::zeros);
		arma::vec S_site(Gamma_max, arma::fill::zeros);
		
		arma::mat S_corr(Gamma_max, subsystem_sizes.size(), arma::fill::zeros);
		arma::mat S_site_corr(Gamma_max, subsystem_sizes.size(), arma::fill::zeros);
		arma::vec NonGauss(Gamma_max, arma::fill::zeros);
		
		arma::mat S_corr_OPDM(Gamma_max, subsystem_sizes.size(), arma::fill::zeros);
		arma::mat S_site_corr_OPDM(Gamma_max, subsystem_sizes.size(), arma::fill::zeros);
		arma::vec NonGauss_OPDM(Gamma_max, arma::fill::zeros);

		// std::vector<boost::dynamic_bitset<>> mb_states;
		// #ifdef FREE_FERMIONS
		// 	if(this->op == 1)		mb_states = QHS::single_particle::mb_config(num_states, this->V, random_generator, N);
		// 	else if(this->op == 2) 	mb_states = QHS::single_particle::mb_config_all(this->V, N);
		// 	else					mb_states = QHS::single_particle::mb_config_free_fermion(this->V, N);

		// 	for(int k = 0; k < this->V; k++){
		// 		single_particle_energy(k) = 2.0 * std::cos(two_pi * double(k) / double(this->V));
		// 		for(int ell = 0; ell < this->V; ell++)
		// 			orbitals(ell, k) = std::exp(-1.0i * two_pi * double(k) / double(this->V) * double(ell)) / std::sqrt(this->V);
		// 	}
		// #else
		// 	orbitals.set_real(this->ptr_to_model->get_eigenvectors());
		// 	if(this->op == 2) 	mb_states = QHS::single_particle::mb_config_all(this->V, N);
		// 	else			 	mb_states = QHS::single_particle::mb_config(num_states, this->V, random_generator, N);
		// #endif
		std::vector<boost::dynamic_bitset<>> mb_states;
		mb_states = QHS::single_particle::mb_config(num_states, this->V, random_generator, N);
		num_states = mb_states.size();
		std::cout << " - - - - - - finished many-body configurations in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl;
		std::cout << "Number of states = \t\t" << num_states << std::endl << std::endl;
		
		start = std::chrono::system_clock::now();

		QHS::single_particle::slater::ManyBodyState<cpx> SlaterConverter(orbitals, _hilbert_space);

		std::cout << " - - - - - - finished setting slater converter in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl;
		// for(auto& VA : subsystem_sizes)
		// arma::mat prs(Gamma_max, qs.size(), arma::fill::zeros);
		// for(int VA_idx = 0; VA_idx < subsystem_sizes.size(); VA_idx++)
		// {
		// 	const long VA = subsystem_sizes(VA_idx);
		// 	auto start_VA = std::chrono::system_clock::now();
			// prs.zeros();

		// #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
			for(int ii = 0; ii < Gammas.size(); ii++)
			{
				int gamma_a = Gammas(ii);
				int counter_states = 0;

				double entropy_single_site = 0;
				// double entropy = 0; 
				arma::vec entropy(subsystem_sizes_MB.size(), arma::fill::zeros);
				arma::vec entropy_corr_mat(subsystem_sizes.size(), arma::fill::zeros);
				arma::vec entropy_corr_mat_OPDM(subsystem_sizes.size(), arma::fill::zeros);
				arma::vec entropy_single_site_corr_mat(subsystem_sizes.size(), arma::fill::zeros);
				arma::vec entropy_single_site_corr_mat_OPDM(subsystem_sizes.size(), arma::fill::zeros);
				
				double non_gaussianity = 0;
				double non_gaussianity_OPDM = 0;
			// #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
				// for(u64 unused = 0; unused < 1; unused++)
				// {
					auto start_G = std::chrono::system_clock::now();
					// auto start_G0 = std::chrono::system_clock::now();
					// arma::vec _prs_(qs.size(), arma::fill::zeros);
					arma::cx_mat U = random_matrix.generate_matrix(gamma_a);
					
					// arma::Col<int> indices = random_integers.uniform(gamma_a, 0, num_states - 1);
					// std::cout << arma::sort(indices) << std::endl;
					arma::Col<int> indices = random_integers.uniform(5 * Gammas(Gamma_max-1), 0, num_states - 1);
					indices = arma::unique(indices);
					indices = indices.rows(0, gamma_a - 1);
					_extra_debug_(  std::cout << arma::sort(indices) << std::endl; )
					int id = random_integers.uniform_dist<int>(0, gamma_a-1);
					
					arma::cx_vec coeff = U.col(id);
					coeff = arma::normalise(coeff);

					QHS::single_particle::slater::ManyBodyState<cpx, false> SlaterConverter(orbitals, _hilbert_space);

					std::vector<arma::cx_mat> _orbitals_;
					std::vector<boost::dynamic_bitset<>> states_for_superposition;
					for(int n = 0; n < gamma_a; n++){
						states_for_superposition.push_back(mb_states[indices(n)]);
						_extra_debug_( std::cout << mb_states[indices(n)] << std::endl; )
						if(this->op == 0)
						{
							// Generate new Gaussian states for different Hamiltonian -------------------------------
							this->ptr_to_model->generate_hamiltonian();
							arma::Mat<element_type> H_temp = this->ptr_to_model->get_dense_hamiltonian();
							arma::vec eigE; 
							arma::Mat<element_type> eigV;
							arma::eig_sym(eigE, eigV, H_temp);
							arma::cx_mat new_orbitals = arma::cx_mat(this->V, this->V, arma::fill::zeros);
							new_orbitals.set_real(eigV);
							_orbitals_.push_back(new_orbitals);
							// --------------------------------------------------------------------------------------
						} else {
							_orbitals_.push_back(orbitals);
						}
					}

					std::cout << "\t- - - - - - finished preamble Gamma = " << gamma_a << " mixings in time:" << tim_s(start_G) << " s - - - - - - " << std::endl; // simuVAtion end
					start_G = std::chrono::system_clock::now();
					
					arma::cx_vec fullstate(ULLPOW(this->V), arma::fill::zeros);
					for(int n = 0; n < gamma_a; n++)
					{
						QHS::single_particle::slater::ManyBodyState<cpx, false>  SlaterConverter(_orbitals_[n], _hilbert_space);
						
						// Fill state with appropriate values ---------------------------------------------------
						SlaterConverter.convert(fullstate, states_for_superposition[n], coeff(n));
						// --------------------------------------------------------------------------------------
					}
					
					fullstate = arma::normalise(fullstate);
					// entropy = entropy::schmidt_decomposition(fullstate, this->V / 2, this->V);
					entropy_single_site = entropy::schmidt_decomposition(fullstate, this->V-1, this->V);
					std::cout << "\t\t - - - - - - finished Many-Body state for Gamma = " << gamma_a << " mixings in time:" << tim_s(start_G) << " s - - - - - - " << std::endl; // simuVAtion end
					start_G = std::chrono::system_clock::now();

					arma::cx_mat J_m_MB(this->V, this->V, arma::fill::zeros);
					for(u64& state : _hilbert_space)
					{
						for(int i = 0; i < this->V; i++)
						{
							auto [_spin, _] = operators::sigma_z(state, this->V, i);
							if( std::real(_spin) > 0){
								J_m_MB(i, i) += std::conj(fullstate(state)) * fullstate(state);
							}
							u64 mask_i = reverseBits( ULLPOW(i)-1, this->V );
							for(int j = i+1; j < this->V; j++)
							{
								u64 mask_j = reverseBits( ULLPOW(j)-1, this->V );
								double sign1 = (__builtin_popcountll(state & mask_i) % 2)? -1 : +1;
								auto [val1, cm] = operators::sigma_minus(state, this->V, j);

								double sign2 = (__builtin_popcountll(cm & mask_j) % 2)? -1 : +1;
		    					auto [val2, cpcm] = operators::sigma_plus(cm, this->V, i);
								if(std::abs(val1 * val2) > 0)
								{
									auto _val_ = std::conj(fullstate(cpcm)) * fullstate(state) * val1 * val2 * sign1 * sign2;
									J_m_MB(i, j) += _val_;
									J_m_MB(j, i) += std::conj(_val_);
									// printSeparated(std::cout, "\t", 20, true, state, boost::dynamic_bitset<>(this->V, state), i, boost::dynamic_bitset<>(this->V, cm), j, boost::dynamic_bitset<>(this->V, cpcm), boost::dynamic_bitset<>(this->V, mask_i), boost::dynamic_bitset<>(this->V, mask_j), sign1, sign2, val1, val2);
								}
							}		
						}	
					}
					
					J_m_MB = 2.0 * J_m_MB - arma::eye(V, V);
					auto lambdas = arma::eig_sym(J_m_MB);
					non_gaussianity = QHS::single_particle::entanglement::vonNeumann(lambdas);

					for(int VA_idx = 0; VA_idx < subsystem_sizes.size(); VA_idx++)
					{
						auto start_VAA = std::chrono::system_clock::now();
						const long VA = subsystem_sizes(VA_idx); 
						arma::uvec row_idx = arma::regspace<arma::uvec>(0, VA-1);
						arma::uvec col_idx = arma::regspace<arma::uvec>(0, VA-1);
						arma::cx_mat J_m_VA = J_m_MB.submat(row_idx, col_idx);

						auto lambdas = arma::eig_sym(J_m_VA);
						entropy_corr_mat(VA_idx) = QHS::single_particle::entanglement::vonNeumann(lambdas);
						
						double lambda = std::real( J_m_MB(VA, VA) );
						entropy_single_site_corr_mat(VA_idx) = QHS::single_particle::entanglement::vonNeumann_helper(lambda);
					}
					std::cout << "\t\t - - - - - - finished correlation matrix using Many-Body state for Gamma = " << gamma_a << " mixings in time:" << tim_s(start_G) << " s - - - - - - " << std::endl; // simuVAtion end
					start_G = std::chrono::system_clock::now();
					for(int VA_idx = 0; VA_idx < subsystem_sizes_MB.size(); VA_idx++)
					{
						auto start_VAA = std::chrono::system_clock::now();
						const long VA = subsystem_sizes_MB(VA_idx);
						entropy(VA_idx) = entropy::schmidt_decomposition(fullstate, this->V - VA, this->V);
						std::cout << "\t\t - - - - - - Schmidt decomposition for VA = " << VA << " mixings in time:" << tim_s(start_VAA) << " s - - - - - - " << std::endl; // simuVAtion end
					}
					// std::cout << J_m_MB << std::endl;
					std::cout << "\t\t - - - - - - finished Schmidt-decompositions from Many-Body state for Gamma = " << gamma_a << " mixings in time:" << tim_s(start_G) << " s - - - - - - " << std::endl; // simuVAtion end
					start_G = std::chrono::system_clock::now();

					cpx normalization = 0;
					for(int n = 0; n < gamma_a; n++)
					{
						auto _matrix_state_n = QHS::single_particle::tools::get_matrix_state(_orbitals_[n], states_for_superposition[n]);
						normalization += std::abs( std::conj(coeff(n)) * coeff(n));
						for(int m = n+1; m < gamma_a; m++)
						{
							auto _matrix_state_m = QHS::single_particle::tools::get_matrix_state(_orbitals_[m], states_for_superposition[m]);
							arma::cx_vec eigs = arma::conj( arma::eig_gen(_matrix_state_n.t() * _matrix_state_m) );
							cpx val = arma::prod(eigs) * std::conj(coeff(n)) * coeff(m);
							normalization += val + std::conj(val);
						}
					}
					coeff = coeff / std::sqrt(normalization);
					std::cout << "\t\t - - - - - - Found normalization for Gamma = " << gamma_a << " mixings with Norm = " << normalization << " in time:" << tim_s(start_G) << " s - - - - - - " << std::endl; // simuVAtion end
					start_G = std::chrono::system_clock::now();

					if(this->op)
					{
						arma::cx_mat OneBodyDensMat(V, V, arma::fill::zeros);
						cpx lambda = 0.0;
						for(int n = 0; n < gamma_a; n++)
						{
							auto state_n = states_for_superposition[n];
							
							// <n|f+_q f_q|n>
							double pre = std::abs( std::conj(coeff(n)) * coeff(n));
							QHS::single_particle::correlators::one_body(orbitals, state_n, V, OneBodyDensMat, lambda, pre);
							
							// <m|f+_q1 f_q2|n> // m<n is included in different q,q'
							for(int m = 0; m < gamma_a; m++)
							{
								auto state_m = states_for_superposition[m];
								auto x = state_n ^ state_m;
								if(x.count() == 2){		// states differ only at two sites, q1 and q2
									std::vector<int> qs;
									for(int q = 0; q < this->V; q++){
										if(x[q]){
											qs.push_back(q);
										}
									}
									
									if(state_n[qs[0]] ^ state_n[qs[1]])	// state n and m differ at q1 and q2 to enable hopping, otherwise skip
									{
										_extra_debug_( std::cout << state_n << "\t\t" << state_m << "\t\t" << x << "\t\t" << qs; )// << std::endl;
										for(int q1 : qs){
											boost::dynamic_bitset<> mask_q1(this->V, ULLPOW(q1) - 1);
											for(int q2 : qs){
												boost::dynamic_bitset<> mask_q2(this->V, ULLPOW(q2) - 1);
												if( (q1 != q2) && state_m[q1] )
												{
													double sign1 = ((state_m & mask_q1).count() % 2)? -1 : 1;
													boost::dynamic_bitset<> _state_m_anih = state_m;
													_state_m_anih[q1] = 0;
													double sign2 = ((_state_m_anih & mask_q2).count() % 2)? -1 : 1;
													cpx pre = sign1 * sign2 * std::conj(coeff(n)) * coeff(m);

													_extra_debug_( std::cout << q1 << "\t\t" << q2 << "\t\t" << state_m[q1] << "\t\t" << state_m[q2] << "\t\t" << sign1 << "\t\t" << sign2 << "\t\t" << pre << std::endl; )
													auto orbital1 = orbitals.col(q1);
													auto orbital2 = orbitals.col(q2);
													OneBodyDensMat += pre * orbital2 * orbital1.t();
												}
											}	
										}
									}
								}
							}
						}

						OneBodyDensMat = 2.0 * OneBodyDensMat - arma::eye(V, V);
						lambdas = arma::eig_sym(OneBodyDensMat);
						non_gaussianity_OPDM = QHS::single_particle::entanglement::vonNeumann(lambdas);

						for(int VA_idx = 0; VA_idx < subsystem_sizes.size(); VA_idx++)
						{
							const long VA = subsystem_sizes(VA_idx); 
							arma::uvec row_idx = arma::regspace<arma::uvec>(this->V - VA, this->V - 1);
							arma::uvec col_idx = arma::regspace<arma::uvec>(this->V - VA, this->V - 1);
							arma::cx_mat ReducedOneBodyDensMat = OneBodyDensMat.submat(row_idx, col_idx);
							auto lambdas = arma::eig_sym(ReducedOneBodyDensMat);
							entropy_corr_mat_OPDM(VA_idx) = QHS::single_particle::entanglement::vonNeumann(lambdas);

							double lambda = std::real( OneBodyDensMat(this->V - 1 - VA, this->V - 1 - VA) );
							entropy_single_site_corr_mat_OPDM(VA_idx) = QHS::single_particle::entanglement::vonNeumann_helper(lambda);
						}
					} else {
						arma::cx_mat OneBodyDensMat(V, V, arma::fill::zeros);
						for(int n = 0; n < gamma_a; n++)
						{
							// cpx lambda = 0;
							// double prefactor = std::abs( std::conj(coeff(n)) * coeff(n));
							// QHS::single_particle::correlators::one_body(_orbitals_[n], states_for_superposition[n], V, OneBodyDensMat_diag, lambda, prefactor);
							
							auto _matrix_state_n = QHS::single_particle::tools::get_matrix_state(_orbitals_[n], states_for_superposition[n]);
							for(int m = 0; m < gamma_a; m++)
							{
								cpx pre = std::conj(coeff(n)) * coeff(m);
								auto _matrix_state_m = QHS::single_particle::tools::get_matrix_state(_orbitals_[m], states_for_superposition[m]);
								for(int i = 0; i < this->V; i++)
								{
									arma::cx_vec created_state_i(this->V, arma::fill::zeros);	
									created_state_i(i) = 1.0;
									arma::cx_mat Wn_ci = arma::join_rows(_matrix_state_n, created_state_i);
									arma::cx_mat Wm_ci = arma::join_rows(_matrix_state_m, created_state_i);
									auto eigs = arma::eig_gen(Wm_ci.t() * Wn_ci);
									cpx val = arma::prod(eigs);
									OneBodyDensMat(i, i) += pre * val; // (1 - ...) because swap of creation operators: ci+ cj -> cj ci+ 
									for(int j = i+1; j < this->V; j++)
									{
										arma::cx_vec created_state_j(this->V, arma::fill::zeros);	
										created_state_j(j) = 1.0;
										arma::cx_mat Wm_cj = arma::join_rows(_matrix_state_m, created_state_j);

										auto eigs = arma::eig_gen(Wm_cj.t() * Wn_ci);
										cpx val = arma::prod(eigs);
										// (-) because swap of creation operators: ci+ cj -> cj ci+ 
										OneBodyDensMat(i, j) += pre * val;
										OneBodyDensMat(j, i) += std::conj( pre * val );
									}
								}
							}
						}
						OneBodyDensMat = ( arma::eye(V, V) - 2.0 * OneBodyDensMat);
						lambdas = arma::eig_sym(OneBodyDensMat);
						non_gaussianity_OPDM = QHS::single_particle::entanglement::vonNeumann(lambdas);

						for(int VA_idx = 0; VA_idx < subsystem_sizes.size(); VA_idx++)
						{
							const long VA = subsystem_sizes(VA_idx); 
							arma::uvec row_idx = arma::regspace<arma::uvec>(this->V - VA, this->V - 1);
							arma::uvec col_idx = arma::regspace<arma::uvec>(this->V - VA, this->V - 1);
							arma::cx_mat ReducedOneBodyDensMat = OneBodyDensMat.submat(row_idx, col_idx);
							auto lambdas = arma::eig_sym(ReducedOneBodyDensMat);
							entropy_corr_mat_OPDM(VA_idx) = QHS::single_particle::entanglement::vonNeumann(lambdas);

							double lambda = std::real( OneBodyDensMat(this->V - 1 - VA, this->V - 1 - VA) );
							entropy_single_site_corr_mat_OPDM(VA_idx) = QHS::single_particle::entanglement::vonNeumann_helper(lambda);
						}
					}
					std::cout << "- - - - - - finished correlation matrix for Gamma = " << gamma_a << " mixings in time:" << tim_s(start_G) << " s - - - - - - " << std::endl << std::endl; // simuVAtion end
					
					// std::cout << J_m_full << std::endl;
					// for(int NA = 0; NA <= min(N, VA); NA++){
					// 	start_G = std::chrono::system_clock::now();
					// 	QHS::U1_subsystem_hilbert_space<QHS::U1::charge, true> _hilbertU1_subA(this->V, VA, NA, N-NA);
					// 	QHS::single_particle::slater::ManyBodyState<cpx, true> SlaterConverter_U1(orbitals, _hilbertU1_subA);
					// 	u64 d_NA = binomial(VA, NA);
					// 	u64 d_NB = binomial(this->V - VA, N - NA);
					// 	arma::cx_vec fullstate(d_NA * d_NB, arma::fill::zeros);
					// 	for(int n = 0; n < gamma_a; n++)
					// 	{
					// 		auto state_n = mb_states[indices(n)];
					// 		// Fill state with appropriate values --------------------------------------------------
					// 		SlaterConverter_U1.convert(fullstate, state_n, coeff(n));
					// 		// --------------------------------------------------------------------------------------
					// 	}
					// 	entropy += entropy::schmidt_decomposition_dims(fullstate, d_NA, d_NB);
					// 	std::cout << "\t\t - - - - - - finished entanglement SVD with U(1) for Gamma = " << gamma_a << " and NA = " << NA << " mixings in time:" << tim_s(start_G) << " s - - - - - - " << std::endl; // simuVAtion end
					// }
					// std::cout << "- - - - - - finished entanglement with U(1) State for Gamma = " << gamma_a << " mixings in time:" << tim_s(start_G0) << " s - - - - - - " << std::endl; // simuVAtion end
					// start_G0 = std::chrono::system_clock::now();
					// arma::cx_mat J_m(VA, VA, arma::fill::zeros);
					// arma::cx_mat J_m_full(this->V, this->V, arma::fill::zeros);
					// cpx lambda = 0.0;
					// for(int n = 0; n < gamma_a; n++)
					// {
					// 	auto state_n = mb_states[indices(n)];
					// 	// <n|f+_q f_q|n>
					// 	double pre = std::abs(coeff(n)) * std::abs(coeff(n));
					// 	QHS::single_particle::correlators::one_body(orbitals, state_n, VA, J_m, lambda, pre);
					// 	QHS::single_particle::correlators::one_body(orbitals, state_n, V, J_m_full, lambda, pre);
					// 	// <m|f+_q1 f_q2|n>
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
					// 				for(auto& qss : v_2d<int>( { qs, v_1d<int>({qs[1], qs[0]}) } ) )
					// 				{
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
					// 					auto orbital1 = orbitals.col(q1);
					// 					auto orbital2 = orbitals.col(q2);
					// 					J_m_full += pre * orbital2 * orbital1.t();
					// 				}
					// 			}
					// 		}
					// 	}
					// }
					// J_m = 2.0 * J_m - arma::eye(VA, VA);
					// auto lambdas = arma::eig_sym(J_m);
					// entropy_corr_mat += QHS::single_particle::entanglement::vonNeumann(lambdas);
					// J_m_full = 2.0 * J_m_full - arma::eye(V, V);
					// lambdas = arma::eig_sym(J_m_full);
					// non_gaussianity += QHS::single_particle::entanglement::vonNeumann(lambdas);
					// std::cout << "- - - - - - finished correlation matrix for Gamma = " << gamma_a << " mixings in time:" << tim_s(start_G0) << " s - - - - - - " << std::endl; // simuVAtion end
					
					// prs.row(ii) += _prs_.t();
					counter_states++;
				// }
				// participation_ratios(ii) = prs(ii) / (double)counter_states;
				for(int VA_idx = 0; VA_idx < subsystem_sizes.size(); VA_idx++)
				{
					S_corr(ii, VA_idx) 		 = entropy_corr_mat(VA_idx);// / (double)counter_states;		// entanglement of subsystem VA using Gaussian approx from Many-Body state
					S_corr_OPDM(ii, VA_idx)  = entropy_corr_mat_OPDM(VA_idx);// / (double)counter_states;	// entanglement of subsystem VA using Gaussian approx from Gaussian calculation
					S_site_corr(ii, VA_idx) 	  = entropy_single_site_corr_mat(VA_idx);// / (double)counter_states;		// entanglement of single site VA using Gaussian approx from Many-Body state
					S_site_corr_OPDM(ii, VA_idx)  = entropy_single_site_corr_mat_OPDM(VA_idx);// / (double)counter_states;	// entanglement of singel site VA using Gaussian approx from Gaussian calculation
				}
				for(int VA_idx = 0; VA_idx < subsystem_sizes_MB.size(); VA_idx++)
				{
					S(ii, VA_idx) 			 = entropy(VA_idx);// / (double)counter_states;					// entanglement of subsystem VA using Slater determiniants
				}

				// S(ii) 			  = entropy;
				NonGauss(ii)	  = non_gaussianity;// / (double)counter_states;		// non-gaussianity using Many-body state
				NonGauss_OPDM(ii) = non_gaussianity_OPDM;// / (double)counter_states;	// non-gaussianity using Gaussian approx
				S_site(ii) 		  = entropy_single_site;// / double(counter_states);	// single site entanglement at site VA
				// std::cout << "\n - - - - - - finished entropy size VA: " << VA << " with Gamma = " << gamma_a << " mixings in time:" << tim_s(start_VA) << " s - - - - - - " << std::endl; // simuVAtion end
			}
    	// 	std::cout << "\n - - - - - - finished entropy size VA: " << VA << " in time:" << tim_s(start_VA) << " s - - - - - - " << std::endl; // simuVAtion end
		// }

		// if(this->realisations > 1)
		{
			std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
			createDirs(dir_realis);
			S.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "entropy"));
			S_corr.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "entropy_corr_mat", arma::hdf5_opts::append));
			S_site_corr.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "entropy_single_site_corr_mat", arma::hdf5_opts::append));
			NonGauss.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "Non-Gaussianity", arma::hdf5_opts::append));
			S_site.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "single_site_entropy", arma::hdf5_opts::append));
			subsystem_sizes.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "subsystem_sizes", arma::hdf5_opts::append));
			subsystem_sizes_MB.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "subsystem_sizes_MB", arma::hdf5_opts::append));
			Gammas.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "Gammas", arma::hdf5_opts::append));
			S_corr_OPDM.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "GaussCalc/entropy_corr_mat", arma::hdf5_opts::append));
			S_site_corr_OPDM.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "GaussCalc/entropy_single_site_corr_mat", arma::hdf5_opts::append));
			NonGauss_OPDM.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "GaussCalc/Non-Gaussianity", arma::hdf5_opts::append));
			// -------- COMPARISON
			arma::mat x = S_corr_OPDM - S_corr;
			x.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "COMPARISON/entropy", arma::hdf5_opts::append));
			arma::vec y = NonGauss_OPDM - NonGauss;
			y.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "COMPARISON/Non-Gaussianity", arma::hdf5_opts::append));
			arma::mat z = S_site_corr_OPDM - S_site_corr;
			z.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "COMPARISON/single_site_entropy", arma::hdf5_opts::append));
			// qs.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "qs", arma::hdf5_opts::append));
			// prs.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "participation_ratio", arma::hdf5_opts::append));
			// single_particle_energy.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "single particle energy", arma::hdf5_opts::append));
		}
		
		counter++;
    	// omp_set_num_threads(this->thread_number);

		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start) << " s - - - - - - " << std::endl; // simuVAtion end
	}
    
    std::cout << " - - - - - - FINISHED ENTROPY CALCUVATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simuVAtion end
}

/// @brief Calculate entanglement entropy for randomly mixed many-body states
/// @tparam Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_quadratic<Hamiltonian>::non_gaussianity()
{
    clk::time_point start = std::chrono::system_clock::now();
	
	std::string dir = this->saving_dir + "Entropy" + kPSep + "NonGaussianity" + kPSep;
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
	u64 num_states = this->num_of_points;//500 * Gamma_max;//ULLPOW(14);
	
	double filling = 0.5;
	const long N = int(filling * this->V);

	// arma::Col<int> Gammas = arma::linspace<arma::Col<int>>(1, this->V, this->V);
	arma::Col<int> Gammas = arma::linspace<arma::Col<int>>(1, 5*this->V, 5*this->V);
	if(5*this->V > 200) Gammas = arma::join_cols(arma::linspace<arma::Col<int>>(1, 50, 50), arma::regspace<arma::Col<int>>(100, 10, 5*this->V));
	else if(5*this->V > 2000) Gammas = arma::join_cols(arma::linspace<arma::Col<int>>(1, 50, 50), arma::regspace<arma::Col<int>>(100, 50, 5*this->V));
	else if(5*this->V > 5000) Gammas = arma::join_cols(arma::linspace<arma::Col<int>>(1, 50, 50), arma::regspace<arma::Col<int>>(100, 100, 5*this->V));
	// const int Gamma_max = Gammas.size();
	std::cout << Gammas << std::endl;
	// arma::vec qs = arma::vec({0.5, 1, 2});

	// arma::Col<int> subsystem_sizes = arma::conv_to<arma::Col<int>>::from(arma::linspace(0, this->V / 2, this->V / 2 + 1));
	arma::Col<int> subsystem_sizes = arma::Col<int>({this->V / 10, this->V / 5, this->V / 4, this->V / 2, this->V});
	// arma::Col<int> subsystem_sizes = arma::Col<int>({this->V / 6, this->V / 4, this->V / 2, this->V / 2});

	std::cout << subsystem_sizes(0) << "...\t" << subsystem_sizes(subsystem_sizes.size() - 1) << std::endl;

	arma::mat entropies(Gammas.size(), subsystem_sizes.size(), arma::fill::zeros);
	arma::mat single_site_entropy(Gammas.size(), subsystem_sizes.size(), arma::fill::zeros);
	// arma::mat participation_ratios(Gamma_max, qs.size(), arma::fill::zeros);

	int counter = 0;


	disorder<double> random_generator(this->seed);
	disorder<int> random_integers(this->seed);
	CUE random_matrix(this->seed);
	//GUE

	// printSeparated(std::cout, "\t", 20, true, "VA", "ManyBody state", "S_opdm", "S_schmidt", "S_opdm - S_schmidt");
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
		start = std::chrono::system_clock::now();

		arma::vec single_particle_energy = this->ptr_to_model->get_eigenvalues();
		
		arma::cx_mat orbitals = arma::cx_mat(this->V, this->V, arma::fill::zeros);
        //<! Make general for complex matrices
        
		arma::mat S_site(Gammas.size(), subsystem_sizes.size(), arma::fill::zeros);
		arma::mat S(Gammas.size(), subsystem_sizes.size(), arma::fill::zeros);
		arma::mat count_offdiag(Gammas.size(), subsystem_sizes.size(), arma::fill::zeros);

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
		
		num_states = mb_states.size();
		
		std::cout << " - - - - - - finished many-body configurations in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl;
		std::cout << "Number of states = \t\t" << num_states << std::endl << std::endl;
		start = std::chrono::system_clock::now();

		for(int VA_idx = 0; VA_idx < subsystem_sizes.size(); VA_idx++)
		{
			const long VA = subsystem_sizes(VA_idx);
			auto start_VA = std::chrono::system_clock::now();
			
			start_VA = std::chrono::system_clock::now();
			// prs.zeros();

		// #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
			for(int ii = 0; ii < Gammas.size(); ii++)
			{
				auto start_G = std::chrono::system_clock::now();
				int gamma_a = Gammas(ii);
				int counter_states = 0;

				double entropy_single_site = 0;
				double entropy = 0;
			// #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
				for(u64 unused = 0; unused < 1; unused++)
				{
					// arma::vec _prs_(qs.size(), arma::fill::zeros);
					arma::cx_mat U = random_matrix.generate_matrix(gamma_a);
					_extra_debug_( std::cout << "\t\t\t\t - - - - - - finished preamble: Haar matrix" << std::endl; )
					arma::Col<int> indices = random_integers.uniform(gamma_a, 0, num_states - 1);
					int id = random_integers.uniform_dist<int>(0, gamma_a-1);
					_extra_debug_( std::cout << "\t\t\t\t - - - - - - finished preamble: random integers for choosing coefficient and many-body states" << std::endl; )

					arma::cx_vec coeff = U.col(id);
					coeff = arma::normalise(coeff);
					_extra_debug_( std::cout << "\t\t\t\t - - - - - - finished preamble: chosen set of coefficients" << std::endl; )
					
					_extra_debug_( std::cout << "\t\t - - - - - - finished preamble Gamma = " << gamma_a << " mixings in time:" << tim_s(start_G) << " s - - - - - - " << std::endl; )
					start_G = std::chrono::system_clock::now();

					arma::cx_mat J_m(VA, VA, arma::fill::zeros);
					// arma::cx_mat J_m_full(this->V, this->V, arma::fill::zeros);
					cpx lambda = 0.0;
					int count_of_offdiag = 0;
				#pragma omp parallel for
					for(int n = 0; n < gamma_a; n++)
					{
						auto state_n = mb_states[indices(n)];
						arma::cx_mat J_m_temp(VA, VA, arma::fill::zeros);
						cpx _lambda_tmp = 0.0;

						// <n|f+_q f_q|n>
						double pre = std::abs(coeff(n)) * std::abs(coeff(n));
						QHS::single_particle::correlators::one_body(orbitals, state_n, VA, J_m_temp, _lambda_tmp, pre);

						int _tmp_ = 0;
						// <m|f+_q1 f_q2|n>
						for(int m = n + 1; m < gamma_a; m++)
						{
							auto state_m = mb_states[indices(m)];
						
							// arma::cx_mat J_m_tmp(VA, VA, arma::fill::zeros);
							auto x = state_n ^ state_m;
							if(x.count() == 2){		// states differ only at two sites, q1 and q2
								_tmp_++;
								// std::cout << state_n << "\t\t" << state_m << std::endl;
								std::vector<int> qs;
								auto prefactor = std::conj(coeff(m)) * coeff(n);
								for(int q = 0; q < this->V; q++)
									if(x[q]) qs.push_back(q);
									
								if(state_n[qs[0]] ^ state_n[qs[1]])	// state n and m differ at q1 and q2 to enable hopping, otherwise skip
								{
									for(auto& qss : v_2d<int>( { qs, v_1d<int>({qs[1], qs[0]}) } ) )
									{
										int q1 = qss[0];
										int q2 = qss[1];

										cpx pre = prefactor;
										if(state_n[q1])		// for one of the 2 cases do conjungation
											pre = std::conj(prefactor);
										
										if(VA < V)
											_lambda_tmp += pre * std::abs(orbitals(q2, VA) * std::conj(orbitals(q1, VA)));
										
										if(VA > 0){
											auto orbital1 = orbitals.col(q1).rows(0, VA - 1);
											auto orbital2 = orbitals.col(q2).rows(0, VA - 1);
											J_m_temp += pre * orbital2 * orbital1.t();
										}
									}
								}
							}
						}
					#pragma omp critical
						{
							J_m += J_m_temp;
							lambda += _lambda_tmp;
							count_of_offdiag += _tmp_;
						}
					}
					J_m = 2.0 * J_m - arma::eye(VA, VA);
					auto lambdas = arma::eig_sym(J_m);
					entropy += QHS::single_particle::entanglement::vonNeumann(lambdas);
					entropy_single_site += QHS::single_particle::entanglement::vonNeumann_helper(2.0 * std::real(lambda) - 1.0);
					count_offdiag(ii, VA_idx) = count_of_offdiag;
					// std::cout << "\t\t- - - - - - finished correlation matrix for Gamma = " << gamma_a << " mixings in time:" << tim_s(start_G) << " s - - - - - - " << std::endl; // simuVAtion end
					
					// prs.row(ii) += _prs_.t();
					counter_states++;
				}
				// participation_ratios(ii) = prs(ii) / (double)counter_states;
				S(ii, VA_idx) 		= entropy / (double)counter_states;				// entanglement of subsystem VA using Slater determiniants
				S_site(ii, VA_idx) 	= entropy_single_site / double(counter_states);	// single site entanglement at site VA
				std::cout << "\t - - - - - - finished entropy size VA: " << VA << " with Gamma = " << gamma_a << " mixings in time:" << tim_s(start_G) << " s - - - - - - " << std::endl; // simuVAtion end
			}
    		std::cout << " - - - - - - finished entropy size VA: " << VA << " in time:" << tim_s(start_VA) << " s - - - - - - " << std::endl; // simuVAtion end
		}
		if(count_offdiag.is_zero())
			std::cout << " - - - - - - IN THIS REALISATION NO CONTRIBUTIONS BETWEEN DIFFERENT STATES WERE FOUND...  - - - - - - ";

		// if(this->realisations > 1)
		{
			std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
			createDirs(dir_realis);
			S.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "entropy"));
			S_site.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "single_site_entropy", arma::hdf5_opts::append));
			subsystem_sizes.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "subsystem_sizes", arma::hdf5_opts::append));
			Gammas.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "Gammas", arma::hdf5_opts::append));
			count_offdiag.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "counter_off_diag", arma::hdf5_opts::append));
			// qs.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "qs", arma::hdf5_opts::append));
			// prs.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "participation_ratio", arma::hdf5_opts::append));
			single_particle_energy.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "single particle energy", arma::hdf5_opts::append));
		}
		entropies += S;
		single_site_entropy += S_site;
		
		counter++;
    	// omp_set_num_threads(this->thread_number);

		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start) << " s - - - - - - " << std::endl; // simuVAtion end
	}
    
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
//						
						// <m|f+_q1 f_q2|n>
					// 	for(int m = n + 1; m < gamma_a; m++)
					// 	{
					// 		auto state_m = mb_states[indices(m)];
//						
					// 		// arma::cx_mat J_m_tmp(VA, VA, arma::fill::zeros);
					// 		auto x = state_n ^ state_m;
					// 		if(x.count() == 2){		// states differ only at two sites, q1 and q2
					// 			// std::cout << state_n << "\t\t" << state_m << std::endl;
					// 			std::vector<int> qs;
					// 			auto prefactor = std::conj(coeff(m)) * coeff(n);
					// 			for(int q = 0; q < this->V; q++)
					// 				if(x[q]) qs.push_back(q);
//									
					// 			if(state_n[qs[0]] ^ state_n[qs[1]])	// state n and m differ at q1 and q2 to enable hopping, otherwise skip
					// 			{
					// 				for(auto& qss : v_2d<int>( { qs, v_1d<int>({qs[1], qs[0]}) } ) ){
					// 					int q1 = qss[0];
					// 					int q2 = qss[1];
//
					// 					cpx pre = prefactor;
					// 					if(state_n[q1])		// for one of the 2 cases do conjungation
					// 						pre = std::conj(prefactor);
//										
					// 					lambda += pre * std::abs(orbitals(q2, VA) * std::conj(orbitals(q1, VA)));
//										
					// 					if(VA > 0){
					// 						auto orbital1 = orbitals.col(q1).rows(0, VA - 1);
					// 						auto orbital2 = orbitals.col(q2).rows(0, VA - 1);
					// 						J_m += pre * orbital2 * orbital1.t();
					// 					}
					// 				}
					// 			}
					// 		}
					// 	}
//
					// fullstate = arma::normalise(fullstate);
					// J_m = 2.0 * J_m - arma::eye(VA, VA);
					// auto lambdas = arma::eig_sym(J_m);
						// entropy 			+= QHS::single_particle::entanglement::vonNeumann(lambdas);
						// entropy_single_site += QHS::single_particle::entanglement::vonNeumann_helper(2.0 * std::real(lambda) - 1.0);
					




// COMPARISON OF FULL STATE TO U(1) DECOMPOSED STATES ENTANGLEMENT FOR DEGENERATE MIXING
// #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
// 				for(u64 unused = 0; unused < 1; unused++)
// 				{
//					
// 					auto start_G = std::chrono::system_clock::now();
// 					arma::vec _prs_(qs.size(), arma::fill::zeros);
// 					arma::cx_mat U = random_matrix.generate_matrix(gamma_a);
// 					arma::cx_mat J_m(VA, VA, arma::fill::zeros);
//					
// 					arma::Col<int> indices = random_integers.uniform(gamma_a, 0, num_states - 1);
// 					int id = random_integers.uniform_dist<int>(0, gamma_a-1);
// 					cpx lambda = 0.0;
//					
// 					arma::cx_vec coeff = U.col(id);
// 					coeff = arma::normalise(coeff);
//					
// 					start_G0 = std::chrono::system_clock::now();
// 					std::cout << "\t\t - - - - - - finished preamble Gamma = " << gamma_a << " mixings in time:" << tim_s(start_G) << " s - - - - - - " << std::endl; // simuVAtion end
// 					start_G = std::chrono::system_clock::now();
//
// 					arma::cx_vec fullstate(ULLPOW(this->V), arma::fill::zeros);
// 					// auto starta = std::chrono::system_clock::now();
// 					for(int n = 0; n < gamma_a; n++)
// 					{
// 						auto state_n = mb_states[indices(n)];
//
// 						// Fill state with appropriate values --------------------------------------------------
// 						SlaterConverter.convert(fullstate, state_n, coeff(n), qs, _prs_);
// 						// --------------------------------------------------------------------------------------
// 					}
// 					std::cout << "\t\t - - - - - - finished creating State for Gamma = " << gamma_a << " mixings in time:" << tim_s(start_G) << " s - - - - - - " << std::endl; // simuVAtion end
// 					start_G = std::chrono::system_clock::now();
// 					entropy += entropy::schmidt_decomposition(fullstate, VA, this->V);
//
// 					std::cout << "\t\t - - - - - - finished entanglement SVD for Gamma = " << gamma_a << " mixings in time:" << tim_s(start_G) << " s - - - - - - " << std::endl; // simuVAtion end
// 					start_G = std::chrono::system_clock::now();
//					
// 					std::cout << "- - - - - - finished entanglement with Full state for Gamma = " << gamma_a << " mixings in time:" << tim_s(start_G0) << " s - - - - - - " << std::endl; // simuVAtion end
// 					start_G0 = std::chrono::system_clock::now();
//
// 					arma::vec schmidt_values;
// 					for(int NA = 0; NA <= min(N, VA); NA++){
// 						QHS::U1_subsystem_hilbert_space<QHS::U1::charge, true> _hilbertU1_subA(this->V, VA, NA, N-NA);
// 						QHS::single_particle::slater::ManyBodyState<cpx, true> SlaterConverter_U1(orbitals, _hilbertU1_subA);
//
// 						u64 d_NA = binomial(VA, NA);
// 						u64 d_NB = binomial(this->V - VA, N - NA);
// 						printSeparated(std::cout, "\t", 20, true, d_NA, d_NB, _hilbertU1_subA.get_hilbert_space_size());
// 						arma::cx_vec fullstate(d_NA * d_NB, arma::fill::zeros);
// 						// auto starta = std::chrono::system_clock::now();
// 						for(int n = 0; n < gamma_a; n++)
// 						{
// 							auto state_n = mb_states[indices(n)];
// 
// 							// Fill state with appropriate values --------------------------------------------------
// 							SlaterConverter_U1.convert(fullstate, state_n, coeff(n), qs, _prs_);
// 							// --------------------------------------------------------------------------------------
// 						}
// 						entropy_test += entropy::schmidt_decomposition_dims(fullstate, d_NA, d_NB);
// 						std::cout << "\t\t - - - - - - finished entanglement SVD with U(1) for Gamma = " << gamma_a << " and NA = " << NA << " mixings in time:" << tim_s(start_G) << " s - - - - - - " << std::endl; // simuVAtion end
// 						start_G = std::chrono::system_clock::now();
// 					}
// 					std::cout << "- - - - - - finished entanglement with U(1) State for Gamma = " << gamma_a << " mixings in time:" << tim_s(start_G0) << " s - - - - - - " << std::endl; // simuVAtion end
// 					// std::cout << "\n - - - - - - finished Many Body state in time:" << tim_s(starta) << " s - - - - - - " << std::endl; // simuVAtion end
// 					// starta = std::chrono::system_clock::now();
//					// 
// 					// std::cout << "\n - - - - - - finished entropy of Many Body state in time:" << tim_s(starta) << " s - - - - - - " << std::endl; // simuVAtion end
// 
// 					prs.row(ii) += _prs_.t();
// 					counter_states++;
// 				}