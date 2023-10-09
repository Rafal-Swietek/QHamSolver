#pragma once


/// @brief Diagonalize model hamiltonian and save spectrum to .hdf5 file
/// @tparam Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_dis<Hamiltonian>::diagonalize(){
	clk::time_point start = std::chrono::system_clock::now();
	std::string dir = this->saving_dir + "DIAGONALIZATION" + kPSep;
	createDirs(dir);

#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
	for(int realis = 0; realis < this->realisations; realis++)	
	{
		int real = realis + this->jobid;
		std::string _suffix = "_real=" + std::to_string(real);
		#ifdef USE_SYMMETRIES
			//<! no suffix for symmetric model
			_suffix = "";
		#endif
		std::string info = this->set_info({});
		std::cout << "\n\t\t--> finished creating model for " << info + _suffix << " - in time : " << tim_s(start) << "s" << std::endl;
		
		if(realis > 0)
			this->ptr_to_model->generate_hamiltonian();
		this->ptr_to_model->diagonalization(!this->ch);
		arma::vec eigenvalues = this->ptr_to_model->get_eigenvalues();
		
		std::cout << "\t\t	--> finished diagonalizing for " << info + _suffix << " - in time : " << tim_s(start) << "s" << std::endl;
		
		// std::cout << "Energies:\n";
		// std::cout << eigenvalues << std::endl;

		std::string name = dir + info + _suffix + ".hdf5";
		eigenvalues.save(arma::hdf5_name(name, "eigenvalues", arma::hdf5_opts::append));
		std::cout << "\t\t	--> finished saving eigenvalues for " << info + _suffix << " - in time : " << tim_s(start) << "s" << std::endl;
		if(this->ch){
			auto H = this->ptr_to_model->get_dense_hamiltonian();
			H.save(arma::hdf5_name(name, "hamiltonian", arma::hdf5_opts::append));
			std::cout << "\t\t	--> finished saving Hamiltonian for " << info << " - in time : " << tim_s(start) << "s" << std::endl;

			auto V = this->ptr_to_model->get_eigenvectors();
			V.save(arma::hdf5_name(name, "eigenvectors", arma::hdf5_opts::append));
			std::cout << "\t\t	--> finished saving eigenvectors for " << info << " - in time : " << tim_s(start) << "s" << std::endl;
		}
	};
	
}

/// @brief Calculate spectral form factor
/// @tparam Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_dis<Hamiltonian>::spectral_form_factor(){
	clk::time_point start = std::chrono::system_clock::now();
	
	std::string dir = this->saving_dir + "SpectralFormFactor" + kPSep;
	if(this->beta > 0){
		dir += "beta=" + to_string_prec(this->beta) + kPSep;
	}
	createDirs(dir);
	
	//------- PREAMBLE
	std::string info = this->set_info();

	const double chi = 0.341345;
	u64 dim = this->ptr_to_model->get_hilbert_size();

	const double wH = sqrt(this->L) / (chi * dim);
	double tH = 1. / wH;
	double r1 = 0.0, r2 = 0.0;
	int time_end = (int)std::ceil(std::log10(5 * tH));
	time_end = (time_end / std::log10(tH) < 1.5) ? time_end + 1 : time_end;

	arma::vec times = arma::logspace(log10(1.0 / (two_pi * dim)), 1, this->num_of_points);
	arma::vec times_fold = arma::logspace(-2, time_end, this->num_of_points);

	arma::vec sff(this->num_of_points, arma::fill::zeros);
	arma::vec sff_fold(this->num_of_points, arma::fill::zeros);
	double Z = 0.0, Z_fold = 0.0;
	double wH_mean = 0.0;
	double wH_typ  = 0.0;
	
//#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
	for(int realis = 0; realis < this->realisations; realis++)
	{
		std::string suffix = "_real=" + std::to_string(realis + this->jobid);
		if(realis > 0)
			this->ptr_to_model->generate_hamiltonian();
		arma::vec eigenvalues = this->get_eigenvalues(suffix);
		
		
		if(this->fun == 1) std::cout << "\t\t	--> finished loading eigenvalues for " << info + suffix << " - in time : " << tim_s(start) << "s" << std::endl;
		if(eigenvalues.empty()) continue;
		dim = eigenvalues.size();

		u64 E_av_idx = spectrals::get_mean_energy_index(eigenvalues);
		const u64 num = dim < 1000? 0.25 * dim : 0.5 * dim;
		const u64 num2 = dim < 2000? 100 : 500;

		// ------------------------------------- calculate level statistics
			double r1_tmp = 0, r2_tmp = 0, wH_mean_r = 0, wH_typ_r = 0;
			int count = 0;
			for(int i = (E_av_idx - num / 2); i < (E_av_idx + num / 2); i++){
				const double gap1 = eigenvalues(i) - eigenvalues(i - 1);
				const double gap2 = eigenvalues(i + 1) - eigenvalues(i);
				const double min = std::min(gap1, gap2);
				const double max = std::max(gap1, gap2);
				wH_mean_r += gap2;
				wH_typ_r += std::log(gap2);
        		if (abs(gap1) <= 1e-15 || abs(gap2) <= 1e-15){ 
        		    std::cout << "Index: " << i << std::endl;
        		    _assert_(false, "Found degeneracy, while doing r-statistics!\n");
        		}
				r1_tmp += min / max;
				if(i >= (E_av_idx - num2 / 2) && i < (E_av_idx + num2 / 2))
					r2_tmp += min / max;
				count++;
			}
			wH_mean_r /= double(count);
			wH_typ_r = std::exp(wH_typ_r / double(count));
			r1_tmp /= double(count);
			r2_tmp /= double(num2);
			if(this->fun == 1) std::cout << "\t\t	--> finished unfolding for " << info + suffix << " - in time : " << tim_s(start) << "s" << std::endl;
		
		// ------------------------------------- calculate sff
			auto [sff_r_folded, Z_r_folded] = statistics::spectral_form_factor(eigenvalues, times_fold, this->beta, 0.5);
			eigenvalues = statistics::unfolding(eigenvalues);

			auto [sff_r, Z_r] = statistics::spectral_form_factor(eigenvalues, times,this->beta, 0.5);
			#pragma omp critical
			{
				r1 += r1_tmp;
				r2 += r2_tmp;

				sff += sff_r;
				Z += Z_r;
				sff_fold += sff_r_folded;
				Z_fold += Z_r_folded;
				
				wH_mean += wH_mean_r;
				wH_typ  += wH_typ_r;
			}
		if(this->fun == 1) std::cout << "\t\t	--> finished realisation for " << info + suffix << " - in time : " << tim_s(start) << "s" << std::endl;
		
		//--------- SAVE REALISATION TO FILE
		#if !defined(MY_MAC)
			std::string dir_re  = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
			createDirs(dir_re);
			save_to_file(dir_re + info + ".dat", 			times, 		sff_r, 		  Z_r, 		  r1_tmp, r2_tmp, wH_mean_r, wH_typ_r);
			save_to_file(dir_re + "folded" + info + ".dat", times_fold, sff_r_folded, Z_r_folded, r1_tmp, r2_tmp, wH_mean_r, wH_typ_r);
		#else
			std::cout << this->jobid + realis << std::endl;
		#endif
	}

	// --------------------------------------------------------------- AVERAGE CURRENT REALISATIONS
	if(sff.is_empty()) return;
	if(sff.is_zero()) return;
	if(this->jobid > 0) return;

	double norm = this->realisations;
	r1 /= norm;
	r2 /= norm;
	sff = sff / Z;
	sff_fold = sff_fold / Z_fold;
	wH_mean /= norm;
	wH_typ /= norm;

	// ---------- find Thouless time
	double eps = 8e-2;
	auto K_GOE = [](double t){
		return t < 1? 2 * t - t * log(1+2*t) : 2 - t * log( (2*t+1) / (2*t-1) );
	};
	double thouless_time = 0;
	double t_max = 2.5;
	double delta_min = 1e6;
	for(int i = 0; i < sff.size(); i++){
		double t = times(i);
		double delta = abs(log10( sff(i) / K_GOE(t) )) - eps;
		delta *= delta;
		if(delta < delta_min){
			delta_min = delta;
			thouless_time = times(i); 
		}
		if(times(i) >= t_max) break;
	}
	save_to_file(dir + info + ".dat", 			 times, 	 sff, 	   1.0 / wH_mean, thouless_time, 		   r1, r2, dim, 1.0 / wH_typ);
	save_to_file(dir + "folded" + info + ".dat", times_fold, sff_fold, 1.0 / wH_mean, thouless_time / wH_mean, r1, r2, dim, 1.0 / wH_typ);
}

/// @brief Average sff over realisations
/// @tparam Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_dis<Hamiltonian>::average_sff(){

	std::string dir = this->saving_dir + "SpectralFormFactor" + kPSep;
	std::string info = this->set_info();
	arma::vec times, times_fold; // are always the same
	arma::vec sff(this->num_of_points, arma::fill::zeros);
	arma::vec sff_fold(this->num_of_points, arma::fill::zeros);
	double Z = 0.0;
	double Z_folded = 0.0;
	double r1 = 0.0;
	double r2 = 0.0;
	double wH = 0.;
	double wH_typ = 0.;
	size_t dim = this->ptr_to_model->get_hilbert_size();
	int counter_realis = 0;
	
	outer_threads = this->thread_number;
	omp_set_num_threads(1);
	std::cout << "THREAD COUNT:\t\t" << outer_threads << "\t\t" << omp_get_num_threads() << std::endl;
#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
	for(int realis = 0; realis < this->realisations; realis++)
	{
		std::string dir_re  = this->saving_dir + "SpectralFormFactor" + kPSep + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
		std::ifstream file;
		
		auto data = readFromFile(file, dir_re + info + ".dat");
		if(data.empty()) continue;
		if(data[0].size() != sff.size()) {
			std::cout << "Incompatible data dimensions" << std::endl;
			continue;
		}
		file.close();
		#pragma omp critical
		{
			times = data[0];
			sff += data[1];
			Z += data[2](0);
			r1 += data[3](0);
			r2 += data[4](0);
			wH += data[5](0);
			wH_typ += data[6](0);
			counter_realis++;
		}

		data = readFromFile(file, dir_re + "folded" + info + ".dat");
		if(data.empty()) continue;
		if(data[0].size() != sff_fold.size()) {
			std::cout << "Incompatible data dimensions" << std::endl;
			continue;
		}
		file.close();
		#pragma omp critical
		{
			times_fold = data[0];
			sff_fold += data[1];
			Z_folded += data[2](0);
		}
	}

	if(sff.is_empty()) return;
	if(sff.is_zero()) return;

	double norm = counter_realis;
	r1 /= norm;
	r2 /= norm;
	sff = sff / Z;
	sff_fold = sff_fold / Z_folded;
	wH /= norm;
	wH_typ /= norm;

	// ---------- find Thouless time
	double eps = 5e-2;
	auto K_GOE = [](double t){
		return t < 1? 2 * t - t * log(1+2*t) : 2 - t * log( (2*t+1) / (2*t-1) );
	};

	double thouless_time = 0;
	double delta_min = 1e6;
	for(int i = 0; i < sff.size(); i++){
		double delta = abs(log10( sff(i) / K_GOE(times(i)) )) - eps;
		delta *= delta;
		if(delta < delta_min){
			delta_min = delta;
			thouless_time = times(i); 
		}
		if(times(i) >= 2.5) break;
	}

	double thouless_time_fold = 0;
	delta_min = 1e6;
	for(int i = 0; i < sff_fold.size(); i++){
		double delta = abs(log10( sff_fold(i) / K_GOE(times_fold(i)) )) - eps;
		delta *= delta;
		if(delta < delta_min){
			delta_min = delta;
			thouless_time_fold = times_fold(i); 
		}
		if(times_fold(i) >= 2.5 / wH) break;
	}
	save_to_file(dir + info + ".dat", 			 times, 	 sff, 	   wH, thouless_time, 	   r1, r2, dim, wH_typ);
	save_to_file(dir + "folded" + info + ".dat", times_fold, sff_fold, wH, thouless_time_fold, r1, r2, dim, wH_typ);
}

/// @brief Analyze all spectra (all realisations). Average spectral quantities and distirbutions (level spacing and gap ratio)
/// @tparam Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_dis<Hamiltonian>::analyze_spectra()
{

	std::string info = this->set_info();

	std::string dir_spacing 	= this->saving_dir + "LevelSpacingDistribution" + kPSep;
	std::string dir_DOS 		= this->saving_dir + "DensityOfStates" + kPSep;
	std::string dir_unfolding 	= this->saving_dir + "Unfolding" + kPSep;
	std::string dir_gap 		= this->saving_dir + "LevelSpacing" + kPSep + "distribution" + kPSep;
	createDirs(dir_DOS, dir_spacing, dir_unfolding, dir_gap);

	size_t N = this->ptr_to_model->get_hilbert_size();
	const u64 num = 0.6 * N;
	double wH 				= 0.0;
	double wH_typ 			= 0.0;
	double wH_typ_unfolded 	= 0.0;

	arma::vec energies_all, energies_unfolded_all;
	arma::vec spacing, spacing_unfolded, spacing_log, spacing_unfolded_log;
	arma::vec gap_ratio, gap_ratio_unfolded;
	//-------SET KERNEL
	int counter_realis = 0;
#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
	for(int realis = 0; realis < this->realisations; realis++)
	{
		std::string suffix = "_real=" + std::to_string(realis + this->jobid);
		arma::vec eigenvalues = this->get_eigenvalues(suffix, false);

		if(eigenvalues.empty()) continue;
		arma::vec energies_unfolded = statistics::unfolding(eigenvalues, std::min((unsigned)20, this->L));
		//------------------- Get 50% spectrum
		double E_av = arma::trace(eigenvalues) / double(N);
		auto i = min_element(begin(eigenvalues), end(eigenvalues), [=](double x, double y) {
			return abs(x - E_av) < abs(y - E_av);
			});
		u64 E_av_idx = i - eigenvalues.begin();
		const long E_min = E_av_idx - num / 2.;
		const long E_max = E_av_idx + num / 2. + 1;
		const long num_small = (N > 1000)? 500 : 100;
		arma::vec energies = this->ch? exctract_vector(eigenvalues, E_av_idx - num_small / 2., E_av_idx + num_small / 2.) :
										exctract_vector(eigenvalues, E_min, E_max);
		arma::vec energies_unfolded_cut = this->ch? exctract_vector(energies_unfolded, E_av_idx - num_small / 2., E_av_idx + num_small / 2.) :
														exctract_vector(energies_unfolded, E_min, E_max);
		
		//------------------- Level Spacings
		arma::vec level_spacings(energies.size() - 1, arma::fill::zeros);
		arma::vec level_spacings_unfolded(energies.size() - 1, arma::fill::zeros);
		for(int i = 0; i < energies.size() - 1; i++){
			const double delta 			= energies(i+1) 			 - energies(i);
			const double delta_unfolded = energies_unfolded_cut(i+1) - energies_unfolded_cut(i);

			wH 				 += delta / double(energies.size()-1);
			wH_typ  		 += std::log(abs(delta)) / double(energies.size()-1);
			wH_typ_unfolded  += std::log(abs(delta_unfolded)) / double(energies.size()-1);

			level_spacings(i) 			= delta;
			level_spacings_unfolded(i) 	= delta_unfolded;
		}
		arma::vec gap = statistics::eigenlevel_statistics_return(eigenvalues);
		arma::vec gap_unfolded = statistics::eigenlevel_statistics_return(energies_unfolded);

		//------------------- Combine realisations
	#pragma omp critical
		{
			energies_all = arma::join_cols(energies_all, energies);
			energies_unfolded_all = arma::join_cols(energies_unfolded_all, energies_unfolded_cut);
			
			spacing = arma::join_cols(spacing, level_spacings);
			spacing_log = arma::join_cols(spacing_log, arma::log10(level_spacings));
			spacing_unfolded = arma::join_cols(spacing_unfolded, level_spacings_unfolded);
			spacing_unfolded_log = arma::join_cols(spacing_unfolded_log, arma::log10(level_spacings_unfolded));
			
			gap_ratio = arma::join_cols(gap_ratio, gap);
			gap_ratio_unfolded = arma::join_cols(gap_ratio_unfolded, gap_unfolded);
			counter_realis++;
		}
	}

	//------CALCULATE FOR MODEL
	double norm = counter_realis;
	
	if(spacing.is_empty() || spacing_log.is_empty() || spacing_unfolded.is_empty() || spacing_unfolded_log.is_empty()
							 || energies_all.is_empty() || energies_unfolded_all.is_empty()){
		std::cout << "Empty arrays, eeeh?" << std::endl;
		return;
	}
	if(spacing.is_zero() || spacing_log.is_zero() || spacing_unfolded.is_zero() || spacing_unfolded_log.is_zero()
							 || energies_all.is_zero() || energies_unfolded_all.is_zero()) {
		std::cout << "Zero arrays, eeeh?" << std::endl;
		return;
	}

	wH /= norm;	wH_typ /= norm;	wH_typ_unfolded /= norm;
	std::string prefix = this->ch ? "_500_states" : "";

	const int num_hist = this->num_of_points;
	statistics::probability_distribution(dir_spacing, prefix + info, spacing, num_hist, std::exp(wH_typ_unfolded), wH, std::exp(wH_typ));
	statistics::probability_distribution(dir_spacing, prefix + "_log" + info, spacing_log, num_hist, wH_typ_unfolded, wH, wH_typ);
	statistics::probability_distribution(dir_spacing, prefix + "unfolded" + info, spacing_unfolded, num_hist, std::exp(wH_typ_unfolded), wH, std::exp(wH_typ));
	statistics::probability_distribution(dir_spacing, prefix + "unfolded_log" + info, spacing_unfolded_log, num_hist, wH_typ_unfolded, wH, wH_typ);
	
	statistics::probability_distribution(dir_DOS, prefix + info, energies_all, num_hist);
	statistics::probability_distribution(dir_DOS, prefix + "unfolded" + info, energies_unfolded_all, num_hist);

	statistics::probability_distribution(dir_gap, info, gap_ratio, num_hist);
	statistics::probability_distribution(dir_gap, info, gap_ratio_unfolded, num_hist);
}

/// @brief Calculate entanglement entropy in all eigenstates and all subsystem sizes using schmidt decomposition
/// @tparam Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_dis<Hamiltonian>::eigenstate_entanglement()
{
    clk::time_point start = std::chrono::system_clock::now();
	
	std::string dir = this->saving_dir + "Entropy" + kPSep + "Eigenstate" + kPSep;
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	#ifdef ARMA_USE_SUPERLU
        const int size = this->ch? 500 : dim;
    #else
        const int size = dim;
    #endif
	std::string info = this->set_info();
	std::string filename = info;// + "_subsize=" + std::to_string(LA);

	arma::mat entropies(size, this->L + 1, arma::fill::zeros);
	arma::mat single_site_entropy = entropies;

	arma::vec energies(size, arma::fill::zeros);
	int counter = 0;

	auto subsystem_sizes = arma::conv_to<arma::Col<int>>::from(arma::linspace(0, this->L, this->L + 1));
	std::cout << subsystem_sizes.t() << std::endl;

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

		std::cout << " - - - - - - finished diagonalization in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
		
		const arma::vec E = this->ptr_to_model->get_eigenvalues();

		arma::mat S(size, this->L + 1, arma::fill::zeros);
		arma::mat S_site = S;

		outer_threads = this->thread_number;
		omp_set_num_threads(1);
		std::cout << outer_threads << "\t\t" << omp_get_num_threads() << std::endl;
		
		for(auto& LA : subsystem_sizes)
		{
			auto start_LA = std::chrono::system_clock::now();
			std::vector<int> p(this->L);
			p[LA % this->L] = 0;
			for(int l = 0; l < this->L; l++)
				if(l != LA % this->L)
					p[l] = (l < (LA % this->L) )? l + 1 : l;
			std::cout << p << std::endl;
			auto permutation = op::_permutation_generator(this->L, p);
			arma::sp_mat P = arma::real(permutation.to_matrix( ULLPOW(this->L) ));

			std::cout << " - - - - - - set permutation matrix for LA = " << LA << " in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl;
			start_LA = std::chrono::system_clock::now();
		#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
			for(int n = 0; n < size; n++){
				
				arma::Col<element_type> state = this->ptr_to_model->get_eigenState(n);
				
				// somehow needs L-LA (computer sees bit representation the opposite way, i.e. take B subsystem)
				S(n, LA) 		= entropy::schmidt_decomposition(this->cast_state(state), this->L - LA, this->L);	// bipartite entanglement at subsystem size LA
				state = P * this->cast_state(state);
				S_site(n, LA) 	= entropy::schmidt_decomposition(state, this->L - 1, this->L);	// single site entanglement at site LA

				//double S2 = entropy::vonNeumann(state, LA, this->L);
				// #pragma omp critical
				// {
				// 	double x = S_site(n, LA) - (double)S(n, LA);
				// 	if(std::abs(x) > 1e-14)
				// 		printSeparated(std::cout, "\t", 16, true, LA, E(n), S_site(n, LA), S(n, LA), x);
				// }
			}
    		std::cout << " - - - - - - finished entropy size LA: " << LA << " in time:" << tim_s(start_LA) << " s - - - - - - " << std::endl; // simulation end
		}
		if(this->realisations > 1){
			std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
			createDirs(dir_realis);
			E.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "energies"));
			S.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "entropy", arma::hdf5_opts::append));
			S_site.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "single_site_entropy", arma::hdf5_opts::append));
		}
		entropies += S;
		single_site_entropy += S_site;
		energies += E;
		
		counter++;
    	omp_set_num_threads(this->thread_number);

		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
	}
    
	energies /= double(counter);
	entropies /= double(counter);
	single_site_entropy /= double(counter);

	filename += "_jobid=" + std::to_string(this->jobid);
    energies.save(arma::hdf5_name(dir + filename + ".hdf5", "energies"));
	entropies.save(arma::hdf5_name(dir + filename + ".hdf5", "entropy", arma::hdf5_opts::append));
	single_site_entropy.save(arma::hdf5_name(dir + filename + ".hdf5", "single_site_entropy", arma::hdf5_opts::append));
    std::cout << " - - - - - - FINISHED ENTROPY CALCULATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
}


/// @brief Calculate entanglement entropy in all eigenstates and all subsystem sizes using schmidt decomposition
/// @tparam Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_dis<Hamiltonian>::eigenstate_entanglement_degenerate()
{
    clk::time_point start = std::chrono::system_clock::now();
	
	std::string dir = this->saving_dir + "Entropy" + kPSep + "Degeneracy" + kPSep;
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	#ifdef ARMA_USE_SUPERLU
        const int size = this->ch? 500 : dim;
    #else
        const int size = dim;
    #endif
	std::string info = this->set_info();
	std::string filename = info;// + "_subsize=" + std::to_string(LA);

	arma::mat entropies(this->num_of_points, this->L + 1, arma::fill::zeros);
	arma::mat single_site_entropy = entropies;

	arma::vec energies(this->num_of_points, arma::fill::zeros);
	int counter = 0;

	auto subsystem_sizes = arma::conv_to<arma::Col<int>>::from(arma::linspace(0, this->L, this->L + 1));
	std::cout << subsystem_sizes.t() << std::endl;

	disorder<double> random_generator(this->seed);
	CUE random_matrix(this->seed);
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

		std::cout << " - - - - - - finished diagonalization in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
		
		const arma::vec Eall = this->ptr_to_model->get_eigenvalues();

		const long E_av_idx = this->ptr_to_model->E_av_idx;
		const long min_idx = dim > 2 * int(this->num_of_points)? E_av_idx - int(this->num_of_points / 2.0) : 0;
		const long max_idx = dim > 2 * int(this->num_of_points)? E_av_idx + int(this->num_of_points / 2.0) : dim - 1;
		std::cout << E_av_idx << "\t\t" << min_idx << "\t\t" << max_idx << std::endl;
		arma::mat S(this->num_of_points, subsystem_sizes.size(), arma::fill::zeros);
		arma::mat S_site = S;
		arma::vec E_av(this->num_of_points);

		outer_threads = this->thread_number;
		omp_set_num_threads(1);
		std::cout << outer_threads << "\t\t" << omp_get_num_threads() << std::endl;
		
		for(auto& LA : subsystem_sizes)
		{
			auto start_LA = std::chrono::system_clock::now();
			std::vector<int> p(this->L);
			p[LA % this->L] = 0;
			for(int l = 0; l < this->L; l++)
				if(l != LA % this->L)
					p[l] = (l < (LA % this->L) )? l + 1 : l;
			std::cout << p << std::endl;
			auto permutation = op::_permutation_generator(this->L, p);
			arma::sp_mat P = arma::real(permutation.to_matrix( size ));

			std::cout << " - - - - - - set permutation matrix for LA = " << LA << " in : " << tim_s(start_LA) << " s for realis = " << realis << " - - - - - - " << std::endl;
			start_LA = std::chrono::system_clock::now();
			for(int gamma_a = 1; gamma_a <= this->num_of_points; gamma_a++)
    		{
				double entropy = 0;
				arma::cx_mat Haar = random_matrix.generate_matrix(gamma_a);
				// realisations to draw states randomly
				double E = 0;
			#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
				for(u64 unused = 0; unused < this->mu; unused++)
				{
					arma::Col<int> indices = random_generator.create_random_vec<int>(gamma_a, min_idx, max_idx);
					int id = random_generator.random_uni<int>(0, gamma_a-1);

					arma::cx_vec state(dim, arma::fill::zeros);
					arma::cx_vec coeff = Haar.col(id) / std::sqrt(arma::cdot(Haar.col(id), Haar.col(id)));
					// arma::cx_vec coeff(gamma_a, arma::fill::ones);
					// coeff /= std::sqrt(arma::cdot(coeff, coeff));
					long idx = 0;
					for(auto& n : indices){
						E += this->ptr_to_model->get_eigenValue(n) - this->ptr_to_model->get_eigenValue(E_av_idx);
						auto eigenstate = this->cast_state(this->ptr_to_model->get_eigenState(n));
						state += coeff(idx++) * eigenstate;
					}
					state = state / std::sqrt(arma::cdot(state, state));
					// arma::Col<element_type> state = this->ptr_to_model->get_eigenState(n);
					
					// somehow needs L-LA (computer sees bit representation the opposite way, i.e. take B subsystem)
					#pragma omp critical
					{
						E_av(gamma_a - 1) += E / double(indices.size());
						S(gamma_a - 1, LA) 		+= entropy::schmidt_decomposition(state, this->L - LA, this->L);// bipartite entanglement at subsystem size LA
						state = P * state;
						S_site(gamma_a - 1, LA) += entropy::schmidt_decomposition(state, this->L - 1, this->L);	// single site entanglement at site LA
					}
				}
			}
    		std::cout << " - - - - - - finished entropy size LA: " << LA << " in time:" << tim_s(start_LA) << " s - - - - - - " << std::endl; // simulation end
		}
    	E_av = E_av / double(this->mu);
		S_site /= double(this->mu);
		S /= double(this->mu);
		if(this->realisations > 1){
			std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
			createDirs(dir_realis);
			Eall.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "energies"));
			S.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "entropy", arma::hdf5_opts::append));
			S_site.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "single_site_entropy", arma::hdf5_opts::append));
		}
		entropies += S;
		single_site_entropy += S_site;
		energies += E_av;
		
		counter++;
    	omp_set_num_threads(this->thread_number);

		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
	}
    
	energies /= double(counter);
	entropies /= double(counter);
	single_site_entropy /= double(counter);

	filename += "_jobid=" + std::to_string(this->jobid);
    energies.save(arma::hdf5_name(dir + filename + ".hdf5", "energies"));
	entropies.save(arma::hdf5_name(dir + filename + ".hdf5", "entropy", arma::hdf5_opts::append));
	single_site_entropy.save(arma::hdf5_name(dir + filename + ".hdf5", "single_site_entropy", arma::hdf5_opts::append));
    std::cout << " - - - - - - FINISHED ENTROPY CALCULATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
}

/// @brief Calculate diagonal matrix elements of local operators
/// @tparam Hamiltonian template parameter for current used model 
template <class Hamiltonian>
void user_interface_dis<Hamiltonian>::diagonal_matrix_elements()
{
	clk::time_point start = std::chrono::system_clock::now();
	
	std::string dir = this->saving_dir + "DiagMatElem" + kPSep;
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	#ifdef ARMA_USE_SUPERLU
        const int size = this->ch? 500 : dim;
    #else
        const int size = dim;
    #endif
	std::string info = this->set_info();
	std::string filename = info;

	arma::cx_vec sigX_mat(size, arma::fill::zeros);
	arma::cx_vec sigZ_mat(size, arma::fill::zeros);
	arma::vec energies(size, arma::fill::zeros);

	int Ll = this->L;
	auto kernel1 = [Ll](u64 state){ auto [val, num] = operators::sigma_x(state, Ll, Ll / 2 ); return std::make_pair(num, val); };
	auto SigmaX_op = op::generic_operator<>(this->L, std::move(kernel1), 1.0);
	auto SigmaX = SigmaX_op.to_matrix(dim);

	auto kernel2 = [Ll](u64 state){ auto [val, num] = operators::sigma_z(state, Ll, Ll / 2 ); return std::make_pair(num, val); };
	auto SigmaZ_op = op::generic_operator<>(this->L, std::move(kernel2), 1.0);
	auto SigmaZ = SigmaZ_op.to_matrix(dim);

	int counter = 0;
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

		std::cout << " - - - - - - finished diagonalization in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
		
		const arma::vec E = this->ptr_to_model->get_eigenvalues();

		arma::cx_vec sigX(size, arma::fill::zeros);
		arma::cx_vec sigZ(size, arma::fill::zeros);

		outer_threads = this->thread_number;
		omp_set_num_threads(1);
		std::cout << outer_threads << "\t\t" << omp_get_num_threads() << std::endl;
		
		start = std::chrono::system_clock::now();
	#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
		for(int n = 0; n < size; n++){
			arma::Col<element_type> state = this->cast_state(this->ptr_to_model->get_eigenState(n));
			sigX(n) = dot_prod(state, arma::cx_vec(SigmaX * state));
			sigZ(n) = dot_prod(state, arma::cx_vec(SigmaZ * state));
		}
    	std::cout << " - - - - - - finished matrix elemnts in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		if(this->realisations > 1){
			std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
			createDirs(dir_realis);
			E.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "energies"));
			sigX.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "sigmaX_L_2", arma::hdf5_opts::append));
			sigZ.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "sigmaZ_L_2", arma::hdf5_opts::append));
		}
		sigX_mat += sigX;
		sigZ_mat += sigZ;
		energies += E;
		
		counter++;
    	omp_set_num_threads(this->thread_number);
		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
	}
    
	energies /= double(counter);
	sigX_mat /= double(counter);
	sigZ_mat /= double(counter);

	filename += "_jobid=" + std::to_string(this->jobid);
    energies.save(arma::hdf5_name(dir + filename + ".hdf5", "energies"));
	sigX_mat.save(arma::hdf5_name(dir + filename + ".hdf5", "sigmaX_L_2", arma::hdf5_opts::append));
	sigZ_mat.save(arma::hdf5_name(dir + filename + ".hdf5", "sigmaZ_L_2", arma::hdf5_opts::append));
    std::cout << " - - - - - - FINISHED CALCULATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
}


template <class Hamiltonian>
void user_interface_dis<Hamiltonian>::multifractality(){

    clk::time_point start = std::chrono::system_clock::now();

	std::string subdir = "ParticipationRatio" + kPSep;
	std::string dir = this->saving_dir + "MultiFractality" + kPSep + subdir;
	createDirs(dir);

	std::string info = this->set_info();
	std::string filename = info;
	
	arma::vec participatio_ratio_100(this->num_of_points, arma::fill::zeros);
	arma::vec participatio_ratio_200(this->num_of_points, arma::fill::zeros);
	arma::vec participatio_ratio_500(this->num_of_points, arma::fill::zeros);
	arma::vec participatio_ratio_D_2(this->num_of_points, arma::fill::zeros);
	arma::vec participatio_ratio_D(this->num_of_points, arma::fill::zeros);
	int counter = 0;
	
	arma::vec q_ipr_list = arma::linspace(2.0 / double(this->num_of_points), 2.0, this->num_of_points);

	size_t dim = this->ptr_to_model->get_hilbert_size();
	#ifdef ARMA_USE_SUPERLU
        const int size = this->ch? 500 : dim;
    #else
        const int size = dim;
    #endif

	for(int realis = 0; realis < this->realisations; realis++)
	{
		if(realis > 0)
			this->ptr_to_model->generate_hamiltonian();
		
    	clk::time_point start_loop = std::chrono::system_clock::now();
		#ifdef ARMA_USE_SUPERLU
			if(this->ch){
				this->ptr_to_model->hamiltonian();
				this->ptr_to_model->diag_sparse(true);
			} else
				this->ptr_to_model->diagonalization();
		
		#else
			this->ptr_to_model->diagonalization();
		#endif

		const arma::vec E = this->ptr_to_model->get_eigenvalues();
		u64 E_av_idx = spectrals::get_mean_energy_index(E);

		arma::vec pr_100(this->num_of_points, arma::fill::zeros);
		arma::vec pr_200(this->num_of_points, arma::fill::zeros);
		arma::vec pr_500(this->num_of_points, arma::fill::zeros);
		arma::vec pr_D_2(this->num_of_points, arma::fill::zeros);
		arma::vec pr_D(this->num_of_points, arma::fill::zeros);
	#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
		for(int n = 0; n < size; n++){
			arma::Col<element_type> eigenstate = this->ptr_to_model->get_eigenState(n);
			eigenstate /= std::sqrt(arma::cdot(eigenstate, eigenstate));
			for(int iq = 0; iq < q_ipr_list.size(); iq++){
				double pr_tmp = statistics::participation_ratio(eigenstate, q_ipr_list(iq));

				if(n > E_av_idx - 50 && n < E_av_idx + 50) 				pr_100(iq) += pr_tmp;
				if(n > E_av_idx - 100 && n < E_av_idx + 100) 			pr_200(iq) += pr_tmp;
				if(n > E_av_idx - 250 && n < E_av_idx + 250) 			pr_500(iq) += pr_tmp;
				if(n > E_av_idx - size / 4 && n < E_av_idx + size / 4) 	pr_D_2(iq) += pr_tmp;
				pr_D(iq) += pr_tmp;
			}
		}
		pr_100 /= 100.0;
		pr_200 /= 200.0;
		pr_500 /= 500.0;
		pr_D_2 /= double(size / 2);
		pr_D /= double(size);
		if( this-> realisations > 1){
			std::string dir_realis = dir + "realisation=" + std::to_string(realis + this->jobid) + kPSep;
			createDirs(dir_realis);
			q_ipr_list.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "qs"));
			pr_100.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "100", arma::hdf5_opts::append));
			pr_200.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "200", arma::hdf5_opts::append));
			pr_500.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "500", arma::hdf5_opts::append));
			pr_D_2.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "D_2", arma::hdf5_opts::append));
			pr_D.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "D", arma::hdf5_opts::append));

		}
		participatio_ratio_100 += pr_100;
		participatio_ratio_200 += pr_200;
		participatio_ratio_500 += pr_500;
		participatio_ratio_D_2 += pr_D_2;
		participatio_ratio_D += pr_D;
		counter++;
		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start_loop) << " s - - - - - - " << std::endl; // simulation end
	};
	participatio_ratio_100 /= double(counter);
	participatio_ratio_200 /= double(counter);
	participatio_ratio_500 /= double(counter);
	participatio_ratio_D_2 /= double(counter);
	participatio_ratio_D /= double(counter);
	
	std::string suffix = "_jobid=" + std::to_string(this->jobid);

	q_ipr_list.save(arma::hdf5_name(dir + filename + ".hdf5", "qs"));
	participatio_ratio_100.save(arma::hdf5_name(dir + filename + suffix + ".hdf5", "100", arma::hdf5_opts::append));
	participatio_ratio_200.save(arma::hdf5_name(dir + filename + suffix + ".hdf5", "200", arma::hdf5_opts::append));
	participatio_ratio_500.save(arma::hdf5_name(dir + filename + suffix + ".hdf5", "500", arma::hdf5_opts::append));
	participatio_ratio_D_2.save(arma::hdf5_name(dir + filename + suffix + ".hdf5", "D_2", arma::hdf5_opts::append));
	participatio_ratio_D.save(arma::hdf5_name(dir + filename + suffix + ".hdf5", "D", arma::hdf5_opts::append));
	
    std::cout << " - - - - - - FINISHED ENTROPY CALCULATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
}

// -------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------- IMPLEMENTATION OF UI

/// @brief Sets model parameters from values in command line
/// @tparam Hamiltonian Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_dis<Hamiltonian>::parse_cmd_options(int argc, std::vector<std::string> argv)
{
	//<! set all general UI parameters
    user_interface<Hamiltonian>::parse_cmd_options(argc, argv);

	std::string choosen_option = "";																// current choosen option

	//---------- SIMULATION PARAMETERS
	
	// disorder
	choosen_option = "-jobid";
	this->set_option(this->jobid, argv, choosen_option, true);
	
	choosen_option = "-r";
	this->set_option(this->realisations, argv, choosen_option, true);
}


/// @brief Sets all UI parameters to default values
/// @tparam Hamiltonian Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_dis<Hamiltonian>::set_default(){
	
	user_interface<Hamiltonian>::set_default();
	
	this->realisations = 1;
	this->seed = static_cast<long unsigned int>(87178291199L);
	this->jobid = 0;
}


/// @brief Prints all general UI option values
/// @tparam Hamiltonian Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_dis<Hamiltonian>::printAllOptions() const {
	
	user_interface<Hamiltonian>::printAllOptions();
	std::cout << std::endl;
	std::cout << "realisations = " << this->realisations << std::endl
		  << "jobid = " << this->jobid << std::endl;

	std::cout << "---------------------------------------CHOSEN MODEL:" << std::endl;
}


