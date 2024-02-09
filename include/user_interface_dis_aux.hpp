#pragma once


/// @brief Diagonalize model hamiltonian and save spectrum to .hdf5 file
/// @tparam Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_dis<Hamiltonian>::diagonalize(){
	clk::time_point start = std::chrono::system_clock::now();
	std::string dir = this->saving_dir + "DIAGONALIZATION" + kPSep;
	createDirs(dir);

// #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
	for(int realis = 0; realis < this->realisations; realis++)	
	{
		int real = realis + this->jobid;
		std::string dir_re  = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
		createDirs(dir_re);
		std::string _suffix = "_real=" + std::to_string(real);

		std::string info = this->set_info({});
		std::cout << "\n\t\t--> finished creating model for " << info + _suffix << " - in time : " << tim_s(start) << "s" << std::endl;
		
		if(realis > 0)
			this->ptr_to_model->generate_hamiltonian();
		this->ptr_to_model->diagonalization(!this->ch);
		arma::vec eigenvalues = this->ptr_to_model->get_eigenvalues();
		
		std::cout << "\t\t	--> finished diagonalizing for " << info + _suffix << " - in time : " << tim_s(start) << "s" << std::endl;
		
		// std::cout << "Energies:\n";
		// std::cout << eigenvalues << std::endl;

		std::string name = dir_re + info + ".hdf5";
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
	arma::vec sff_raw(this->num_of_points, arma::fill::zeros);
	double Z = 0.0, Z_fold = 0.0, Z_raw = 0.0;
	double wH_mean = 0.0;
	double wH_typ  = 0.0;
	int counter = 0;
//#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
	for(int realis = 0; realis < this->realisations; realis++)
	{
		std::string prefix = "realisation=" + std::to_string(realis + this->jobid) + kPSep;
		// if(realis > 0)
		// 	this->ptr_to_model->generate_hamiltonian();
		arma::vec eigenvalues = this->get_eigenvalues(prefix);
		
		
		if(this->fun == 1) std::cout << "\t\t	--> finished loading eigenvalues for " << prefix + info << " - in time : " << tim_s(start) << "s" << std::endl;
		if(eigenvalues.empty()) continue;
		dim = eigenvalues.size();

		u64 E_av_idx = spectrals::get_mean_energy_index(eigenvalues);
		const u64 num = dim < 1000? 0.25 * dim : 0.5 * dim;
		const u64 num2 = dim < 1000? 100 : 500;

		// ------------------------------------- calculate level statistics
			double r1_tmp = 0, r2_tmp = 0, wH_mean_r = 0, wH_typ_r = 0;
			int count = 0;
			for(int i = 1; i < dim - 1; i++){
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
				if(i >= (E_av_idx - num / 2) && i < (E_av_idx + num / 2))
					r1_tmp += min / max;
				if(i >= (E_av_idx - num2 / 2) && i < (E_av_idx + num2 / 2))
					r2_tmp += min / max;
				count++;
			}
			if(this->fun == 1) std::cout << "\t\t	--> finished unfolding for " << prefix + info << " - in time : " << tim_s(start) << "s" << std::endl;
		
			wH_mean_r /= double(count);
			r1_tmp /= double(count);
			r2_tmp /= double(num2);
		// ------------------------------------- calculate sff
			// statistics::SFF<statistics::filters::raw> _sff_raw(1.0);
			// auto sff_r_raw = _sff_raw.calculate(eigenvalues, times_fold);
			// auto [Z_r_raw, _tmp1, _tmp2] = _sff_raw.get_norms();

			// statistics::SFF<statistics::filters::gauss> _sff_filter(0.3);
			// auto sff_r_folded = _sff_filter.calculate(eigenvalues, times_fold);
			// auto [Z_r_folded, _tmp3, _tmp4] = _sff_filter.get_norms();
			
			// eigenvalues = statistics::unfolding(eigenvalues);
			// auto sff_r = _sff_filter.calculate(eigenvalues, times);
			// auto [Z_r, _tmp5, _tmp6]= _sff_filter.get_norms();

			auto [sff_r_raw, Z_r_raw] = statistics::spectral_form_factor(eigenvalues, times_fold, this->beta, 1.5);

			auto [sff_r_folded, Z_r_folded] = statistics::spectral_form_factor(eigenvalues, times_fold, this->beta, 0.5);
			eigenvalues = statistics::unfolding(eigenvalues);

			auto [sff_r, Z_r] = statistics::spectral_form_factor(eigenvalues, times,this->beta, 0.5);
			#pragma omp critical
			{
				r1 += r1_tmp;
				r2 += r2_tmp;

				sff_raw += sff_r_raw;
				Z_raw += Z_r_raw;
				sff += sff_r;
				Z += Z_r;
				sff_fold += sff_r_folded;
				Z_fold += Z_r_folded;
				
				wH_mean += wH_mean_r;
				wH_typ  += wH_typ_r / double(count);
				counter++;
			}
			wH_typ_r = std::exp(wH_typ_r / double(count));
		if(this->fun == 1) std::cout << "\t\t	--> finished realisation for " << prefix + info << " - in time : " << tim_s(start) << "s" << std::endl;
		
		//--------- SAVE REALISATION TO FILE
		#if !defined(MY_MAC)
			std::string dir_re  = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
			createDirs(dir_re);

			times.save(arma::hdf5_name(dir_re + info + ".hdf5", "times"));
			sff_r.save(arma::hdf5_name(dir_re + info + ".hdf5", "sff", arma::hdf5_opts::append));
			arma::vec({Z_r}).save(arma::hdf5_name(dir_re + info + ".hdf5", "Z", arma::hdf5_opts::append));
			times_fold.save(arma::hdf5_name(dir_re + info + ".hdf5", "times_fold", arma::hdf5_opts::append));
			sff_r_folded.save(arma::hdf5_name(dir_re + info + ".hdf5", "sff_fold", arma::hdf5_opts::append));
			arma::vec({Z_r_folded}).save(arma::hdf5_name(dir_re + info + ".hdf5", "Z_fold", arma::hdf5_opts::append));
			arma::vec({r2_tmp}).save(arma::hdf5_name(dir_re + info + ".hdf5", "r_500", arma::hdf5_opts::append));
			arma::vec({r1_tmp}).save(arma::hdf5_name(dir_re + info + ".hdf5", "r_D_2", arma::hdf5_opts::append));
			arma::vec({wH_mean_r}).save(arma::hdf5_name(dir_re + info + ".hdf5", "wH", arma::hdf5_opts::append));
			arma::vec({wH_typ_r}).save(arma::hdf5_name(dir_re + info + ".hdf5", "wH_typ", arma::hdf5_opts::append));
			// save_to_file(dir_re + info + ".dat", 			times, 		sff_r, 		  Z_r, 		  r1_tmp, r2_tmp, wH_mean_r, wH_typ_r);
			// save_to_file(dir_re + "folded" + info + ".dat", times_fold, sff_r_folded, Z_r_folded, r1_tmp, r2_tmp, wH_mean_r, wH_typ_r);
		#endif
	}

	// --------------------------------------------------------------- AVERAGE CURRENT REALISATIONS
	if(sff.is_empty()) return;
	if(sff.is_zero()) return;
	if(this->jobid > 0) return;
	if(counter == 0) return;
	double norm = counter;
	r1 /= norm;
	r2 /= norm;
	sff = sff / Z;
	sff_raw = sff_raw / Z_raw;
	sff_fold = sff_fold / Z_fold;
	wH_mean /= norm;
	wH_typ /= norm;

	// #ifdef MY_MAC
	times.save(arma::hdf5_name(dir + info + ".hdf5", "times"));
	sff.save(arma::hdf5_name(dir + info + ".hdf5", "sff", arma::hdf5_opts::append));
	times_fold.save(arma::hdf5_name(dir + info + ".hdf5", "times_fold", arma::hdf5_opts::append));
	sff_fold.save(arma::hdf5_name(dir + info + ".hdf5", "sff_fold", arma::hdf5_opts::append));
	sff_raw.save(arma::hdf5_name(dir + info + ".hdf5", "sff_raw", arma::hdf5_opts::append));
	arma::vec({r2}).save(arma::hdf5_name(dir + info + ".hdf5", "r_500", arma::hdf5_opts::append));
	arma::vec({r1}).save(arma::hdf5_name(dir + info + ".hdf5", "r_D_2", arma::hdf5_opts::append));
	arma::uvec({dim}).save(arma::hdf5_name(dir + info + ".hdf5", "D", arma::hdf5_opts::append));
	arma::vec({two_pi / (wH)}).save(arma::hdf5_name(dir + info + ".hdf5", "tH", arma::hdf5_opts::append));
	arma::vec({two_pi / std::exp(wH_typ)}).save(arma::hdf5_name(dir + info + ".hdf5", "tH_typ", arma::hdf5_opts::append));
	// #endif
	// save_to_file(dir + info + ".dat", 			 times, 	 sff, 	   1.0 / wH_mean, thouless_time, 		   r1, r2, dim, 1.0 / wH_typ);
	// save_to_file(dir + "folded" + info + ".dat", times_fold, sff_fold, 1.0 / wH_mean, thouless_time / wH_mean, r1, r2, dim, 1.0 / wH_typ);
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
		// arma::sff_re, times_re, Z_re, r1_re, r2_re, wH_re, wH_typ
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
		arma::vec participation_entropy(size, arma::fill::zeros);

		outer_threads = this->thread_number;
		omp_set_num_threads(1);
		std::cout << outer_threads << "\t\t" << omp_get_num_threads() << std::endl;
		
		for(int LA_idx = 0; LA_idx < subsystem_sizes.size() - 1; LA_idx++)
		{	
			int LA = subsystem_sizes[LA_idx];
			auto start_LA = std::chrono::system_clock::now();
			std::vector<int> p(this->L);
			p[LA % this->L] = 0;
			for(int l = 0; l < this->L; l++)
				if(l != LA % this->L)
					p[l] = (l < (LA % this->L) )? l + 1 : l;
			std::cout << p << std::endl;
			auto permutation = QOps::_permutation_generator(this->L, p);
			arma::sp_mat P = arma::real(permutation.to_matrix( ULLPOW(this->L) ));

			std::cout << " - - - - - - set permutation matrix for LA = " << LA << " in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl;
			start_LA = std::chrono::system_clock::now();
		#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
			for(int n = 0; n < size; n++){
				
				arma::Col<element_type> state = arma::normalise(this->ptr_to_model->get_eigenState(n));
				
				// somehow needs L-LA (computer sees bit representation the opposite way, i.e. take B subsystem)
				S(n, LA_idx) = entropy::schmidt_decomposition(this->cast_state(state), this->L - LA, this->L);	// bipartite entanglement at subsystem size LA
				
				if(LA_idx == 0)
				{
				#pragma omp parallel for
					for(int k = 0; k < dim; k++){
						auto value = std::abs(state(k)) * std::abs(state(k));
    					participation_entropy(n) += (std::abs(value) > 0) ? -value * std::log(value) : 0;
					}
				}
				
				state = P * this->cast_state(state);
				if(LA < this->L)
					S_site(n, LA_idx) 	= entropy::schmidt_decomposition(state, this->L - 1, this->L);	// single site entanglement at site LA

			}
    		std::cout << " - - - - - - finished entropy size LA: " << LA << " in time:" << tim_s(start_LA) << " s - - - - - - " << std::endl; // simulation end
		}
		// if(this->realisations > 1)
		{
			std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
			createDirs(dir_realis);
			E.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "energies"));
			S.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "entropy", arma::hdf5_opts::append));
			S_site.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "single_site_entropy", arma::hdf5_opts::append));
			subsystem_sizes.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "subsystem sizes", arma::hdf5_opts::append));
			participation_entropy.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "von Neumann participation entropy", arma::hdf5_opts::append));
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
	#ifdef MY_MAC
		filename += "_jobid=" + std::to_string(this->jobid);
		energies.save(arma::hdf5_name(dir + filename + ".hdf5", "energies"));
		entropies.save(arma::hdf5_name(dir + filename + ".hdf5", "entropy", arma::hdf5_opts::append));
		single_site_entropy.save(arma::hdf5_name(dir + filename + ".hdf5", "single_site_entropy", arma::hdf5_opts::append));
	#endif
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
			auto permutation = QOps::_permutation_generator(this->L, p);
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
					arma::Col<int> indices = random_generator.create_random_vec<int, dist::uniform>(gamma_a, min_idx, max_idx);
					int id = random_generator.uniform_dist<int>(1, gamma_a-1);

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

	#ifdef MY_MAC
		filename += "_jobid=" + std::to_string(this->jobid);
		energies.save(arma::hdf5_name(dir + filename + ".hdf5", "energies"));
		entropies.save(arma::hdf5_name(dir + filename + ".hdf5", "entropy", arma::hdf5_opts::append));
		single_site_entropy.save(arma::hdf5_name(dir + filename + ".hdf5", "single_site_entropy", arma::hdf5_opts::append));
	#endif
    std::cout << " - - - - - - FINISHED ENTROPY CALCULATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
}

/// @brief Calculate entanglement evolution for random initial state and all subsystem sizes using schmidt decomposition
/// @tparam Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_dis<Hamiltonian>::entanglement_evolution()
{
    clk::time_point start = std::chrono::system_clock::now();
	
	std::string dir = this->saving_dir + "Entropy" + kPSep + "Evolution" + kPSep;
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	#ifdef ARMA_USE_SUPERLU
        const int size = this->ch? 500 : dim;
    #else
        const int size = dim;
    #endif
	std::string info = this->set_info();
	std::string filename = info;// + "_subsize=" + std::to_string(LA);

	int counter = 0;

	auto subsystem_sizes = arma::conv_to<arma::Col<int>>::from(arma::linspace(0, this->L, this->L + 1));
	std::cout << subsystem_sizes.t() << std::endl;

	const double tH = (0.341345 * dim) / std::sqrt(this->L);
	int time_end = (int)std::ceil(std::log10(5 * tH));
    arma::vec times = dim > 8e4? arma::regspace(this->dt, this->dt, this->tend)
									: arma::linspace(0.1, 10 * tH, int(20 * tH / 0.1) + 1);
								//  : arma::logspace(-2, time_end, this->num_of_points);

	arma::mat entropies(times.size(), this->L + 1, arma::fill::zeros);
	arma::mat entropies_squared(times.size(), this->L + 1, arma::fill::zeros);
	
	arma::mat single_site_entropy = entropies;
	arma::mat single_site_entropy_squared = entropies;

	std::vector<arma::sp_mat> permutation_matrices;
	for(int LA_idx = 0; LA_idx < subsystem_sizes.size() - 1; LA_idx++)
	{	
		int LA = subsystem_sizes[LA_idx];
		auto start_LA = std::chrono::system_clock::now();
		std::vector<int> p(this->L);
		p[LA % this->L] = 0;
		for(int l = 0; l < this->L; l++)
			if(l != LA % this->L)
				p[l] = (l < (LA % this->L) )? l + 1 : l;
		auto permutation = QOps::_permutation_generator(this->L, p);
		arma::sp_mat P = arma::real(permutation.to_matrix( ULLPOW(this->L) ));
		permutation_matrices.push_back(P);
		std::cout << " - - - - - - set permutation matrix for LA = " << LA << " in : " << tim_s(start_LA) << " s - - - - - - " << std::endl;
	}
// #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
	for(int realis = 0; realis < this->realisations; realis++)
	{
		if(realis > 0)
			this->ptr_to_model->generate_hamiltonian();
		
		auto start_re = std::chrono::system_clock::now();
		start = std::chrono::system_clock::now();
		
		arma::cx_vec initial_state = this->random_product_state();
		_assert_(this->ptr_to_model->get_hilbert_size() == initial_state.size(), 
					"Hamiltonian is in reduced basis, random state is in full. Not implemented otherwise"
					);
		std::cout << " - - - - - - finished generating state in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();

		arma::mat S(times.size(), this->L + 1, arma::fill::zeros);
		arma::mat S_site = S;

        if(dim < 8e4)
		{
			this->ptr_to_model->diagonalization();
			std::cout << " - - - - - - finished diagonalization in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
			
			const arma::vec E = this->ptr_to_model->get_eigenvalues();

			outer_threads = this->thread_number;
			omp_set_num_threads(1);
			std::cout << outer_threads << "\t\t" << omp_get_num_threads() << std::endl;

			start = std::chrono::system_clock::now();
		#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
			for(int n = 0; n < times.size(); n++)
			{
				double time = times(n);
				arma::cx_vec state = arma::cx_vec(initial_state.size(), arma::fill::zeros);
				for(int k = 0; k < dim; k++){
					arma::cx_vec psi_k(dim, arma::fill::zeros);
					psi_k.set_real(this->ptr_to_model->get_eigenState(k));
					auto overlap = arma::cdot(psi_k, initial_state);
					state += std::exp(-1i * E(k) * time) * overlap * this->ptr_to_model->get_eigenState(k);
				}
				for(int LA_idx = 0; LA_idx < subsystem_sizes.size(); LA_idx++)
				{	
					int LA = subsystem_sizes[LA_idx];
					// somehow needs L-LA (computer sees bit representation the opposite way, i.e. take B subsystem)
					S(n, LA_idx) = entropy::schmidt_decomposition(state, this->L - LA, this->L);	// bipartite entanglement at subsystem size LA
					if(LA < this->L){
						arma::cx_vec permuted_state = permutation_matrices[LA_idx] * state;
						S_site(n, LA_idx) 	= entropy::schmidt_decomposition(permuted_state, this->L - 1, this->L);	// single site entanglement at site LA
					}
				}
			}
			std::cout << " - - - - - - finished entropy for all times in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
			omp_set_num_threads(this->thread_number);
		} 
		else 
		{
			arma::sp_cx_mat H(dim, dim);
			H.set_real(this->ptr_to_model->get_hamiltonian());
			
			arma::cx_vec _state_ = initial_state;
			for(int n = 0; n < times.size(); n++)
			{
				lanczos::Lanczos<cpx> lancz(H, this->l_steps, -1, 0, this->seed, true, false, _state_);
				lancz.time_evolution_step(_state_, this->dt);

				for(int LA_idx = 0; LA_idx < subsystem_sizes.size(); LA_idx++)
				{
					int LA = subsystem_sizes[LA_idx];
					// somehow needs L-LA (computer sees bit representation the opposite way, i.e. take B subsystem)
					S(n, LA_idx) = entropy::schmidt_decomposition(_state_, this->L - LA, this->L);	// bipartite entanglement at subsystem size LA
					if(LA < this->L){
						arma::cx_vec permuted_state = permutation_matrices[LA_idx] * _state_;
						S_site(n, LA_idx) = entropy::schmidt_decomposition(permuted_state, this->L - 1, this->L);	// single site entanglement at site LA
					}
				}
			}
		}
		// if(this->realisations > 1)
		{
			std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
			createDirs(dir_realis);
    		times.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "times"));
			S.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "entropy", arma::hdf5_opts::append));
			S_site.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "single_site_entropy", arma::hdf5_opts::append));
		}
		entropies += S;
		entropies_squared += arma::square(S);
		single_site_entropy += S_site;
		single_site_entropy_squared += arma::square(S_site);
		
		counter++;
		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start_re) << " s - - - - - - " << std::endl; // simulation end
	}
    
	entropies /= double(counter);
	single_site_entropy /= double(counter);
	entropies_squared /= double(counter);
	// single_site_entropy_squared = single_site_entropy_squared / double(counter) - 

	filename += "_jobid=" + std::to_string(this->jobid);
    times.save(arma::hdf5_name(dir + filename + ".hdf5", "times"));
	entropies.save(arma::hdf5_name(dir + filename + ".hdf5", "entropy", arma::hdf5_opts::append));
	single_site_entropy.save(arma::hdf5_name(dir + filename + ".hdf5", "single_site_entropy", arma::hdf5_opts::append));
    std::cout << " - - - - - - FINISHED ENTROPY CALCULATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
}


/// @brief Calculate survival probability
/// @tparam Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_dis<Hamiltonian>::survival_probability()
{
	clk::time_point start_tot = std::chrono::system_clock::now();
	
	std::string dir = this->saving_dir + "SurvivalProbability" + kPSep;
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info();
	std::string filename = info;// + "_subsize=" + std::to_string(LA);

	const double tH = (0.341345 * dim) / std::sqrt(this->L);
	int time_end = (int)std::ceil(std::log10(7.5 * tH));
	time_end = (time_end / std::log10(tH) < 3) ? time_end + 1 : time_end;
    arma::vec times = arma::join_cols(arma::vec({0}), arma::logspace(-2, time_end, this->num_of_points - 1));

	#ifdef MY_MAC
		arma::vec survival(times.size(), arma::fill::zeros);
		arma::vec survival_gauss(times.size(), arma::fill::zeros);
		arma::vec survival_projected(times.size(), arma::fill::zeros);

		arma::vec participation(dim, arma::fill::zeros);
		arma::vec energies(dim, arma::fill::zeros);
	#endif
	double wH_mean = 0, wH_typ = 0;
	int counter = 0;
// #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
	for(int realis = 0; realis < this->realisations; realis++)
	{
		if(realis > 0)
			this->ptr_to_model->generate_hamiltonian();
		clk::time_point start_re = std::chrono::system_clock::now();
		clk::time_point start = std::chrono::system_clock::now();
    
        this->ptr_to_model->diagonalization();

		std::cout << " - - - - - - finished diagonalization in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();

		const arma::vec E = this->ptr_to_model->get_eigenvalues();
		
		u64 E_av_idx = spectrals::get_mean_energy_index(E);
		const u64 num = dim < 1000? 0.25 * dim : 0.5 * dim;

		// ------------------------------------- calculate level spacing (average and typical)
		double wH_mean_r = 0, wH_typ_r = 0;
		int count = 0;
		for(int i = 0; i < dim - 1; i++){
			const double gap = E(i + 1) - E(i);
			wH_mean_r += gap;
			wH_typ_r += std::log(gap);
			count++;
		}
		wH_mean_r /= double(count);
		wH_typ_r /= double(count);
		
		start = std::chrono::system_clock::now();
		
        const double mean = arma::mean(E);
        const double stddev = arma::stddev(E);
        const double denom = 2.0 * 0.3 * 0.3 * stddev * stddev;

		arma::mat A(dim, dim, arma::fill::zeros);
		arma::mat A_gauss(dim, dim, arma::fill::zeros);
		arma::mat A_projected(dim, dim, arma::fill::zeros);

		arma::cx_mat B(times.size(), dim, arma::fill::zeros);
		arma::vec pr(dim, arma::fill::zeros);
		
		//<! Calculate intermediate elements
	#pragma omp parallel for
		for(long alfa = 0; alfa < dim; alfa++)
		{
			const double filter = std::exp( -(E(alfa) - mean) * (E(alfa) - mean) / denom );
			for(long k = 0; k < dim; k++){
				double coeff = std::abs(this->ptr_to_model->get_eigenStateCoeff(alfa, k));
				coeff *= coeff;
				
				// participation ratio
				pr(alfa) += coeff * coeff;
				
				//diagonal
				A(alfa, k) = coeff;
				A_gauss(alfa, k) = coeff * filter * filter;
				if( (alfa >= this->ptr_to_model->E_av_idx - 0.1 * dim) && (alfa < this->ptr_to_model->E_av_idx + 0.1 * dim) )
					A_projected(alfa, k) = coeff;
			}
			for(long t_idx = 0; t_idx < times.size(); t_idx++)
			{
				double time = times(t_idx);
				B(t_idx, alfa) = std::exp(-1i * time * E(alfa));
			}
		}
		arma::rowvec _norm_gauss 	 = arma::sum(A_gauss, 0);
		arma::rowvec _norm_projected = arma::sum(A_projected, 0);
	#pragma omp parallel for
		for(long k = 0; k < dim; k++)
		{
			A_gauss.col(k) = A_gauss.col(k) / _norm_gauss(k);
			A_projected.col(k) = A_projected.col(k) / _norm_projected(k);
		}
		std::cout << " - - - - - - finished calculating matrices in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();
		
		//<! Calculate survival probability
		arma::vec _surv_prob = arma::sum( arma::square(arma::abs( B * A )) , 1) / double(dim);
		arma::vec _surv_prob_gauss = arma::sum( arma::square(arma::abs( B * A_gauss )) , 1) / double(dim);
		arma::vec _surv_prob_projected = arma::sum( arma::square(arma::abs( B * A_projected )) , 1) / double(dim);
		
		// _surv_prob_gauss /= _surv_prob_gauss(0);
		// _surv_prob_projected /= _surv_prob_projected(0);

		std::cout << " - - - - - - finished survival probability in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
		{
			std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
			createDirs(dir_realis);
			times.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "times"));
			_surv_prob.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "survival", arma::hdf5_opts::append));
			_surv_prob_gauss.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "survival gaussian", arma::hdf5_opts::append));
			_surv_prob_projected.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "survival projected", arma::hdf5_opts::append));
			E.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "energies", arma::hdf5_opts::append));
			pr.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "pr", arma::hdf5_opts::append));
			arma::vec({ two_pi / wH_mean_r }).save(arma::hdf5_name(dir_realis + filename + ".hdf5", "tH", arma::hdf5_opts::append));
			arma::vec({ two_pi / std::exp(wH_typ_r) }).save(arma::hdf5_name(dir_realis + filename + ".hdf5", "tH_typ", arma::hdf5_opts::append));
		}

		#ifdef MY_MAC
			survival += _surv_prob;
			survival_gauss += _surv_prob_gauss;
			survival_projected += _surv_prob_projected;
			participation += pr;
			energies += E;
			wH_mean += wH_mean_r;
			wH_typ += wH_typ_r;

			counter++;
		#endif

		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start_re) << " s - - - - - - \n\n" << std::endl; // simulation end
	}
    
	#ifdef MY_MAC
		survival /= double(counter);
		survival_gauss /= double(counter);
		survival_projected /= double(counter);
		participation /= double(counter);
		energies /= double(counter);
		wH_mean /= double(counter);
		wH_typ /= double(counter);

		filename += "_jobid=" + std::to_string(this->jobid);
		times.save(arma::hdf5_name(dir + filename + ".hdf5", "times"));
		survival.save(arma::hdf5_name(dir + filename + ".hdf5", "survival", arma::hdf5_opts::append));
		survival_gauss.save(arma::hdf5_name(dir + filename + ".hdf5", "survival gaussian", arma::hdf5_opts::append));
		survival_projected.save(arma::hdf5_name(dir + filename + ".hdf5", "survival projected", arma::hdf5_opts::append));
		energies.save(arma::hdf5_name(dir + filename + ".hdf5", "energies", arma::hdf5_opts::append));
		participation.save(arma::hdf5_name(dir + filename + ".hdf5", "pr", arma::hdf5_opts::append));
		arma::vec({ two_pi / wH_mean }).save(arma::hdf5_name(dir + filename + ".hdf5", "tH", arma::hdf5_opts::append));
		arma::vec({ two_pi / std::exp(wH_typ) }).save(arma::hdf5_name(dir + filename + ".hdf5", "tH_typ", arma::hdf5_opts::append));
	#endif
    std::cout << " - - - - - - FINISHED SURVIVAL CALCULATION IN : " << tim_s(start_tot) << " seconds - - - - - - " << std::endl; // simulation end
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
	auto SigmaX_op = QOps::generic_operator<>(this->L, std::move(kernel1), 1.0);
	auto SigmaX = SigmaX_op.to_matrix(dim);

	auto kernel2 = [Ll](u64 state){ auto [val, num] = operators::sigma_z(state, Ll, Ll / 2 ); return std::make_pair(num, val); };
	auto SigmaZ_op = QOps::generic_operator<>(this->L, std::move(kernel2), 1.0);
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

/// @brief Calculate matrix elements of local operators
/// @tparam Hamiltonian template parameter for current used model 
template <class Hamiltonian>
void user_interface_dis<Hamiltonian>::matrix_elements()
{
	std::string dir = this->saving_dir + "MatrixElements" + kPSep;
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info();

	arma::vec sites = arma::linspace(0, this->L-1, this->L);
	// arma::vec sites = arma::vec({3, this->L / 2, this->L - 1});
	arma::vec agp_norm(sites.size(), arma::fill::zeros);
	arma::vec typ_susc(sites.size(), arma::fill::zeros);
	arma::vec susc(sites.size(), arma::fill::zeros);

	std::vector<arma::sp_mat> Sz_ops;
	int Ll = this->L;
	for(int site : sites){
		auto kernel = [Ll, site](u64 state){ 
			auto [val, num] = operators::sigma_z(state, Ll, site ); 
			return std::make_pair(num, val); 
			};
		auto _operator = QOps::generic_operator<>(this->L, std::move(kernel), 1.0);
		Sz_ops.push_back(arma::real(_operator.to_matrix(dim)));
	}

	int counter = 0;
// #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
	for(int realis = 0; realis < this->realisations; realis++)
	{
		clk::time_point start_re = std::chrono::system_clock::now();
		if(realis > 0)
			this->ptr_to_model->generate_hamiltonian();
		
		clk::time_point start = std::chrono::system_clock::now();
    	this->ptr_to_model->diagonalization();

		std::cout << " - - - - - - finished diagonalization in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();
		
		const arma::vec E = this->ptr_to_model->get_eigenvalues();
		const auto& V = this->ptr_to_model->get_eigenvectors();
		
		arma::vec agp_norm_r(sites.size(), arma::fill::zeros);
		arma::vec susc_r(sites.size(), arma::fill::zeros);
		arma::vec typ_susc_r(sites.size(), arma::fill::zeros);
		
		for(int i = 0; i < sites.size(); i++)
		{
			start = std::chrono::system_clock::now();
			arma::Mat<element_type> mat_elem = V * Sz_ops[i] * V.t();
			auto [_agp, _typ_susc, _susc, tmp] = adiabatics::gauge_potential(mat_elem, E, this->L);
			agp_norm_r(i) = _agp;
			typ_susc_r(i) = _typ_susc;
			susc_r(i) = _susc;
    		std::cout << " - - - - - - finished matrix elements for site i = " << sites(i) << "in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		}
		#ifndef MY_MAC
		{
			std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
			createDirs(dir_realis);
			sites.save(arma::hdf5_name(dir_realis + info + ".hdf5", "sites"));
			agp_norm_r.save(arma::hdf5_name(dir_realis + info + ".hdf5", "agp norm", arma::hdf5_opts::append));
			typ_susc_r.save(arma::hdf5_name(dir_realis + info + ".hdf5", "typical susceptibility", arma::hdf5_opts::append));
			susc_r.save(arma::hdf5_name(dir_realis + info + ".hdf5", "susceptibility", arma::hdf5_opts::append));
			// sigX.save(arma::hdf5_name(dir_realis + info + ".hdf5", "sigmaX_L_2", arma::hdf5_opts::append));
			// sigZ.save(arma::hdf5_name(dir_realis + info + ".hdf5", "sigmaZ_L_2", arma::hdf5_opts::append));
		}
		#endif
		
		agp_norm += agp_norm_r;
		typ_susc += typ_susc_r;
		susc += susc_r;
		counter++;
		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start_re) << " s - - - - - - " << std::endl; // simulation end
	}
	if(counter == 0) return;
	
	#ifdef MY_MAC
		agp_norm /= double(counter);
		typ_susc /= double(counter);
		susc /= double(counter);
		sites.save(arma::hdf5_name(dir + info + ".hdf5", "sites"));
		agp_norm.save(arma::hdf5_name(dir + info + ".hdf5", "agp norm", arma::hdf5_opts::append));
		typ_susc.save(arma::hdf5_name(dir + info + ".hdf5", "typical susceptibility", arma::hdf5_opts::append));
		susc.save(arma::hdf5_name(dir + info + ".hdf5", "susceptibility", arma::hdf5_opts::append));
	#endif
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


/// @brief Checkl accuracy of krylov expansion fir different dt and lanczos steps
/// @tparam Hamiltonian template parameter for current used model 
template <class Hamiltonian>
void user_interface_dis<Hamiltonian>::check_krylov_evolution()
{
	clk::time_point start = std::chrono::system_clock::now();

	size_t dim = this->ptr_to_model->get_hilbert_size();
	arma::sp_cx_mat H(dim, dim);
	H.set_real(this->ptr_to_model->get_hamiltonian());
	this->ptr_to_model->diagonalization();
	std::cout << " - - - - - - finished diagonalization in : " << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
	
	const arma::vec E = this->ptr_to_model->get_eigenvalues();
	
	int LA = this->L / 2;
	arma::cx_vec initial_state = this->random_product_state();

	printSeparated(std::cout, "\t", 16, true, "M", "dt", "time [s]", "errors (1e-14) [%]", "errors (1e-12) [%]", "errors (1e-10) [%]");
	for(double _dt : {0.05, 0.1, 0.2, 0.5})
	{
		this->dt = _dt;
		auto times = arma::regspace(this->dt, this->dt, this->tend);
		for(int M : {5, 10, 12, 15, 20, 25})
		{
			this->l_steps = M;
			clk::time_point starter = std::chrono::system_clock::now();

			arma::cx_vec lancz0s_state = initial_state;
			int counter1 = 0, counter2 = 0, counter3 = 0;
			for(int n = 0; n < times.size(); n++)
			{
				double time = times(n);
				arma::cx_vec state = arma::cx_vec(initial_state.size(), arma::fill::zeros);
				for(int k = 0; k < dim; k++){
					auto overlap = dot_prod(this->ptr_to_model->get_eigenState(k), initial_state);
					state += std::exp(-1i * E(k) * time) * overlap * this->ptr_to_model->get_eigenState(k);
				}
				
				// somehow needs L-LA (computer sees bit representation the opposite way, i.e. take B subsystem)
				double S_ED = entropy::schmidt_decomposition(state, this->L - LA, this->L);	// bipartite entanglement at subsystem size LA
				
				lanczos::Lanczos<cpx> lancz(H, this->l_steps, -1, 0, this->seed, true, false, lancz0s_state);
				lancz.time_evolution_step(lancz0s_state, this->dt);
				double Ss = entropy::schmidt_decomposition(lancz0s_state, this->L - LA, this->L);
				
				if(std::abs(S_ED - Ss) > 1e-13) counter1++;
				if(std::abs(S_ED - Ss) > 1e-12) counter2++;
				if(std::abs(S_ED - Ss) > 1e-10) counter3++;
			}

			printSeparated(std::cout, "\t", 16, true, this->l_steps, this->dt, tim_s(starter),
								double(counter1) / times.size() * 100.0, double(counter2) / times.size() * 100.0, double(counter3) / times.size() * 100.0);
		}
	}
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


