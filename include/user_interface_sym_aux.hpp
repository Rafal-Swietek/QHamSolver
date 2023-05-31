#pragma once

/// @brief Diagonalize model hamiltonian and save spectrum to .hdf5 file
/// @tparam Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_sym<Hamiltonian>::diagonalize(){
	clk::time_point start = std::chrono::system_clock::now();
	std::string dir = this->saving_dir + "DIAGONALIZATION" + kPSep;
	createDirs(dir);

    std::string info = this->set_info({});
    std::cout << "\n\t\t--> finished creating model for " << info << " - in time : " << tim_s(start) << "s" << std::endl;
    
    this->ptr_to_model->diagonalization(!this->ch);
    arma::vec eigenvalues = this->ptr_to_model->get_eigenvalues();
    
    std::cout << "\t\t	--> finished diagonalizing for " << info << " - in time : " << tim_s(start) << "s" << std::endl;
    
    //std::cout << eigenvalues.t() << std::endl;

    std::string name = dir + info + ".hdf5";
    eigenvalues.save(arma::hdf5_name(name, "eigenvalues", arma::hdf5_opts::append));
    std::cout << "\t\t	--> finished saving eigenvalues for " << info << " - in time : " << tim_s(start) << "s" << std::endl;
    if(this->ch){
        auto H = this->ptr_to_model->get_dense_hamiltonian();
        H.save(arma::hdf5_name(name, "hamiltonian", arma::hdf5_opts::append));
        std::cout << "\t\t	--> finished saving Hamiltonian for " << info << " - in time : " << tim_s(start) << "s" << std::endl;

        auto V = this->ptr_to_model->get_eigenvectors();
        V.save(arma::hdf5_name(name, "eigenvectors", arma::hdf5_opts::append));
        std::cout << "\t\t	--> finished saving eigenvectors for " << info << " - in time : " << tim_s(start) << "s" << std::endl;
    }
}


/// @brief Analyze all spectra (all realisations). Average spectral quantities and distirbutions (level spacing and gap ratio)
/// @tparam Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_sym<Hamiltonian>::analyze_spectra()
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
	
    arma::vec eigenvalues = this->get_eigenvalues("", false);

    if(eigenvalues.empty()) return;

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

        wH 				 += delta;
        wH_typ  		 += std::log(abs(delta));
        wH_typ_unfolded  += std::log(abs(delta_unfolded));

        level_spacings(i) 			= delta;
        level_spacings_unfolded(i) 	= delta_unfolded;
    }
    wH              /= double(energies.size()-1);
    wH_typ          /= double(energies.size()-1);
    wH_typ_unfolded /= double(energies.size()-1);

    arma::vec gap_ratio = statistics::eigenlevel_statistics_return(eigenvalues);
    arma::vec gap_ratio_unfolded = statistics::eigenlevel_statistics_return(energies_unfolded);
	
	std::string prefix = this->ch ? "_500_states" : "";

	const int num_hist = this->num_of_points;
	statistics::probability_distribution(dir_spacing, prefix +                  info, level_spacings,                       num_hist, std::exp(wH_typ_unfolded), wH, std::exp(wH_typ));
	statistics::probability_distribution(dir_spacing, prefix + "_log" +         info, arma::log10(level_spacings),          num_hist, wH_typ_unfolded, wH, wH_typ);
	statistics::probability_distribution(dir_spacing, prefix + "unfolded" +     info, level_spacings_unfolded,              num_hist, std::exp(wH_typ_unfolded), wH, std::exp(wH_typ));
	statistics::probability_distribution(dir_spacing, prefix + "unfolded_log" + info, arma::log10(level_spacings_unfolded), num_hist, wH_typ_unfolded, wH, wH_typ);
	
	statistics::probability_distribution(dir_DOS, prefix + info, eigenvalues, num_hist);
	statistics::probability_distribution(dir_DOS, prefix + "unfolded" + info, energies_unfolded, num_hist);

	statistics::probability_distribution(dir_gap, info, gap_ratio, num_hist);
	statistics::probability_distribution(dir_gap, info, gap_ratio_unfolded, num_hist);
}

/// @brief Calculate entanglement entropy in all eigenstates and all subsystem sizes using schmidt decomposition
/// @tparam Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_sym<Hamiltonian>::eigenstate_entanglement()
{
   clk::time_point start = std::chrono::system_clock::now();
	
	std::string dir = this->saving_dir + "Entropy" + kPSep + "Eigenstate" + kPSep;
	createDirs(dir);
	
    int LA = this->site;
	size_t dim = this->ptr_to_model->get_hilbert_size();
	
	std::string info = this->set_info();
	std::string filename = info;// + "_subsize=" + std::to_string(LA);

    #ifdef ARMA_USE_SUPERLU
        const int size = this->ch? 500 : dim;
        if(this->ch){
            this->ptr_to_model->hamiltonian();
            this->ptr_to_model->diag_sparse(true);
        } else
            this->ptr_to_model->diagonalization();
    
    #else
        const int size = dim;
        this->ptr_to_model->diagonalization();
    #endif

    std::cout << " - - - - - - FINISHED DIAGONALIZATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
    
    start = std::chrono::system_clock::now();
    const arma::vec E = this->ptr_to_model->get_eigenvalues();
    
    const auto U = this->ptr_to_model->get_model_ref().get_hilbert_space().symmetry_rotation();
    
    std::cout << " - - - - - - FINISHED CREATING SYMMETRY TRANSFORMATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
    start = std::chrono::system_clock::now();

    auto subsystem_sizes = arma::conv_to<arma::Col<int>>::from(arma::linspace(0, this->L / 2, this->L / 2 + 1));
    std::cout << subsystem_sizes.t() << std::endl;

    arma::mat S(size, subsystem_sizes.size(), arma::fill::zeros);
    int th_num = this->thread_number;
    omp_set_num_threads(1);
    std::cout << th_num << "\t\t" << omp_get_num_threads() << std::endl;
    
        // auto start_LA = std::chrono::system_clock::now();
#pragma omp parallel for num_threads(th_num) schedule(dynamic)
    for(int n = 0; n < size; n++){
        auto eigenstate = this->ptr_to_model->get_eigenState(n);
        arma::Col<element_type> state = U * eigenstate;
        
        for(auto& LA : subsystem_sizes){
            S(n, LA) = entropy::schmidt_decomposition(state, LA, this->L);
            // auto S2 = entropy::vonNeumann(state, LA, this->L);
            // #pragma omp critical
            // {
            //     double x = S2 - S(n, LA);
            //     //if(std::abs(x) > 1e-14)
            //     printSeparated(std::cout, "\t", 16, true, LA, E(n), S2, S(n, LA), x);
            // }
        }
    }
    std::cout << " - - - - - - FINISHED ENTROPY CALCULATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
    
    omp_set_num_threads(this->thread_number);
    th_num = 1;
    
    E.save(arma::hdf5_name(dir + filename + ".hdf5", "energies"));
	S.save(arma::hdf5_name(dir + filename + ".hdf5", "entropy", arma::hdf5_opts::append));
        
    // #ifdef ARMA_USE_SUPERLU
    //     arma::Mat<element_type> V;
    //     if(this->ch) V = this->ptr_to_model->get_eigenvectors();
    //     else         V = this->ptr_to_model->get_eigenvectors().submat(0, this->ptr_to_model->E_av_idx - 50, dim - 1, this->ptr_to_model->E_av_idx + 50);
    // #else
    //     arma::Mat<element_type> V = this->ptr_to_model->get_eigenvectors().submat(0, this->ptr_to_model->E_av_idx - 50, dim - 1, this->ptr_to_model->E_av_idx + 50);
    // #endif
    // V.save(arma::hdf5_name(dir + filename + ".hdf5", "eigenvectors",arma::hdf5_opts::append));
}
