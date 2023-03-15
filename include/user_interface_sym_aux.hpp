#pragma once


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
   
}
