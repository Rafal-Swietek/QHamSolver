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
    
    this->ptr_to_model->diagonalization(this->ch);
    arma::vec eigenvalues = this->ptr_to_model->get_eigenvalues();
    // std::cout << eigenvalues.t() << std::endl;
    std::cout << "\t\t	--> finished diagonalizing for " << info << " - in time : " << tim_s(start) << "s" << std::endl;
    
    // std::cout << "Energies:\n";
    // std::cout << eigenvalues << std::endl;

    std::string name = dir + info + ".hdf5";
    eigenvalues.save(arma::hdf5_name(name, "energies"));
    std::cout << "\t\t	--> finished saving eigenvalues for " << info << " - in time : " << tim_s(start) << "s" << std::endl;
    if(this->ch){
        auto H = this->ptr_to_model->get_dense_hamiltonian();
        H.save(arma::hdf5_name(name, "hamiltonian", arma::hdf5_opts::append));
        std::cout << "\t\t	--> finished saving Hamiltonian for " << info << " - in time : " << tim_s(start) << "s" << std::endl;

        const auto U = this->ptr_to_model->get_model_ref().get_hilbert_space().symmetry_rotation();
        arma::Mat<element_type> V = U * this->ptr_to_model->get_eigenvectors();
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
	
	std::string info = this->set_info();
	std::string filename = info;// + "_subsize=" + std::to_string(LA);
    
    int LA = this->site;
	size_t dim = this->ptr_to_model->get_hilbert_size();
	
    arma::vec emtpy_vec(1);
    if(dim == 0){
        emtpy_vec.save(arma::hdf5_name(dir + filename + ".hdf5", "nope"));
        return;
    }
    
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
        
        for(auto& LA : subsystem_sizes)
            S(n, LA) = entropy::schmidt_decomposition(state, LA, this->L);
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

/// @brief Calculate entanglement entropy in randomly mixed eigenstates (mimic degenerate states)
/// @tparam Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_sym<Hamiltonian>::eigenstate_entanglement_degenerate()
{
   clk::time_point start = std::chrono::system_clock::now();
	
	std::string dir = this->saving_dir + "Entropy" + kPSep + "Degeneracy" + kPSep;// + "Deterministic" + kPSep;
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
    // const arma::vec E = this->ptr_to_model->get_eigenvalues();
    const long E_av_idx = this->ptr_to_model->E_av_idx;
    const long min_idx = dim > 2 * int(this->num_of_points)? E_av_idx - int(this->num_of_points / 2.0) : 0;
    const long max_idx = dim > 2 * int(this->num_of_points)? E_av_idx + int(this->num_of_points / 2.0) : dim - 1;
    std::cout << E_av_idx << "\t\t" << min_idx << "\t\t" << max_idx << std::endl;
    const auto U = this->ptr_to_model->get_model_ref().get_hilbert_space().symmetry_rotation();

    std::cout << " - - - - - - FINISHED CREATING SYMMETRY TRANSFORMATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
    start = std::chrono::system_clock::now();

    auto subsystem_sizes = arma::conv_to<arma::Col<int>>::from(arma::linspace(0, this->L / 2, this->L / 2 + 1));
    std::cout << subsystem_sizes.t() << std::endl;

    arma::mat S(this->num_of_points, subsystem_sizes.size(), arma::fill::zeros);
    arma::vec E_av(this->num_of_points);
    int th_num = this->thread_number;
    omp_set_num_threads(1);
    std::cout << th_num << "\t\t" << omp_get_num_threads() << std::endl;
    auto seed = std::random_device{}();

	disorder<double> random_generator(seed);
	CUE random_matrix(seed);
        // auto start_LA = std::chrono::system_clock::now();
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

            arma::cx_vec state(U.n_cols, arma::fill::zeros);
            arma::cx_vec coeff = Haar.col(id) / std::sqrt(arma::cdot(Haar.col(id), Haar.col(id)));
            // arma::cx_vec coeff(gamma_a, arma::fill::ones);
            // coeff /= std::sqrt(arma::cdot(coeff, coeff));
            long idx = 0;
            for(auto& n : indices){
                E += this->ptr_to_model->get_eigenValue(n) - this->ptr_to_model->get_eigenValue(E_av_idx);
                auto eigenstate = this->ptr_to_model->get_eigenState(n);
                state += coeff(idx++) * eigenstate;
            }
            state = U * state / std::sqrt(arma::cdot(state, state));
            #pragma omp critical
            {
                E_av(gamma_a-1) += E / double(indices.size());
                for(auto& LA : subsystem_sizes)
                    S(gamma_a-1, LA) += entropy::schmidt_decomposition(state, this->L - LA, this->L);
            }
        }
    }
    E_av = E_av / double(this->mu);
    S /= double(this->mu);
    // std::cout << S.col(this->L / 2 - 1).t() << std::endl;
    
    std::cout << " - - - - - - FINISHED ENTROPY CALCULATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
    
    omp_set_num_threads(this->thread_number);
    th_num = 1;
    
    arma::Col<u64> dimension = arma::Col<u64>({dim});
    E_av.save(arma::hdf5_name(dir + filename + ".hdf5", "energies"));
	S.save(arma::hdf5_name(dir + filename + ".hdf5", "entropy", arma::hdf5_opts::append));
	dimension.save(arma::hdf5_name(dir + filename + ".hdf5", "dim", arma::hdf5_opts::append));
        
    // #ifdef ARMA_USE_SUPERLU
    //     arma::Mat<element_type> V;
    //     if(this->ch) V = this->ptr_to_model->get_eigenvectors();
    //     else         V = this->ptr_to_model->get_eigenvectors().submat(0, this->ptr_to_model->E_av_idx - 50, dim - 1, this->ptr_to_model->E_av_idx + 50);
    // #else
    //     arma::Mat<element_type> V = this->ptr_to_model->get_eigenvectors().submat(0, this->ptr_to_model->E_av_idx - 50, dim - 1, this->ptr_to_model->E_av_idx + 50);
    // #endif
    // V.save(arma::hdf5_name(dir + filename + ".hdf5", "eigenvectors",arma::hdf5_opts::append));
}

/// @brief Calculate and save Matrix elments for different operaotors (for now kinetic energy and S_q=0 only)
/// @tparam Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface_sym<Hamiltonian>::diagonal_matrix_elements(){
    
    clk::time_point start = std::chrono::system_clock::now();
	
	std::string dir = this->saving_dir + "DiagonalMatrixElements" + kPSep;
	createDirs(dir);

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

    const auto U = this->ptr_to_model->get_model_ref().get_hilbert_space().symmetry_rotation();
    const arma::vec E = this->ptr_to_model->get_eigenvalues();
    int Ll = this->L;

    const size_t dim_max = ULLPOW(Ll);
    // auto disorder_generator = disorder<double>(this->seed);
    // auto dis_K = disorder_generator.uniform(this->L, -1.0, 1.0);
    // auto dis_K2 = disorder_generator.uniform(this->L, -1.0, 1.0);
    // auto dis_U = disorder_generator.uniform(this->L, -1.0, 1.0);
    // // std::cout << dis_K.t() << std::endl;
    // // std::cout << dis_K2.t() << std::endl;
    // // std::cout << dis_U.t() << std::endl;

    arma::Col<double> KineticEnergy(size, arma::fill::zeros);
    arma::Col<double> KineticEnergy_loc(size, arma::fill::zeros);

    arma::Col<double> NextHopping(size, arma::fill::zeros);
    arma::Col<double> NextHopping_loc(size, arma::fill::zeros);

    arma::Col<double> Interaction(size, arma::fill::zeros);
    arma::Col<double> Interaction_loc(size, arma::fill::zeros);

    arma::Col<double> Sq0_diagmat(size, arma::fill::zeros);
    
    arma::cx_vec jE_diagmat(size, arma::fill::zeros);
    auto check_spin = QOps::__builtins::get_digit(Ll);

#pragma omp parallel for
    for(long n = 0; n < size; n++){
        auto eigenstate = this->ptr_to_model->get_eigenState(n);
        arma::Col<element_type> full_state = U * eigenstate;

        for(long k = 0; k < dim_max; k++)
        {
            // std::cout << boost::dynamic_bitset<>(this->L, k) << "\t\t";
            for(int i = 0; i < this->L; i++)
            {
                int Si = check_spin(k, i);
                
                // Nearest neighbor hopping
                int nei = (this->boundary_conditions)? i + 1 : (i + 1)%this->L;
                if( nei < this->L ){
                    int S_nei = check_spin(k, nei);
                    Interaction(n) += (Si - 0.5) * (S_nei - 0.5) * std::abs(full_state(k) * full_state(k));
                    // Interaction_dis(n) += double(Si * S_nei) / 4.0 * dis_U(i) * std::abs(full_state(k) * full_state(k));
                    if(i == this->L / 2)
                        Interaction_loc(n) += (Si - 0.5) * (S_nei - 0.5) * std::abs(full_state(k) * full_state(k));
                    
                    if( (!Si) && S_nei )
                    {
                        auto [val, state_tmp]   = operators::sigma_minus(k, this->L, nei);
                        auto [val2, new_idx]      = operators::sigma_plus(state_tmp, this->L, i);
                        if(val != 0.0 && val2 != 0.0){
                            KineticEnergy(n) += 2.0 * std::real(std::conj(full_state(new_idx)) * full_state(k));   // because z + z* = 2 Re(z)
                            // KineticEnergy_dis(n) += 2.0 * dis_K(i) * std::real(std::conj(full_state(new_idx)) * full_state(k));
                            if(i == this->L / 2)
                                KineticEnergy_loc(n) += 2.0 * std::real(std::conj(full_state(new_idx)) * full_state(k));
                        }
                    }
                }

                // Next-nearest neighbor hopping
                nei = (this->boundary_conditions)? i + 2 : (i + 2)%this->L;
                if( nei < this->L ){
                    int S_nei = check_spin(k, nei);
                    if( (!Si) && S_nei )
                    {
                        auto [val, state_tmp]   = operators::sigma_minus(k, this->L, nei);
                        auto [val2, new_idx]      = operators::sigma_plus(state_tmp, this->L, i);
                        if(val != 0.0 && val2 != 0.0){
                            NextHopping(n) += 2.0 * std::real(std::conj(full_state(new_idx)) * full_state(k)); // because z + z* = 2 Re(z)
                            // NextHopping_dis(n) += 2.0 * dis_K2(i) * std::real(std::conj(full_state(new_idx)) * full_state(k));
                            if(i == this->L / 2){
                                NextHopping_loc(n) += 2.0 * std::real(std::conj(full_state(new_idx)) * full_state(k));
                            }
                        }
                    }
                }
                // Energy current
                jE_diagmat(n) += 1.0i * jE_mat_elem_kernel(full_state, full_state, i, k, check_spin);

                //<! Hard-core boson momentum occupation
                for(int j = 0; j < this->L; j++)
                {
                    int Sj = check_spin(k, j);
                    if( (!Si) && Sj ){
                        auto [val, state_tmp]   = operators::sigma_minus(k, this->L, j);
                        auto [val2, new_idx]      = operators::sigma_plus(state_tmp, this->L, i);
                        if(val != 0.0 && val2 != 0.0)
                            Sq0_diagmat(n) += std::real(std::conj(full_state(new_idx)) * full_state(k));
                    }
                }
                // diagonal part
                if(Si)
                    Sq0_diagmat(n) += std::real(std::conj(full_state(k)) * full_state(k));
            }
            // std::cout << std::endl;
            // printSeparated(std::cout, "\t", 16, true, 
            //     KineticEnergy(n), KineticEnergy_loc(n), NextHopping(n), NextHopping_loc(n), Interaction(n), Interaction_loc(n), Sq0_diagmat(n));
        }
        // jE_diagmat(n) /= double(this->L);
    }
    Sq0_diagmat = Sq0_diagmat / double(this->L / std::sqrt(2));
    jE_diagmat = jE_diagmat / std::sqrt(this->L);
    
    double norm = this->boundary_conditions? std::sqrt( (this->L - 1) / 2.0) : std::sqrt(this->L / 2.0);
    KineticEnergy = KineticEnergy / norm;
    
    norm = this->boundary_conditions? std::sqrt( (this->L - 2) / 2.0) : std::sqrt(this->L / 2.0);
    NextHopping = NextHopping / norm;

    norm = this->boundary_conditions? std::sqrt( (this->L - 1)) / 4.0 : std::sqrt(this->L) / 4.0;
    Interaction = Interaction / norm;
    // KineticEnergy_dis = KineticEnergy_dis / double(this->L);
    // NextHopping_dis = NextHopping_dis / double(this->L);
    // Interaction_dis = Interaction_dis / double(this->L);
    std::cout << " - - - - - - Calculated Diagonal Matrix Elements IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end

    E.save(arma::hdf5_name(dir + filename + ".hdf5", "energies"));
	KineticEnergy.save(arma::hdf5_name(dir + filename + ".hdf5", "Kin", arma::hdf5_opts::append));
	KineticEnergy_loc.save(arma::hdf5_name(dir + filename + ".hdf5", "Kin_loc", arma::hdf5_opts::append));
	// KineticEnergy_dis.save(arma::hdf5_name(dir + filename + ".hdf5", "Kin_dis", arma::hdf5_opts::append));

	NextHopping.save(arma::hdf5_name(dir + filename + ".hdf5", "Kin2", arma::hdf5_opts::append));
	NextHopping_loc.save(arma::hdf5_name(dir + filename + ".hdf5", "Kin2_loc", arma::hdf5_opts::append));
	// NextHopping_dis.save(arma::hdf5_name(dir + filename + ".hdf5", "Kin2_dis", arma::hdf5_opts::append));

	Interaction.save(arma::hdf5_name(dir + filename + ".hdf5", "Int", arma::hdf5_opts::append));
	Interaction_loc.save(arma::hdf5_name(dir + filename + ".hdf5", "Int_loc", arma::hdf5_opts::append));
	// Interaction_dis.save(arma::hdf5_name(dir + filename + ".hdf5", "Int_dis", arma::hdf5_opts::append));
	Sq0_diagmat.save(arma::hdf5_name(dir + filename + ".hdf5", "Sq0", arma::hdf5_opts::append));
    jE_diagmat.save(arma::hdf5_name(dir + filename + ".hdf5", "jE", arma::hdf5_opts::append));
}