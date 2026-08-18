#include "includes/XXZ_UI.hpp"

int outer_threads = 1;
int num_of_threads = 1;


 /* Overloading * operator */
std::string operator*(std::string a, int b) 
{
    string output = "";
    while (b--) {
        output += a;
    }
    return output;
};

namespace XXZ_UI{

void ui::make_sim(){
    printAllOptions();
    // compare_energies();
    // return;

	this->ptr_to_model = this->create_new_model_pointer();
    
	clk::time_point start = std::chrono::system_clock::now();
    switch (this->fun)
	{
	case 0: 
		diagonalize(); 
		break;
	case 1:
		eigenstate_entanglement();
		break;
    case 2:
        diagonal_matrix_elements();
        break;
	case 3:
		eigenstate_entanglement_degenerate();
		break;
	case 4:
		diagonal_matrix_elements();
		break;
    case 5: 
        spectrals(); break;
    case 6: 
        fractality_in_clean_basis(); break;
    case 7: 
        spectral_form_factor(); break;
    case 8: 
        ErgodicNonDiffusive(); break;
    case 9: 
        gec_moments_calculation(); break;
    case 10: 
        long_time_prediction(); break;
	default:
		#define generate_scaling_array(name) arma::linspace(this->name, this->name + this->name##s * (this->name##n - 1), this->name##n)
        #define for_loop(param, var) for (auto& param : generate_scaling_array(var))

        for_loop(system_size, L){ 
            for_loop(J1x, J1){           
                for_loop(J2x, J2){ 
                    for_loop(delta1x, delta1){   
                        for_loop(delta2x, delta2){       
                                for_loop(hzx, hz){ 
                                    for_loop(wx, w)
        {
            this->L = system_size;
            this->J1 = J1x;
            this->J2 = J2x;
            this->delta1 = delta1x;
            this->delta2 = delta2x;
            this->hz = hzx;
            this->w = wx;
            this->site = this->L / 2.;
            const auto start_loop = std::chrono::system_clock::now();
            std::cout << " - - START NEW ITERATION:\t\t par = "; // simulation end
            printSeparated(std::cout, "\t", 16, true, this->L, this->J1, this->J2, this->delta1, this->delta2, this->hz, this->w);
            
            auto kernel = [&](int k, int p, int zx)
                                    {
                                        this->syms.k_sym = k;
                                        this->syms.p_sym = p;
                                        this->syms.zx_sym = zx;
                                        
                                        this->reset_model_pointer();
                                        // this->diagonal_matrix_elements();
                                        this->spectrals();
                                        // this->diagonalize();
                                        // this->eigenstate_entanglement();
                                        // this->eigenstate_entanglement_degenerate();

                                    };
            // loopSymmetrySectors(kernel); continue;
            this->reset_model_pointer();
            // auto Hamil = this->ptr_to_model->get_hamiltonian();
            // this->l_steps = 0.05 * Hamil.n_cols;
            // if(this->l_steps > 200) this->l_steps = 200;
            // auto polfed = polfed::POLFED<ui::element_type>(Hamil, this->l_steps, this->l_bundle, -1, this->tol, 0.2, this->seed, true);
            // continue;
            // this->eigenstate_entanglement_degenerate(); 
            ErgodicNonDiffusive(); continue;
            // spectrals(); continue;

            diagonal_matrix_elements();
            std::cout << "\t\t - - - - - - FINISHED ITERATION IN : " << tim_s(start_loop) << " seconds\n\t\t\t Total time : " << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
        }}}}}}
            }
        std::cout << "Add default function" << std::endl;
	}
	std::cout << " - - - - - - FINISHED CALCULATIONS IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
}

void ui::fractality_in_clean_basis(){

#ifndef USE_SYMMETRIES
    std::string dir = this->saving_dir + "Fractality" + kPSep;
    // std::string dir = this->saving_dir + "energy_current" + kPSep;
	createDirs(dir);
    const size_t dim_max = 1e5;
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info();
	const size_t size = dim > dim_max? this->l_steps : dim;

    clk::time_point start = std::chrono::system_clock::now();
    auto _hilbert_space = this->ptr_to_model->get_model_ref().get_hilbert_space();
    arma::vec E0;
    arma::cx_mat V0(dim, dim);
    u64 col_start = 0;
    auto kernel = [&](int ks, int ps, int zxs)
        {
            auto model_sym = std::make_unique<QHS::QHamSolver<XXZsym>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->hz, ks, ps, zxs, this->syms.Sz);
            u64 dim_sector = model_sym->get_hilbert_size();
            
            if(dim_sector > 0){
                model_sym->diagonalization();
                const arma::vec Esym = model_sym->get_eigenvalues();
                const auto& Vsym = model_sym->get_eigenvectors();
                const auto U = model_sym->get_model_ref().get_hilbert_space().symmetry_rotation(_hilbert_space);
                // if(ks > 0 && ks < this->L / 2.){
                //     E0 = arma::join_cols(E0, Esym, Esym);
                //     V0.cols(col_start, col_start + dim_sector - 1) = (U * Vsym);
                //     col_start += dim_sector;
                //     V0.cols(col_start, col_start + dim_sector - 1) = (U * Vsym);
                //     col_start += dim_sector;
                // } else 
                {
                    E0 = arma::join_cols(E0, Esym);
                    V0.cols(col_start, col_start + dim_sector - 1) = (U * Vsym);
                    col_start += dim_sector;
                }
            }
        };
    loopSymmetrySectors(kernel);

    // std::cout << "DIM_TOT = " << dim_tot << std::endl;
    auto permut = sort_permutation(E0, [](const double a, const double b)
                            { return a < b; });
    // apply_permutation(E0, permut);
    // apply_permutation(V0, permut);
    const double x = this->w;
    this->w = 0;
    std::cout << " Check what happening INFO = " << this->set_info() << " - - - - - - " << std::endl; // simulation end
    auto unperturbed_ptr = this->create_new_model_pointer();
    const auto& H0 = unperturbed_ptr->get_hamiltonian();
    this->w = x;
    
    arma::sp_mat disord(dim, dim);
    arma::vec h_ell = this->ptr_to_model->get_model_ref()._disorder;
    auto check_spin = QOps::__builtins::get_digit(this->L);
    for (u64 k = 0; k < dim; k++) 
    {
		double s_i;
		u64 base_state = _hilbert_space(k);
		for (int j = 0; j < this->L; j++) 
        {
			s_i = check_spin(base_state, j) ? 0.5 : -0.5;				// true - spin up, false - spin down
            disord(k, k) += s_i * h_ell(j);
		}
	}
    std::cout << " - - - - - - finished collecting unperturbed eigenstates in : " << tim_s(start) << " s. Found D0 = " << col_start << " eigenvalues in H0 for D = " << dim << " hilbert space size - - - - - - " << std::endl; // simulation end
    for(int realis = 0; realis < this->realisations; realis++)
	{
		clk::time_point start_re = std::chrono::system_clock::now();
		if(realis > 0)
			this->ptr_to_model->generate_hamiltonian();
		
		start = std::chrono::system_clock::now();
		if(dim > dim_max){
			this->ptr_to_model->diag_sparse(this->l_steps, this->l_bundle, this->tol, this->seed);	
		}
		else{
        	this->ptr_to_model->diagonalization();
		}
		std::cout << " - - - - - - finished diagonalization in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();
		
		const arma::vec E = this->ptr_to_model->get_eigenvalues();
		const auto& V = this->ptr_to_model->get_eigenvectors();
        // const auto& H = this->ptr_to_model->get_hamiltonian();
        // arma::sp_mat H2 = H*H;

        // arma::vec gec(dim, arma::fill::zeros);
        // for(u64 k = 0; k < dim; k++)
        //     gec(k) = 2 * H2(k,k) - H(k,k) * H(k,k);   
        // gec = gec / arma::trace(H2);

		// arma::mat(H).save(   arma::hdf5_name("Hamiltonian_XXZ.hdf5", "H"));
		// arma::mat(H0).save(   arma::hdf5_name("Hamiltonian_XXZ.hdf5", "H0", arma::hdf5_opts::append));
		// E.save(   arma::hdf5_name("Hamiltonian_XXZ.hdf5", "E", arma::hdf5_opts::append));
		// E0.save(   arma::hdf5_name("Hamiltonian_XXZ.hdf5", "E0", arma::hdf5_opts::append));
		// V.save(   arma::hdf5_name("Hamiltonian_XXZ.hdf5", "V", arma::hdf5_opts::append));
		// V0.save(   arma::hdf5_name("Hamiltonian_XXZ.hdf5", "V0", arma::hdf5_opts::append));
		// arma::mat H_in_H0 = arma::real(V0.t() * H * V0);
		// H_in_H0.save(   arma::hdf5_name("Hamiltonian_XXZ.hdf5", "H_in_H0", arma::hdf5_opts::append));
        // H_in_H0 = arma::imag(V0.t() * H * V0);
		// H_in_H0.save(   arma::hdf5_name("Hamiltonian_XXZ.hdf5", "H_in_H0_im", arma::hdf5_opts::append));

		// arma::cx_vec dis_in_H0 = arma::diagvec( V0.t() * disord * V0 );
		// dis_in_H0.save(   arma::hdf5_name("Hamiltonian_XXZ.hdf5", "dis_of_H0", arma::hdf5_opts::append));
		double E_av = arma::trace(E) / double(dim);

		auto i = std::min_element(std::begin(E), std::end(E), [=](double x, double y) {
			return abs(x - E_av) < abs(y - E_av);
		});
		const long Eav_idx = i - std::begin(E);

        u64 num_of_states_for_Cn = 10;//std::min( (u64)50, u64(0.01*dim) );

        u64 num_of_states = std::min( u64(this->l_steps), u64(0.1*dim) );
		u64	Emin = Eav_idx - num_of_states / 2;
		u64	Emax = Eav_idx + num_of_states / 2;
        const arma::vec energy_density = arma::regspace(0.05, 0.02, 0.95);

        const u64 _size_ipr = dim > 40000? num_of_states : size;
        arma::vec qs = arma::linspace(0, 3.0, 16);
        arma::mat part_ratio_d2(num_of_states, qs.size(), arma::fill::zeros);
		arma::mat part_ratio_d2_comp(num_of_states, qs.size(), arma::fill::zeros);
		arma::mat ldos(num_of_states, energy_density.size()-1, arma::fill::zeros);

        arma::mat coefficients(dim, num_of_states_for_Cn, arma::fill::zeros);
    #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
        for(int n = 0; n < num_of_states; n++)
        {
            arma::Col<element_type> eigenstate = arma::normalise(V.col(n + Emin));
            
            for(int iiq = 0; iiq < qs.size(); iiq++)
                part_ratio_d2_comp(n, iiq) = statistics::participation_ratio(eigenstate, qs(iiq));

            arma::vec overlaps = arma::abs(V0.t() * eigenstate);
              
            // apply_permutation(overlaps, permut);
            
            if(n >= (num_of_states - num_of_states_for_Cn) / 2 && n < (num_of_states + num_of_states_for_Cn) / 2)
                 coefficients.col(n - (num_of_states - num_of_states_for_Cn) / 2) = arma::square(overlaps);
            
            for(int iiq = 0; iiq < qs.size(); iiq++)
                for(int n0 = 0; n0 < dim; n0++)
                    part_ratio_d2(n, iiq) += std::pow(overlaps(n0), 2*qs(iiq));
                    
            //!------- LDOS CALCULATION
            double dE0 = E0(E0.size() - 1) - E0(0);
            for(int e = 0; e < energy_density.size()-1; e++)
            {
                double E_minus = energy_density(e) * dE0 + E0(0);
                double E_plus = energy_density(e+1) * dE0 + E0(0);
                arma::uvec indices = arma::find(E0 >= E_minus && E0 < E_plus);
                ldos(n, e) = arma::accu( arma::square( arma::abs(overlaps.rows(indices)) ) ) / double(indices.size());
            }
        }
        std::cout << " - - - - - - finished realization = " << realis << " in : " << tim_s(start_re) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
		
        std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
        
        createDirs(dir_realis);
        E.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "energies"));
        E0.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "E0",   arma::hdf5_opts::append));
        // gec.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "GEC",   arma::hdf5_opts::append));
        coefficients.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "coefficients",   arma::hdf5_opts::append));
        
        ldos.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "LDOS",   arma::hdf5_opts::append));
        energy_density.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "energy_density",   arma::hdf5_opts::append));
        
        qs.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "qs",   arma::hdf5_opts::append));
        part_ratio_d2.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "P2",   arma::hdf5_opts::append));
        part_ratio_d2_comp.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "P2_comp",   arma::hdf5_opts::append));
        // energy_density.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "energy_density",   arma::hdf5_opts::append));
        // energy_density.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "energy_density",   arma::hdf5_opts::append));
    }
#endif
}

void ui::spectrals()
{
	std::string dir = this->saving_dir + "Spectrals_SzSz" + kPSep;
    // std::string dir = this->saving_dir + "energy_current" + kPSep;
    // std::string dir = this->saving_dir + "parity" + kPSep;
	createDirs(dir);
	
    const size_t dim_max = 1e5;
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info();
	const size_t size = dim > dim_max? this->l_steps : dim;

	int Ll = this->L;
	int counter = 0;
	
	const double _bandwidth_def = std::sqrt(this->L);
	const double _tH = double(dim) / _bandwidth_def;
	
	const arma::vec omegax = arma::logspace(std::log10(1.0/dim) - 1, std::log10( _bandwidth_def ) + 1, 10 * this->L);
	const arma::vec energy_density = arma::regspace(0.05, 0.02, 0.95);
	
	arma::vec times = arma::logspace(-2, (std::log10(1000 * _tH)) + 1.5, this->num_of_points);

	double window_width = 0.04;
	auto _hilbert_space = this->ptr_to_model->get_model_ref().get_hilbert_space();
	
    // const size_t dim_full = ULLPOW(this->L);
    arma::sp_mat kinetic(dim, dim);
    arma::sp_mat perturbation(dim, dim);
    // // arma::sp_mat U_U1(dim_full, dim);
    auto check_spin = QOps::__builtins::get_digit(this->L);

    arma::vec disorder = this->ptr_to_model->get_model_ref()._disorder;
    // auto op = std::make_unique<QHS::QHamSolver<XXZ>>(this->boundary_conditions, this->L, 1, 0, 0, 0, 0, this->syms.Sz, this->add_parity_breaking, this->w, this->seed);
    for (u64 k = 0; k < dim; k++) 
    {
		double s_i;
		u64 base_state = _hilbert_space(k);
		// s_i = check_spin(base_state, 0) ? 0.5 : -0.5;
        // kinetic(k, k) = s_i;
        // U_U1(base_state, k) = 1.0;
		for (int j = 0; j < this->L; j++) 
        {
			s_i = check_spin(base_state, j) ? 0.5 : -0.5;				// true - spin up, false - spin down
            perturbation(k,k) += disorder(j) * s_i;
            
            int nei = j + 1;
            if(nei >= this->L)
                nei = (this->boundary_conditions>0)? -1 : nei % this->L;
            
            double s_j = check_spin(base_state, nei) ? 0.5 : -0.5;				// true - spin up, false - spin down
            if(nei >= 0){
                kinetic(k, k) += s_i * s_j;
            }
		}
	}
    kinetic = kinetic * 1. / std::sqrt(this->L);
    // auto kernel = [&check_spin, Ll](u64 state) -> std::pair<u64, double>
	// 			{ 
	// 			double s_i = check_spin(state, Ll-1) ? 0.5 : -0.5;
	// 			return std::make_pair(state, s_i);
	// 			};
    // auto _operator = QOps::generic_operator<double>(this->L, std::move(kernel), 1.0);
    // kinetic = (_operator.to_matrix(dim));
    // arma::sp_cx_mat kinetic = spin_current();
    // double Jx = this->J1;
    // double Jy = this->J1;
    // double Jz = this->delta1;
    // arma::sp_cx_mat kinetic(dim, dim);
    // for(int j = 0; j < this->L; j++)
    // {
    //     int nei = j + 1;
    //     if(nei >= this->L)
    //         nei = (this->boundary_conditions>0)? -1 : nei % this->L;
        
    //     int nei2 = j + 2;
    //     if(nei2 >= this->L)
    //         nei2 = (this->boundary_conditions>0)? -1 : nei2 % this->L;
    //     if(nei >0 && nei2 > 0)
    //     {
    //         for(long k = 0; k < dim; k++)
    //         {
    //             u64 base_state = _hilbert_space(k);
    //             // double Si = double(check_spin(k, i)) - 0.5;
    //             // double Snei = double(check_spin(k, nei)) - 0.5;
    //             // {
    //             //     auto [val, state_tmp]   = operators::sigma_x<cpx>(base_state, this->L, i);
    //             //     auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei);
    //             //     u64 idx = _hilbert_space.find(new_idx);
    //             //     if(idx < dim)
    //             //         kinetic(idx, k) -= (val * val2);
    //             // }{
    //             //     auto [val, state_tmp]   = operators::sigma_x<cpx>(base_state, this->L, nei);
    //             //     auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, i);
    //             //     u64 idx = _hilbert_space.find(new_idx);
    //             //     if(idx < dim)
    //             //         kinetic(idx, k) += (val * val2);
    //             // }
    //             double Si = double(check_spin(base_state, j)) - 0.5;
    //             double Snei = double(check_spin(base_state, nei)) - 0.5;
    //             double Snei2 = double(check_spin(base_state, nei2)) - 0.5;
    //             {
    //                 // + Sx Sz Sy
    //                 auto [val, state_tmp]   = operators::sigma_x<cpx>(base_state, this->L, j);
    //                 auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei2);
    //                 u64 idx = _hilbert_space.find(new_idx);
    //                 if(idx < dim)
    //                     kinetic(idx, k) += 0.8*(Jx * Jy * Snei * val * val2);
    //             }{
    //                 // - Sy Sz Sx
    //                 auto [val, state_tmp]   = operators::sigma_x<cpx>(base_state, this->L, nei2);
    //                 auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, j);
    //                 u64 idx = _hilbert_space.find(new_idx);
    //                 if(idx < dim)
    //                     kinetic(idx, k) -= 0.8*(Jx * Jy * Snei * val * val2);
    //             }{
    //                 // + Sy Sx Sz
    //                 auto [val, state_tmp]   = operators::sigma_x<cpx>(base_state, this->L, nei);
    //                 auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, j);
    //                 u64 idx = _hilbert_space.find(new_idx);
    //                 if(idx < dim)
    //                     kinetic(idx, k) += 0.8*(Jz * Jy * Snei2 * val * val2);
    //             }{
    //                 // - Sx Sy Sz
    //                 auto [val, state_tmp]   = operators::sigma_x<cpx>(base_state, this->L, j);
    //                 auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei);
    //                 u64 idx = _hilbert_space.find(new_idx);
    //                 if(idx < dim)
    //                     kinetic(idx, k) -= 0.8*(Jz * Jx * Snei2 * val * val2);
                    
    //             }{
    //                 // Sz Sy Sx
    //                 auto [val, state_tmp]   = operators::sigma_x<cpx>(base_state, this->L, nei2);
    //                 auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei);
    //                 u64 idx = _hilbert_space.find(new_idx);
    //                 if(idx < dim)
    //                     kinetic(idx, k) += 0.8*(Jz * Jx * Si * val * val2);
    //             }{
    //                 // - Sz Sx Sy
    //                 auto [val, state_tmp]   = operators::sigma_x<cpx>(base_state, this->L, nei);
    //                 auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei2);
    //                 u64 idx = _hilbert_space.find(new_idx);
    //                 if(idx < dim)
    //                     kinetic(idx, k) -= 0.8*(Jz * Jy * Si * val * val2);
    //             }
    //         }
    //     }
    // }
    // auto parity = QOps::_parity_symmetry(this->L, this->syms.p_sym);
    // arma::sp_cx_mat kinetic = parity.to_reduced_matrix(_hilbert_space);
    // std::cout << arma::mat(arma::real(kinetic)) << std::endl;
    double _operator_HSnorm = arma::trace(kinetic.t() * kinetic) / double(dim);
	kinetic = kinetic / std::sqrt(_operator_HSnorm);

    // const auto U = this->ptr_to_model->get_model_ref().get_hilbert_space().symmetry_rotation();
    // arma::sp_cx_mat kinetic2 = U.t() * energy_current() * U;
    // arma::sp_cx_mat kinetic2 = U_U1.t() * spin_current() * U_U1;
	// cpx _operator_HSnorm = arma::trace(kinetic2 * kinetic2.t()) / double(dim);
    // std::cout << "Hilbert-Schmidt Norm\t\t" << _operator_HSnorm << std::endl;
	// kinetic2 = kinetic2 / std::sqrt(_operator_HSnorm);
    // kinetic = arma::imag(kinetic2);
	std::cout << "Hilbert-Schmidt Norm\t\t" << _operator_HSnorm << "Prediction\t\t" << this->L / 8 << "\t\tNew Norm\t\t" << arma::trace(kinetic.t() * kinetic.t()) / double(dim) << std::endl;
    

    // auto subsystem_sizes = arma::conv_to<arma::Col<int>>::from(arma::linspace(0, this->L, this->L + 1));
	// std::cout << subsystem_sizes.t() << std::endl;
	// std::vector<QOps::generic_operator<element_type>> permutation_op;
	// for(int LA_idx = 0; LA_idx < subsystem_sizes.size() - 1; LA_idx++)
	// {	
	// 	int LA = subsystem_sizes[LA_idx];
	// 	auto start_LA = std::chrono::system_clock::now();
	// 	std::vector<int> p(this->L);
	// 	p[LA % this->L] = 0;
	// 	for(int l = 0; l < this->L; l++){
	// 		if(l != LA % this->L){
	// 			p[l] = (l < (LA % this->L) )? l + 1 : l;
	// 		}
	// 	}
	// 	// std::cout << LA << "\t\t" << p << "\t\t" << p2 << std::endl;
	// 	auto permutation = QOps::_permutation_generator<element_type>(this->L, p);
	// 	permutation_op.push_back(permutation);

	// 	std::cout << " - - - - - - set permutation matrix for LA = " << LA << " in : " << tim_s(start_LA) << " s - - - - - - " << std::endl;
	// }
// #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)

#ifdef USE_SYMMETRIES
    int RRR  = 1;
#else
    int RRR = this->realisations;
#endif
	for(int realis = 0; realis < RRR; realis++)
	{
		clk::time_point start_re = std::chrono::system_clock::now();
		if(realis > 0)
			this->ptr_to_model->generate_hamiltonian();
		
		clk::time_point start = std::chrono::system_clock::now();
		if(dim > dim_max){
			this->ptr_to_model->diag_sparse(this->l_steps, this->l_bundle, this->tol, this->seed);	
		}
		else{
        	this->ptr_to_model->diagonalization();
		}
		std::cout << " - - - - - - finished diagonalization in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();
		
		const arma::vec E = this->ptr_to_model->get_eigenvalues();
		const auto& V = this->ptr_to_model->get_eigenvectors();
		double E_av = arma::trace(E) / double(dim);

		auto i = std::min_element(std::begin(E), std::end(E), [=](double x, double y) {
			return abs(x - E_av) < abs(y - E_av);
		});
		const long Eav_idx = i - std::begin(E);
		long int E_min = dim > dim_max? 1 : Eav_idx - long(dim / 4);
		long int E_max = dim > dim_max? size-1 : Eav_idx + long(dim / 4);

		double wH = 0, r = 0, count = 0;
		for (long int i = E_min; i < E_max; i++){
            double dE1 = E(i+1) - E(i);
            double dE2 = E(i) - E(i-1);
			wH += E(i+1) - E(i);
            r += min(dE1, dE2) / max(dE1, dE2);
            count += 1;
        }
		wH /= double(count);
        r /= double(count);

		start = std::chrono::system_clock::now();
		
        double quench_E, tot_spin_init;
        arma::vec quench;
        arma::cx_mat psi;
        arma::Col<element_type> coeff;
        if(dim < dim_max){
            arma::Col<element_type> Hdiagonal = arma::diagvec( this->ptr_to_model->get_dense_hamiltonian() );

            quench = arma::vec(times.size(), arma::fill::zeros);
        //     psi = arma::cx_mat(dim, times.size(), arma::fill::zeros);
        //     start = std::chrono::system_clock::now();
        //     std::cout << " - - - - - - finished finding product state with energy E = " << quench_E << " compared to mean energy <H> = " << E_av << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
            

        //     start = std::chrono::system_clock::now();
        // #pragma omp parallel for
        //     for(long t_idx = 0; t_idx < times.size(); t_idx++)
        //     {
        //         double time = times(t_idx);
        //         for(long alfa = 0; alfa < size; alfa++)
        //         {
        //             auto state = V.col(alfa);
        //             psi.col(t_idx) += std::exp(-1i * time * E(alfa)) * state * state(idx);
        //         }
        //     }

        //     std::cout << " - - - - - - finished preparing initial states for all times in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
        }
		arma::Mat<ui::element_type> mat_elem = ( V.t() * kinetic * V);
        // std::cout << mat_elem << std::endl;
		arma::Col<ui::element_type> diag_mat_elem = arma::diagvec( mat_elem );
        // std::cout << mat_elem << std::endl;
        // std::cout << V.t() * kinetic * V << std::endl;
		// arma::mat xx = arma::abs(mat_elem);
		// xx.save(   arma::hdf5_name("MAT_ELEM" + info + ".hdf5", "mat_elem"));
		// xx = ( arma::mat(total_spin) );
		// xx.save(   arma::hdf5_name("MAT_ELEM" + info + ".hdf5", "sparse", arma::hdf5_opts::append));

		std::cout << " - - - - - - finished matrix elements in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();
		auto [_susc, _susc_r] = adiabatics::gauge_potential_save((mat_elem), E, this->L, wH);

		std::cout << " - - - - - - finished AGP in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();
		arma::mat _integrated_spectral_fun(omegax.size()-1, energy_density.size(), arma::fill::zeros);
		arma::mat _spectral_fun(omegax.size()-1, energy_density.size(), arma::fill::zeros);
		arma::mat _spectral_fun_typ(omegax.size()-1, energy_density.size(), arma::fill::zeros);
		arma::mat _element_count(omegax.size()-1, energy_density.size(), arma::fill::zeros);
		
		const double bandwidth = E(E.size() - 1) - E(0);
	#pragma omp parallel for
		for(int ii = 0; ii < energy_density.size(); ii++){
			const double eps = energy_density(ii);
			const double energyx = eps * bandwidth + E(0);
			spectrals::preset_omega set_omega(E, window_width, energyx);
			arma::vec omegas_i, matter;
            std::tie(omegas_i, matter) = set_omega.get_matrix_elements(mat_elem);
            for(int k = 0; k < omegax.size() - 1; k++){
                arma::uvec indices = arma::find(omegas_i >= omegax[k] && omegas_i < omegax[k+1]);
                if(indices.size() > 0){
                    _element_count(k, ii) = indices.size();
                    arma::vec x = arma::vec( omegas_i.elem(indices) );
                    arma::vec y = arma::vec( matter.elem(indices) );
                    _spectral_fun(k, ii) = arma::accu( y );
                    _spectral_fun_typ(k, ii) = arma::accu( arma::log(y) );
                }
                indices = arma::find(omegas_i < omegax[k+1]);
                if(indices.size() > 0){
                    arma::vec y = arma::vec( matter.elem(indices) );
                    _integrated_spectral_fun(k, ii) = arma::accu(y);
                }
            }
		}

        arma::vec _spectral_fun_all(omegax.size()-1, arma::fill::zeros);
        arma::vec _element_count_all(omegax.size()-1, arma::fill::zeros);
        const double dw_log = std::log10(omegax[1]) - std::log10(omegax[0]);
        const double w0_log = std::log10(omegax[0]);
        {
            for(int n = 0; n < E.size() - 1; n++){
                for(int m = n+1; m < E.size() - 1; m++){
                    double wnm = E(m) - E(n);
                    const auto idx = int( (std::log10(wnm) - w0_log) / dw_log);
                    if(idx < omegax.size() && idx >= 0){
                        _spectral_fun_all(idx) += 2 * std::abs(mat_elem(n, m) * mat_elem(m, n));
                        _element_count_all(idx) += 2;
                    }
                }	
            }
            std::cout << " - - - - - - finished spectral function at all energy density for realis = " << realis << " in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
        }
		std::cout << " - - - - - - finished \hat{V} matrix elements in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end

        if(dim < dim_max){
            start = std::chrono::system_clock::now();
        #pragma omp parallel for
            for(long t_idx = 0; t_idx < times.size(); t_idx++){
                for(int n = 0; n < E.size() - 1; n++){
                    quench(t_idx) += std::abs(mat_elem(n,n) * std::conj(mat_elem(n,n)));
                    for(int m = n+1; m < E.size() - 1; m++){
                        double wnm = E(m) - E(n);
                        quench(t_idx) += 2.0 * std::abs(mat_elem(n,m) * std::conj(mat_elem(m,n))) * std::cos(wnm * times(t_idx));
                    }	
                }
                quench(t_idx) /= double(dim);
            }
                // quench(t_idx) = std::real( arma::cdot(psi.col(t_idx), kinetic * psi.col(t_idx)) );
            
        std::cout << " - - - - - - finished time evolution in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
        }
        start = std::chrono::system_clock::now();

		// E_min = dim > dim_max? 0 : Eav_idx - std::min(50, int(dim/50));
		// E_max = dim > dim_max? size : Eav_idx + std::min(50, int(dim/50));
        // u64 num = E_max - E_min;
        // arma::mat S(num, this->L + 1, arma::fill::zeros);
		// arma::mat S_site = S;
		// arma::vec participation_entropy(num, arma::fill::zeros);
		
    //     #ifdef USE_SYMMETRIES
    //         const auto U = this->ptr_to_model->get_model_ref().get_hilbert_space().symmetry_rotation();
    //     #endif
	// #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
	// 	for(int _n = 0; _n < num; _n++){
    //         auto n = _n + E_min;
	// 		arma::Col<element_type> state = arma::normalise(this->ptr_to_model->get_eigenState(n));
    //         #ifdef USE_SYMMETRIES
    //             state = U * state;
    //         #endif
	// 		// arma::Col<element_type> state2 = arma::normalise(this->ptr_to_model->get_eigenState(n));
			
	// 		#pragma omp parallel for
	// 			for(int k = 0; k < dim; k++){
	// 				auto value = std::abs(state(k)) * std::abs(state(k));
	// 				participation_entropy(_n) += (std::abs(value) > 0) ? -value * std::log(value) : 0;
	// 			}

	// 		state = this->cast_state(state);

	// 		for(int LA_idx = 0; LA_idx < subsystem_sizes.size() - 1; LA_idx++)
	// 		{	
	// 			int LA = subsystem_sizes[LA_idx];
	// 			S(_n, LA_idx) = entropy::schmidt_decomposition(state, this->L - LA, this->L);	// bipartite entanglement at subsystem size LA
				
	// 			arma::Col<element_type> permuted_state = permutation_op[LA_idx].multiply(state);
	// 			S_site(_n, LA_idx) = entropy::schmidt_decomposition(permuted_state, this->L - 1, this->L);	// single site entanglement at site LA
	// 		}
	// 	}
    // 	omp_set_num_threads(this->thread_number);
    //     std::cout << " - - - - - - finished entanglement entropy in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
        start = std::chrono::system_clock::now();
		u64 num_of_states = std::min( u64(this->l_steps), u64(0.02*dim) );
		u64	Emin = Eav_idx - num_of_states / 2;
		u64	Emax = Eav_idx + num_of_states / 2;

        #ifdef USE_SYMMETRIES
            const double x = this->delta2;
            this->delta2 = 0;
        #else
            const double x = this->w;
            this->w = 0;
        #endif
            std::cout << " Check what happening INFO = " << this->set_info() << " - - - - - - " << std::endl; // simulation end
            auto unperturbed_ptr = this->create_new_model_pointer();
            unperturbed_ptr->diagonalization();
            const arma::vec E0 = unperturbed_ptr->get_eigenvalues();
		    const auto& V0 = unperturbed_ptr->get_eigenvectors();

            double dE0 = E0(E0.size()-1) - E0(0);

        #ifdef USE_SYMMETRIES
            this->delta2 = x;
        #else
            this->w = x;
        #endif

        // const auto& H = this->ptr_to_model->get_hamiltonian();
        // const auto& H0 = unperturbed_ptr->get_hamiltonian();
		// arma::mat(H).save(   arma::hdf5_name("Hamiltonian_XXZ_sym.hdf5", "H"));
		// arma::mat(H0).save(   arma::hdf5_name("Hamiltonian_XXZ_sym.hdf5", "H0", arma::hdf5_opts::append));
		// E.save(   arma::hdf5_name("Hamiltonian_XXZ_sym.hdf5", "E", arma::hdf5_opts::append));
		// E0.save(   arma::hdf5_name("Hamiltonian_XXZ_sym.hdf5", "E0", arma::hdf5_opts::append));
		// V.save(   arma::hdf5_name("Hamiltonian_XXZ_sym.hdf5", "V", arma::hdf5_opts::append));
		// V0.save(   arma::hdf5_name("Hamiltonian_XXZ_sym.hdf5", "V0", arma::hdf5_opts::append));
		// arma::mat H_in_H0 = arma::real(V0.t() * H * V0);
		// H_in_H0.save(   arma::hdf5_name("Hamiltonian_XXZ_sym.hdf5", "H_in_H0", arma::hdf5_opts::append));
        // H_in_H0 = arma::imag(V0.t() * H * V0);
		// H_in_H0.save(   arma::hdf5_name("Hamiltonian_XXZ_sym.hdf5", "H_in_H0_im", arma::hdf5_opts::append));

		// arma::Col<elem_ty> pert_in_H0 = arma::diagvec( V0.t() * kinetic * V0 );
		// pert_in_H0.save(   arma::hdf5_name("Hamiltonian_XXZ_sym.hdf5", "dis_of_H0", arma::hdf5_opts::append));
        std::cout << " - - - - - - finished diagonalization of unperturbed H in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
        start = std::chrono::system_clock::now();

		double wH0 = 0, r0 = 0, count0 = 0;
		for (long int i = E_min; i < E_max; i++){
            double dE1 = E0(i+1) - E0(i);
            double dE2 = E0(i) - E0(i-1);
			wH0 += E0(i+1) - E0(i);
            r0 += min(dE1, dE2) / max(dE1, dE2);
            count0 += 1;
        }
		wH0 /= double(count0);
        r0 /= double(count0);

		outer_threads = this->thread_number;
		omp_set_num_threads(1);
		std::cout << outer_threads << "\t\t" << omp_get_num_threads() << std::endl;

	    arma::vec energy_density2 = arma::vec({0.0, 0.0831, 0.1265, 0.1572, 0.1814, 0.2017, 0.2194, 0.235, 0.2493, 0.2623, 0.2744, 0.2857, 0.2964, 0.3065, 0.3162, 0.3254, 0.3343, 0.3429, 0.3512, 0.3592, 0.367, 0.3747, 0.3821, 0.3894, 0.3965, 0.4036, 0.4105, 0.4172, 0.4239, 0.4306, 0.4371, 0.4436, 0.45, 0.4563, 0.4627, 0.4689, 0.4752, 0.4814, 0.4876, 0.4938, 0.5, 0.5062, 0.5124, 0.5186, 0.5248, 0.5311, 0.5373, 0.5437, 0.55, 0.5564, 0.5629, 0.5694, 0.5761, 0.5828, 0.5895, 0.5964, 0.6035, 0.6106, 0.6179, 0.6253, 0.633, 0.6408, 0.6488, 0.6571, 0.6657, 0.6746, 0.6838, 0.6935, 0.7036, 0.7143, 0.7256, 0.7377, 0.7507, 0.765, 0.7806, 0.7983, 0.8186, 0.8428, 0.8735, 0.9169, 1.0	});
		
        const u64 _size_ipr = dim > 40000? num_of_states : size;
        arma::vec part_ratio_d2(_size_ipr, arma::fill::zeros);
		arma::vec part_ratio_d2_comp(_size_ipr, arma::fill::zeros);
		arma::mat ldos(num_of_states, energy_density.size()-1, arma::fill::zeros);
        arma::mat ldos2(num_of_states, energy_density2.size()-1, arma::fill::zeros);

        if(dim > 40000)
        {
        #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
            for(int n = 0; n < num_of_states; n++)
            {
                arma::Col<element_type> eigenstate = arma::normalise(V.col(n + Emin));
                
                part_ratio_d2_comp(n) = statistics::participation_ratio(eigenstate, 2);
                part_ratio_d2(n) = statistics::participation_ratio(eigenstate, V0, 2);
                
                //!------- LDOS CALCULATION

                // const auto idx = int( (std::log10(E0(n)) - E0(0)) / energy_window);
                arma::Col<element_type> overlaps = V0.t() * eigenstate; //.rows(indices_E0);
                for(int e = 0; e < energy_density.size()-1; e++)
                {
                    double E_minus = energy_density(e) * dE0 + E0(0);
                    double E_plus = energy_density(e+1) * dE0 + E0(0);
                    arma::uvec indices = arma::find(E0 >= E_minus && E0 < E_plus);
                    ldos(n, e) = arma::accu( arma::square( arma::abs(overlaps.rows(indices)) ) ) / double(indices.size());
                }
                for(int e = 0; e < energy_density2.size()-1; e++)
                {
                    double E_minus = energy_density2(e) * dE0 + E0(0);
                    double E_plus = energy_density2(e+1) * dE0 + E0(0);
                    arma::uvec indices = arma::find(E0 >= E_minus && E0 < E_plus);
                    ldos2(n, e) = arma::accu( arma::square( arma::abs(overlaps.rows(indices)) ) ) / double(indices.size());
                }
            }
        } 
        else 
        {
        #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
            for(int n = 0; n < size; n++)
            {
                const u64 _idx_ = dim < dim_max? n - Emin : n;
                arma::Col<element_type> eigenstate = arma::normalise(V.col(n));
                
                part_ratio_d2_comp(n) = statistics::participation_ratio(eigenstate, 2);
                part_ratio_d2(n) = statistics::participation_ratio(eigenstate, V0, 2);
                
                //!------- LDOS CALCULATION
                if(n >= Emin && n < Emax)
                {
                    // const auto idx = int( (std::log10(E0(n)) - E0(0)) / energy_window);
                    arma::Col<element_type> overlaps = V0.t() * eigenstate; //.rows(indices_E0);
                    for(int e = 0; e < energy_density.size()-1; e++)
                    {
                        double E_minus = energy_density(e) * dE0 + E0(0);
                        double E_plus = energy_density(e+1) * dE0 + E0(0);
                        arma::uvec indices = arma::find(E0 >= E_minus && E0 < E_plus);
                        ldos(_idx_, e) = arma::accu( arma::square( arma::abs(overlaps.rows(indices)) ) ) / double(indices.size());
                    }
                    for(int e = 0; e < energy_density2.size()-1; e++)
                    {
                        double E_minus = energy_density2(e) * dE0 + E0(0);
                        double E_plus = energy_density2(e+1) * dE0 + E0(0);
                        arma::uvec indices = arma::find(E0 >= E_minus && E0 < E_plus);
                        ldos2(n-Emin, e) = arma::accu( arma::square( arma::abs(overlaps.rows(indices)) ) ) / double(indices.size());
                    }
                }
            }
        }
        outer_threads = 1;
		omp_set_num_threads(this->thread_number);

		std::cout << " - - - - - - finished IPR all for q=2 in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
        start = std::chrono::system_clock::now();
        mat_elem = (V0.t() * kinetic * V0);
		arma::Col<ui::element_type> diag_mat_elem0 = arma::diagvec(mat_elem);
		// arma::mat xx = arma::abs(mat_elem);
		// xx.save(   arma::hdf5_name("MAT_ELEM" + info + ".hdf5", "mat_elem"));
		// xx = ( arma::mat(total_spin) );
		// xx.save(   arma::hdf5_name("MAT_ELEM" + info + ".hdf5", "sparse", arma::hdf5_opts::append));

		std::cout << " - - - - - - finished matrix elements in unperturbed basis in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();
		arma::mat _integrated_spectral_fun0(omegax.size()-1, energy_density.size(), arma::fill::zeros);
		arma::mat _spectral_fun0(omegax.size()-1, energy_density.size(), arma::fill::zeros);
		arma::mat _spectral_fun_typ0(omegax.size()-1, energy_density.size(), arma::fill::zeros);
		arma::mat _element_count0(omegax.size()-1, energy_density.size(), arma::fill::zeros);
		
		const double bandwidth0 = E0(E0.size() - 1) - E0(0);
	#pragma omp parallel for
		for(int ii = 0; ii < energy_density.size(); ii++){
			const double eps = energy_density(ii);
			const double energyx = eps * bandwidth0 + E0(0);
			spectrals::preset_omega set_omega(E0, window_width, energyx);
			arma::vec omegas_i, matter;
            std::tie(omegas_i, matter) = set_omega.get_matrix_elements(mat_elem);
            for(int k = 0; k < omegax.size() - 1; k++){
                arma::uvec indices = arma::find(omegas_i >= omegax[k] && omegas_i < omegax[k+1]);
                if(indices.size() > 0){
                    _element_count0(k, ii) = indices.size();
                    arma::vec x = arma::vec( omegas_i.elem(indices) );
                    arma::vec y = arma::vec( matter.elem(indices) );
                    _spectral_fun0(k, ii) = arma::accu( y );
                    _spectral_fun_typ0(k, ii) = arma::accu( arma::log(y) );
                }
                indices = arma::find(omegas_i < omegax[k+1]);
                if(indices.size() > 0){
                    arma::vec y = arma::vec( matter.elem(indices) );
                    _integrated_spectral_fun0(k, ii) = arma::accu(y);
                }
            }
		}

        arma::vec _spectral_fun_all0(omegax.size()-1, arma::fill::zeros);
        arma::vec _element_count_all0(omegax.size()-1, arma::fill::zeros);
        {
            for(int n = 0; n < E.size() - 1; n++){
                for(int m = n+1; m < E.size() - 1; m++){
                    double wnm = E(m) - E(n);
                    const auto idx = int( (std::log10(wnm) - w0_log) / dw_log);
                    if(idx < omegax.size() && idx >= 0){
                        _spectral_fun_all0(idx) += 2 * std::abs(mat_elem(n, m) * mat_elem(m, n));
                        _element_count_all0(idx) += 2;
                    }
                }	
            }
            std::cout << " - - - - - - finished spectral function at all energy density for realis = " << realis << " in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
            // start = std::chrono::system_clock::now();
        }
		std::cout << " - - - - - - finished \hat{V} matrix elements in unperturbed basis in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
        start = std::chrono::system_clock::now();
        mat_elem = (V0.t() * perturbation * V0);
        arma::vec FGR_width(E.size(), arma::fill::zeros);
        arma::vec FGR_width_typ(E.size(), arma::fill::zeros);
        for(int n = 0; n < E.size(); n++)
        {
            for(int m = n+1; m < E.size(); m++){
                double _elem_ = std::abs(mat_elem(n,m));
                FGR_width(n) += 2 * (_elem_ * _elem_);
                FGR_width_typ(n) += 2 * std::log( (_elem_ * _elem_) );
            }	
        }
        std::cout << " - - - - - - finished Fermi Golden Rule in unperturbed basis in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
    	omp_set_num_threads(this->thread_number);
        {
            #ifdef USE_SYMMETRIES
                std::string dir_realis = dir;
            #else
			    std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
            #endif
			createDirs(dir_realis);
			E.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "energies"));
			omegax.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "omegas",   arma::hdf5_opts::append));
			_integrated_spectral_fun.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "integrated_spectral_fun",   arma::hdf5_opts::append));
			energy_density.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "energy_density",   arma::hdf5_opts::append));
			_spectral_fun.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "spectral_fun",   arma::hdf5_opts::append));
			_spectral_fun_typ.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "log(_spectral_fun_typ)",   arma::hdf5_opts::append));
			_element_count.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "element_count",   arma::hdf5_opts::append));
            _spectral_fun_all.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "spectral_fun_all",   arma::hdf5_opts::append));
            _element_count_all.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "element_count_all",   arma::hdf5_opts::append));

			FGR_width.save(	 arma::hdf5_name(dir_realis + info + ".hdf5", "FGR_width",     arma::hdf5_opts::append));
			FGR_width_typ.save(arma::hdf5_name(dir_realis + info + ".hdf5", "FGR_width_typ", arma::hdf5_opts::append));

			_susc.save(	 arma::hdf5_name(dir_realis + info + ".hdf5", "susc",     arma::hdf5_opts::append));
			_susc_r.save(arma::hdf5_name(dir_realis + info + ".hdf5", "susc_reg", arma::hdf5_opts::append));

			coeff.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "coefficients", arma::hdf5_opts::append));
			arma::vec x = arma::real(diag_mat_elem);  x.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "diag_mat",   arma::hdf5_opts::append));
            x = arma::imag(diag_mat_elem);            x.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "diag_mat_im",   arma::hdf5_opts::append));
			times.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "times",   arma::hdf5_opts::append));
			quench.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "autocorrelation",   arma::hdf5_opts::append));
			// arma::vec( {quench_E} ).save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench_energy",   arma::hdf5_opts::append));
			// arma::vec( {tot_spin_init} ).save(   arma::hdf5_name(dir_realis + info + ".hdf5", "tot_spin_init",   arma::hdf5_opts::append));
			// arma::vec( {_operator_HSnorm} ).save(   arma::hdf5_name(dir_realis + info + ".hdf5", "HSnorm",   arma::hdf5_opts::append));

			arma::vec( {wH} ).save(   arma::hdf5_name(dir_realis + info + ".hdf5", "wH",   arma::hdf5_opts::append));
			arma::vec( {r} ).save(   arma::hdf5_name(dir_realis + info + ".hdf5", "gap_ratio",   arma::hdf5_opts::append));
			arma::vec( {wH0} ).save(   arma::hdf5_name(dir_realis + info + ".hdf5", "wH0",   arma::hdf5_opts::append));
			arma::vec( {r0} ).save(   arma::hdf5_name(dir_realis + info + ".hdf5", "gap_ratio0",   arma::hdf5_opts::append));

			// S.save(arma::hdf5_name(dir_realis + info + ".hdf5", "entropy", arma::hdf5_opts::append));
			// S_site.save(arma::hdf5_name(dir_realis + info + ".hdf5", "single_site_entropy", arma::hdf5_opts::append));
			// subsystem_sizes.save(arma::hdf5_name(dir_realis + info + ".hdf5", "subsystem sizes", arma::hdf5_opts::append));
			// participation_entropy.save(arma::hdf5_name(dir_realis + info + ".hdf5", "von Neumann participation entropy", arma::hdf5_opts::append));

            E0.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "E0",   arma::hdf5_opts::append));
            part_ratio_d2.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "Pr",   arma::hdf5_opts::append));
            part_ratio_d2_comp.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "Pr_comp",   arma::hdf5_opts::append));

            ldos.save(arma::hdf5_name(dir_realis + info + ".hdf5", "LDOS", arma::hdf5_opts::append));
            ldos2.save(arma::hdf5_name(dir_realis + info + ".hdf5", "LDOS2", arma::hdf5_opts::append));
            energy_density2.save(arma::hdf5_name(dir_realis + info + ".hdf5", "energy_density2", arma::hdf5_opts::append));
			
            
            x = arma::real(diag_mat_elem0);  x.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "diag_mat0",   arma::hdf5_opts::append));
            x = arma::imag(diag_mat_elem0);  x.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "diag_mat0_im",   arma::hdf5_opts::append));
            _integrated_spectral_fun0.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "integrated_spectral_fun0",   arma::hdf5_opts::append));
			_spectral_fun0.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "spectral_fun0",   arma::hdf5_opts::append));
			_spectral_fun_typ0.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "log(_spectral_fun_typ)0",   arma::hdf5_opts::append));
			_element_count0.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "element_count0",   arma::hdf5_opts::append));
            _spectral_fun_all0.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "spectral_fun_all0",   arma::hdf5_opts::append));
            _element_count_all0.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "element_count_all0",   arma::hdf5_opts::append));
		}
		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start_re) << " s - - - - - - " << std::endl; // simulation end
	}
}

void ui::gec_moments_calculation(){
    size_t dim = this->ptr_to_model->get_hilbert_size();
	auto _hilbert = this->ptr_to_model->get_model_ref().get_hilbert_space();

	int Ll = this->L;
	auto kernel = [Ll](u64 state) -> std::pair<u64, double>
	{ 
		auto [val1, state_z] = operators::sigma_z<double>(state, Ll, Ll / 2 );
		return std::make_pair(state_z, val1);
	};
	auto _operator = QOps::generic_operator<double>(this->L, std::move(kernel), 1.0);
	auto op_mat_Szell = _operator.to_reduced_matrix(_hilbert);
	// std::cout << op_mat_Szell << std::endl;

	auto kernel1 = [Ll](u64 state) -> std::pair<u64, double>
	{ 
		auto [val1, state_z] = operators::sigma_z<double>(state, Ll, Ll / 2 );
		return std::make_pair(state_z, (0.5 + val1));
	};
	auto _operator2 = QOps::generic_operator<double>(this->L, std::move(kernel1), 1.0);
	auto op_mat_nell = _operator2.to_reduced_matrix(_hilbert);
	// std::cout << op_mat_nell << std::endl;

    arma::vec wx_vals = arma::linspace(0, 5, 51);
    // for(auto wx : wx_vals){
    //     this->w = wx;
    arma::mat H2ii(this->realisations, 3, arma::fill::zeros);
    arma::mat autocorr_earlytime_Sz(this->realisations, 3, arma::fill::zeros);
    arma::mat autocorr_earlytime_n(this->realisations, 3, arma::fill::zeros);
    // arma::vec cross_term_Sz(this->realisations, 3, arma::fill::zeros);
    // arma::vec cross_term_n(this->realisations, 3, arma::fill::zeros);

    arma::mat rescaled_H2ii(this->realisations, 3, arma::fill::zeros);
    arma::mat rescaled_autocorr_earlytime_Sz(this->realisations, 3, arma::fill::zeros);
    arma::mat rescaled_autocorr_earlytime_n(this->realisations, 3, arma::fill::zeros);
    // arma::vec rescaled_cross_term_Sz(this->realisations, 3, arma::fill::zeros);
    // arma::vec rescaled_cross_term_n(this->realisations, 3, arma::fill::zeros);

    arma::mat gec(this->realisations, 3, arma::fill::zeros);
    arma::mat gec_H2ii(this->realisations, 3, arma::fill::zeros);
    arma::mat gec_Hii2(this->realisations, 3, arma::fill::zeros);
    arma::mat rescaled_gec(this->realisations, 3, arma::fill::zeros);
    arma::mat rescaled_gec_H2ii(this->realisations, 3, arma::fill::zeros);
    arma::mat rescaled_gec_Hii2(this->realisations, 3, arma::fill::zeros);
#pragma omp parallel for num_threads(outer_threads)
    for(int r = 0; r < this->realisations; r++)
    {
        // start = std::chrono::system_clock::now();
        clk::time_point start0 = std::chrono::system_clock::now();
        {
            this->seed = std::random_device{}();
            // this->reset_model_pointer();
            auto point = std::make_unique<QHS::QHamSolver<XXZ>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->hz, this->syms.Sz, this->add_parity_breaking, this->w, this->seed);
            
            arma::sp_mat H = point->get_hamiltonian();
            arma::sp_mat H2 = H*H;
            double meanH = arma::trace(H) / double(dim);
            double varH = arma::trace(H2) / double(dim) - meanH * meanH;
            
        // #pragma omp parallel for
            for(u64 k = 0; k < dim; k++){
                H2ii(r, 0) += H2(k,k) / double(dim);
                H2ii(r, 1) += H2(k,k) * H2(k,k) / double(dim);
                if(H2(k,k) > 0) H2ii(r, 2) += std::log(H2(k,k)) / double(dim);
                double cross_term_Sz = 0;
                double cross_term_n = 0;
                for(u64 m = 0; m < dim; m++){
                    cross_term_Sz += op_mat_Szell(k,k) * op_mat_Szell(m,m) * H(k,m) * H(m, k);
                    cross_term_n += op_mat_nell(k,k) * op_mat_nell(m,m) * H(k,m) * H(m, k);
                }
                double tau_Sz = op_mat_Szell(k,k) * op_mat_Szell(k,k) * H2(k,k) - cross_term_Sz;
                autocorr_earlytime_Sz(r, 0) += tau_Sz / double(dim);
                autocorr_earlytime_Sz(r, 1) += tau_Sz * tau_Sz / double(dim);
                if(tau_Sz > 0) autocorr_earlytime_Sz(r, 2) += std::log( tau_Sz ) / double(dim);
                
                // autocorr_earlytime_n(r, k) = op_mat_nell(k,k) * op_mat_nell(k,k) * H2(k,k) - cross_term_n;
                double tau_n = op_mat_nell(k,k) * op_mat_nell(k,k) * H2(k,k) - cross_term_n;
                autocorr_earlytime_n(r, 0) += tau_n / double(dim);
                autocorr_earlytime_n(r, 1) += tau_n * tau_n / double(dim);
                if(tau_n > 0) autocorr_earlytime_n(r, 2) += std::log( tau_n ) / double(dim);

                double _gec1 = (2 * H2(k,k) - H(k,k) * H(k,k) ) / varH;
                gec(r, 0) += _gec1 / double(dim);
                gec(r, 1) += _gec1 * _gec1 / double(dim);
                if(_gec1 > 0) gec(r, 2) += std::log(_gec1) / double(dim);

                _gec1 = H2(k,k) / varH;
                gec_H2ii(r, 0) += _gec1 / double(dim);
                gec_H2ii(r, 1) += _gec1 * _gec1 / double(dim);
                if(_gec1 > 0) gec_H2ii(r, 2) += std::log(_gec1) / double(dim);

                _gec1 = H(k,k) * H(k,k) / varH;
                gec_Hii2(r, 0) += _gec1 / double(dim);
                gec_Hii2(r, 1) += _gec1 * _gec1 / double(dim);
                if(_gec1 > 0) gec_Hii2(r, 2) += std::log(_gec1) / double(dim);
                // gec(r, k) = (2 * H2(k,k) - H(k,k) * H(k,k) ) / varH;
                // gec_H2ii(r, k) = H2(k,k) / varH;
                // gec_Hii2(r, k) = H(k,k) * H(k,k) / varH;
            }

            varH = arma::trace(H2) / double(dim) - meanH * meanH;
            H = (H - meanH * arma::eye<arma::sp_mat>(dim, dim)) / std::sqrt(varH);
            H2 = H*H;

        // #pragma omp parallel for
            for(u64 k = 0; k < dim; k++){
                // rescaled_H2ii(r, k) = H2(k,k);
                // rescaled_cross_term_Sz(r, k) = 0;
                // rescaled_cross_term_n(r, k) = 0;
                // for(u64 m = 0; m < dim; m++){
                //     rescaled_cross_term_Sz(r, k) += op_mat_Szell(k,k) * op_mat_Szell(m,m) * H(k,m) * H(m, k);
                //     rescaled_cross_term_n(r, k) += op_mat_nell(k,k) * op_mat_nell(m,m) * H(k,m) * H(m, k);
                // }
                // rescaled_autocorr_earlytime_Sz(r, k) = op_mat_Szell(k,k) * op_mat_Szell(k,k) * H2(k,k) - rescaled_cross_term_Sz(r, k);
                // rescaled_autocorr_earlytime_n(r, k) = op_mat_nell(k,k) * op_mat_nell(k,k) * H2(k,k) - rescaled_cross_term_n(r, k);

                // rescaled_gec(r, k) = (2 * H2(k,k) - H(k,k) * H(k,k) );
                // rescaled_gec_H2ii(r, k) = H2(k,k);
                // rescaled_gec_Hii2(r, k) = H(k,k) * H(k,k);

                rescaled_H2ii(r, 0) += H2(k,k) / double(dim);
                rescaled_H2ii(r, 1) += H2(k,k) * H2(k,k) / double(dim);
                if(H2(k,k) > 0) rescaled_H2ii(r, 2) += std::log(H2(k,k)) / double(dim);
                
                double cross_term_Sz = 0;
                double cross_term_n = 0;
                for(u64 m = 0; m < dim; m++){
                    cross_term_Sz += op_mat_Szell(k,k) * op_mat_Szell(m,m) * H(k,m) * H(m, k);
                    cross_term_n += op_mat_nell(k,k) * op_mat_nell(m,m) * H(k,m) * H(m, k);
                }
                double tau_Sz = op_mat_Szell(k,k) * op_mat_Szell(k,k) * H2(k,k) - cross_term_Sz;
                rescaled_autocorr_earlytime_Sz(r, 0) += tau_Sz / double(dim);
                rescaled_autocorr_earlytime_Sz(r, 1) += tau_Sz * tau_Sz / double(dim);
                if(tau_Sz > 0) rescaled_autocorr_earlytime_Sz(r, 2) += std::log( tau_Sz ) / double(dim);
                
                // autocorr_earlytime_n(r, k) = op_mat_nell(k,k) * op_mat_nell(k,k) * H2(k,k) - cross_term_n;
                double tau_n = op_mat_nell(k,k) * op_mat_nell(k,k) * H2(k,k) - cross_term_n;
                rescaled_autocorr_earlytime_n(r, 0) += tau_n / double(dim);
                rescaled_autocorr_earlytime_n(r, 1) += tau_n * tau_n / double(dim);
                if(tau_n > 0) rescaled_autocorr_earlytime_n(r, 2) += std::log( tau_n ) / double(dim);

                double _gec1 = (2 * H2(k,k) - H(k,k) * H(k,k) );
                rescaled_gec(r, 0) += _gec1 / double(dim);
                rescaled_gec(r, 1) += _gec1 * _gec1 / double(dim);
                if(_gec1 > 0) rescaled_gec(r, 2) += std::log(_gec1) / double(dim);

                _gec1 = H2(k,k);
                rescaled_gec_H2ii(r, 0) += _gec1 / double(dim);
                rescaled_gec_H2ii(r, 1) += _gec1 * _gec1 / double(dim);
                if(_gec1 > 0) rescaled_gec_H2ii(r, 2) += std::log(_gec1) / double(dim);

                _gec1 = H(k,k) * H(k,k);
                rescaled_gec_Hii2(r, 0) += _gec1 / double(dim);
                rescaled_gec_Hii2(r, 1) += _gec1 * _gec1 / double(dim);
                if(_gec1 > 0) rescaled_gec_Hii2(r, 2) += std::log(_gec1) / double(dim);
            }
        }
    #pragma omp critical
        std::cout << " - - - - - - finished realization r=" << r << "\t in :" << tim_s(start0) << " seconds - - - - - - " << std::endl; // simulation end
    }
    // av /= double(this->realisations);
    // av2 /= double(this->realisations);
    // var /= double(this->realisations);
    // var2 /= double(this->realisations);
    // H_trace /= double(this->realisations);
    // H_trace2 /= double(this->realisations);

    // av2 = av2 / H_trace2;
    // var2 = var2 / arma::square(H_trace2);
    
    std::string dir = this->saving_dir + "GEC_data_Testing_moms/" + kPSep;
    createDirs(dir);
    std::string info = "_L=" + std::to_string(this->L) + "_w=" + to_string_prec(this->w) + "_id=" + std::to_string(this->jobid);
    gec.save(	  arma::hdf5_name(dir + info + ".hdf5", "GEC"));
    gec_H2ii.save(	  arma::hdf5_name(dir + info + ".hdf5", "GEC_H2ii", arma::hdf5_opts::append));
    gec_Hii2.save(	  arma::hdf5_name(dir + info + ".hdf5", "GEC_Hii2", arma::hdf5_opts::append));
    rescaled_gec.save(	  arma::hdf5_name(dir + info + ".hdf5", "rescaled_GEC", arma::hdf5_opts::append));
    rescaled_gec_H2ii.save(	  arma::hdf5_name(dir + info + ".hdf5", "rescaled_GEC_H2ii", arma::hdf5_opts::append));
    rescaled_gec_Hii2.save(	  arma::hdf5_name(dir + info + ".hdf5", "rescaled_GEC_Hii2", arma::hdf5_opts::append));

    H2ii.save(	  arma::hdf5_name(dir + info + ".hdf5", "H2ii", arma::hdf5_opts::append));
    autocorr_earlytime_Sz.save(	  arma::hdf5_name(dir + info + ".hdf5", "autocorr_earlytime_Sz", arma::hdf5_opts::append));
    // cross_term_Sz.save(	  arma::hdf5_name(dir + info + ".hdf5", "cross_term_Sz", arma::hdf5_opts::append));
    autocorr_earlytime_n.save(	  arma::hdf5_name(dir + info + ".hdf5", "autocorr_earlytime_n", arma::hdf5_opts::append));
    // cross_term_n.save(	  arma::hdf5_name(dir + info + ".hdf5", "cross_term_n", arma::hdf5_opts::append));
    
    rescaled_H2ii.save(	  arma::hdf5_name(dir + info + ".hdf5", "rescaled_H2ii", arma::hdf5_opts::append));
    rescaled_autocorr_earlytime_Sz.save(	  arma::hdf5_name(dir + info + ".hdf5", "rescaled_autocorr_earlytime_Sz", arma::hdf5_opts::append));
    // rescaled_cross_term_Sz.save(	  arma::hdf5_name(dir + info + ".hdf5", "rescaled_cross_term_Sz", arma::hdf5_opts::append));
    rescaled_autocorr_earlytime_n.save(	  arma::hdf5_name(dir + info + ".hdf5", "rescaled_autocorr_earlytime_n", arma::hdf5_opts::append));
    // rescaled_cross_term_n.save(	  arma::hdf5_name(dir + info + ".hdf5", "rescaled_cross_term_n", arma::hdf5_opts::append));

// }
return;
}

void ui::long_time_prediction(){
    std::string dir = this->saving_dir + "GEC_AND_LONGTIME" + kPSep;
	createDirs(dir);
    const size_t dim_max = 1e5;
	size_t dim = this->ptr_to_model->get_hilbert_size();
    auto _hilbert = this->ptr_to_model->get_model_ref().get_hilbert_space();
    std::string info = this->set_info();

	int Ll = this->L;
	auto kernel = [Ll](u64 state) -> std::pair<u64, double>
	{ 
		auto [val1, state_z] = operators::sigma_z<double>(state, Ll, Ll / 2 );
		return std::make_pair(state_z, val1);
	};
	auto _operator = QOps::generic_operator<double>(this->L, std::move(kernel), 1.0);
	auto op_mat_Szell = _operator.to_reduced_matrix(_hilbert);

	auto kernel1 = [Ll](u64 state) -> std::pair<u64, double>
	{ 
		auto [val1, state_z] = operators::sigma_z<double>(state, Ll, Ll / 2 );
		return std::make_pair(state_z, (0.5 + val1));
	};
	auto _operator2 = QOps::generic_operator<double>(this->L, std::move(kernel1), 1.0);
	auto op_mat_nell = _operator2.to_reduced_matrix(_hilbert);

    arma::vec disord(dim);
    arma::vec unperturbed(dim);
    arma::vec h_ell = this->ptr_to_model->get_model_ref()._disorder;
    auto check_spin = QOps::__builtins::get_digit(this->L);
    for (u64 k = 0; k < dim; k++) 
    {
		double s_i, s_j;
		u64 base_state = _hilbert(k);
		for (int j = 0; j < this->L; j++) 
        {
            // Disorder terms
			s_i = check_spin(base_state, j) ? 0.5 : -0.5;
            disord(k) += s_i * h_ell(j);
            
            // Diagonal terms
            int nei = j + 1;
            if(nei >= this->L)
                nei = (this->boundary_conditions > 0)? -1 : nei % this->L;
            s_j = check_spin(base_state, nei) ? 0.5 : -0.5;
            unperturbed(k) += s_i * h_ell(j) + this->delta1 * s_i * s_j;
		}
	}
    
    for(int r = 0; r < this->realisations; r++)
    {
        // start = std::chrono::system_clock::now();
        clk::time_point start0 = std::chrono::system_clock::now();
        if(r > 0)
			this->ptr_to_model->generate_hamiltonian();
		
        clk::time_point start = std::chrono::system_clock::now();
        this->ptr_to_model->diagonalization();
		
		const arma::vec E = this->ptr_to_model->get_eigenvalues();
		const auto& V = this->ptr_to_model->get_eigenvectors();
        // std::cout << E << std::endl;
		std::cout << " - - - - - - finished diagonalization in : " << tim_s(start) << " s for realis = " << r << " - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();
        
        arma::sp_mat H = this->ptr_to_model->get_hamiltonian();
        arma::sp_mat H2 = H*H;
	    // std::cout << dim << "\t\t" << op_mat_Szell.n_cols << "\t\t" << H.n_cols << "\t\t" << H2.n_cols << std::endl;
        double meanH = arma::trace(H) / double(dim);
        double varH = arma::trace(H2) / double(dim) - meanH * meanH;
                
        arma::vec Hii(dim, arma::fill::zeros);
        arma::vec H2ii(dim, arma::fill::zeros);
        arma::vec autocorr_earlytime_Sz(dim, arma::fill::zeros);
        arma::vec autocorr_earlytime_n(dim, arma::fill::zeros);
        arma::vec autocorr_earlytime_pert(dim, arma::fill::zeros);
        arma::vec autocorr_earlytime_unpert(dim, arma::fill::zeros);

        arma::vec rescaled_Hii(dim, arma::fill::zeros);
        arma::vec rescaled_H2ii(dim, arma::fill::zeros);
        arma::vec rescaled_autocorr_earlytime_Sz(dim, arma::fill::zeros);
        arma::vec rescaled_autocorr_earlytime_n(dim, arma::fill::zeros);
        arma::vec rescaled_autocorr_earlytime_pert(dim, arma::fill::zeros);
        arma::vec rescaled_autocorr_earlytime_unpert(dim, arma::fill::zeros);

        arma::vec gec(dim, arma::fill::zeros);
        arma::vec rescaled_gec(dim, arma::fill::zeros);
    #pragma omp parallel for
        for(u64 k = 0; k < dim; k++){
            // std:: cout << "1\t\t" << k << std::endl;
            Hii(k) = H(k,k);
            H2ii(k) = H2(k,k);
            double cross_term_unpert = 0;
            double cross_term_pert = 0;
            double cross_term_Sz = 0;
            double cross_term_n = 0;
            for(u64 m = 0; m < dim; m++){
                cross_term_unpert += unperturbed(k) * unperturbed(m) * H(k,m) * H(m, k);
                cross_term_pert += disord(k) * disord(m) * H(k,m) * H(m, k);
                cross_term_Sz += op_mat_Szell(k,k) * op_mat_Szell(m,m) * H(k,m) * H(m, k);
                cross_term_n += op_mat_nell(k,k) * op_mat_nell(m,m) * H(k,m) * H(m, k);
            }
            autocorr_earlytime_unpert(k) = unperturbed(k) * unperturbed(k) * H2(k,k) - cross_term_unpert;
            autocorr_earlytime_pert(k) = disord(k) * disord(k) * H2(k,k) - cross_term_pert;
            autocorr_earlytime_Sz(k) = op_mat_Szell(k,k) * op_mat_Szell(k,k) * H2(k,k) - cross_term_Sz;
            autocorr_earlytime_n(k) = op_mat_nell(k,k) * op_mat_nell(k,k) * H2(k,k) - cross_term_n;
            
            gec(k) = (2 * H2(k,k) - H(k,k) * H(k,k) ) / varH;
        }

        varH = arma::trace(H2) / double(dim) - meanH * meanH;
        H = (H - meanH * arma::eye<arma::sp_mat>(dim, dim)) / std::sqrt(varH);
        H2 = H*H;

    #pragma omp parallel for
        for(u64 k = 0; k < dim; k++){
            // std:: cout << "2\t\t" << k << std::endl;
            rescaled_Hii(k) = H(k,k);
            rescaled_H2ii(k) = H2(k,k);
            
            double cross_term_unpert = 0;
            double cross_term_pert = 0;
            double cross_term_Sz = 0;
            double cross_term_n = 0;
            for(u64 m = 0; m < dim; m++){
                cross_term_unpert += unperturbed(k) * unperturbed(m) * H(k,m) * H(m, k);
                cross_term_pert += disord(k) * disord(m) * H(k,m) * H(m, k);
                cross_term_Sz += op_mat_Szell(k,k) * op_mat_Szell(m,m) * H(k,m) * H(m, k);
                cross_term_n += op_mat_nell(k,k) * op_mat_nell(m,m) * H(k,m) * H(m, k);
            }
            rescaled_autocorr_earlytime_unpert(k) = unperturbed(k) * unperturbed(k) * H2(k,k) - cross_term_unpert;
            rescaled_autocorr_earlytime_pert(k) = disord(k) * disord(k) * H2(k,k) - cross_term_pert;
            rescaled_autocorr_earlytime_Sz(k) = op_mat_Szell(k,k) * op_mat_Szell(k,k) * H2(k,k) - cross_term_Sz;
            rescaled_autocorr_earlytime_n(k) = op_mat_nell(k,k) * op_mat_nell(k,k) * H2(k,k) - cross_term_n;
            
            rescaled_gec(k) = (2 * H2(k,k) - H(k,k) * H(k,k) );
        }
        std::cout << " - - - - - - finished GEC and short-time expansion in : " << tim_s(start) << " s for realis = " << r << " - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();

        arma::vec diagonal_ensemble_n(dim, arma::fill::zeros);
        arma::vec diagonal_ensemble_Sz(dim, arma::fill::zeros);
        arma::vec diagonal_ensemble_pert(dim, arma::fill::zeros);
        arma::vec diagonal_ensemble_unpert(dim, arma::fill::zeros);

        // Diagonal elements of O in the product basis
        arma::vec nL_2_product(dim, arma::fill::zeros);
        arma::vec SzL_2_product(dim, arma::fill::zeros);
        for(long j = 0; j < dim; j++){
            nL_2_product(j) = std::real( double(op_mat_nell(j, j)) );
            SzL_2_product(j) = std::real( double(op_mat_Szell(j, j)) );
        }

        // O_{alpha,alpha} = sum_j |V_{j,alpha}|^2 O_j
        arma::vec nL_2_eigen(dim, arma::fill::zeros);
        arma::vec SzL_2_eigen(dim, arma::fill::zeros);
        arma::vec pert_2_eigen(dim, arma::fill::zeros);
        arma::vec H0_2_eigen(dim, arma::fill::zeros);
        for(long alpha = 0; alpha < dim; alpha++){
            nL_2_eigen(alpha) = arma::dot(arma::square(arma::abs(V.col(alpha))), nL_2_product);
            SzL_2_eigen(alpha) = arma::dot(arma::square(arma::abs(V.col(alpha))), SzL_2_product);
            pert_2_eigen(alpha) = arma::dot(arma::square(arma::abs(V.col(alpha))), disord);
            H0_2_eigen(alpha) = arma::dot(arma::square(arma::abs(V.col(alpha))), unperturbed);
        }
        

        // C_DE(i) = O_i * sum_alpha |V_{i,alpha}|^2 O_{alpha,alpha}
        for(long i = 0; i < dim; i++)
        {
            diagonal_ensemble_n(i) = nL_2_product(i) * arma::dot( arma::square(arma::abs(V.row(i))).t(), nL_2_eigen );
            diagonal_ensemble_Sz(i) = SzL_2_product(i) * arma::dot( arma::square(arma::abs(V.row(i))).t(), SzL_2_eigen );
            diagonal_ensemble_pert(i) = disord(i) * arma::dot( arma::square(arma::abs(V.row(i))).t(), pert_2_eigen );
            diagonal_ensemble_unpert(i) = unperturbed(i) * arma::dot( arma::square(arma::abs(V.row(i))).t(), H0_2_eigen );
        }

        std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + r) + kPSep;
        createDirs(dir_realis);

        gec.save(	        arma::hdf5_name(dir_realis + info + ".hdf5", "GEC"));
        rescaled_gec.save(	arma::hdf5_name(dir_realis + info + ".hdf5", "rescaled_GEC", arma::hdf5_opts::append));

        Hii.save(	                  arma::hdf5_name(dir_realis + info + ".hdf5", "Hii", arma::hdf5_opts::append));
        H2ii.save(	                  arma::hdf5_name(dir_realis + info + ".hdf5", "H2ii", arma::hdf5_opts::append));
        autocorr_earlytime_Sz.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "Sz_L_2/autocorr_earlytime_Sz", arma::hdf5_opts::append));
        autocorr_earlytime_n.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "n_L_2/autocorr_earlytime", arma::hdf5_opts::append));
        autocorr_earlytime_pert.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "perturbation/autocorr_earlytime_pert", arma::hdf5_opts::append));
        autocorr_earlytime_unpert.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "H0/autocorr_earlytime_unpert", arma::hdf5_opts::append));
        
        rescaled_Hii.save(	                arma::hdf5_name(dir_realis + info + ".hdf5", "rescaled_Hii", arma::hdf5_opts::append));
        rescaled_H2ii.save(	                arma::hdf5_name(dir_realis + info + ".hdf5", "rescaled_H2ii", arma::hdf5_opts::append));
        rescaled_autocorr_earlytime_Sz.save(arma::hdf5_name(dir_realis + info + ".hdf5", "Sz_L_2/rescaled_autocorr_earlytime", arma::hdf5_opts::append));
        rescaled_autocorr_earlytime_n.save( arma::hdf5_name(dir_realis + info + ".hdf5", "n_L_2/rescaled_autocorr_earlytime", arma::hdf5_opts::append));
        rescaled_autocorr_earlytime_pert.save(arma::hdf5_name(dir_realis + info + ".hdf5", "perturbation/rescaled_autocorr_earlytime_pert", arma::hdf5_opts::append));
        rescaled_autocorr_earlytime_unpert.save( arma::hdf5_name(dir_realis + info + ".hdf5", "H0/rescaled_autocorr_earlytime_unpert", arma::hdf5_opts::append));

        E.save( arma::hdf5_name(dir_realis + info + ".hdf5", "E", arma::hdf5_opts::append));
        diagonal_ensemble_n.save( arma::hdf5_name(dir_realis + info + ".hdf5", "n_L_2/C_DE(i)", arma::hdf5_opts::append));
        diagonal_ensemble_Sz.save( arma::hdf5_name(dir_realis + info + ".hdf5", "Sz_L_2/C_DE(i)", arma::hdf5_opts::append));
        diagonal_ensemble_pert.save( arma::hdf5_name(dir_realis + info + ".hdf5", "perturbation/C_DE(i)", arma::hdf5_opts::append));
        diagonal_ensemble_unpert.save( arma::hdf5_name(dir_realis + info + ".hdf5", "H0/C_DE(i)", arma::hdf5_opts::append));
        
        h_ell.save( arma::hdf5_name(dir_realis + info + ".hdf5", "disorder", arma::hdf5_opts::append));
        std::cout << " - - - - - - finished realization r=" << r << "\t in :" << tim_s(start0) << " seconds - - - - - - " << std::endl; // simulation end
    }
}
// -------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------- MODEL DEPENDENT FUNCTIONS

// ------------------------------------------------ OVERRIDEN METHODS
arma::Col<ui::element_type> ui::cast_state(const arma::Col<ui::element_type>& state)
{
    auto U1sector = U1Hilbert(this->L, this->syms.Sz);
    arma::Col<ui::element_type> full_state(ULLPOW(this->L), arma::fill::zeros);
    for(int i = 0; i < U1sector.get_hilbert_space_size(); i++)
        full_state(U1sector(i)) = state(i);
    return full_state;
}

/// @brief 
/// @param skip 
/// @param sep 
/// @return 
std::string ui::set_info(std::vector<std::string> skip, std::string sep) const
{
        std::string name = "L=" + std::to_string(this->L) + \
            ",J1=" + to_string_prec(this->J1) + \
            ",J2=" + to_string_prec(this->J2) + \
            ",d1=" + to_string_prec(this->delta1) + \
            ",d2=" + to_string_prec(this->delta2) + \
            ",hz=" + to_string_prec(this->hz);
        #ifdef USE_SYMMETRIES
            if(this->boundary_conditions == 0)      name += ",k=" + std::to_string(this->syms.k_sym);
            if(this->k_real_sec(this->syms.k_sym))  name += ",p=" + std::to_string(this->syms.p_sym);
            if(this->use_flip_X())                  name += ",zx=" + std::to_string(this->syms.zx_sym);
            name += ",Sz=" + to_string_prec(this->syms.Sz, 1);
            // name += ",edge=" + std::to_string((int)this->add_edge_fields);
        #else
            name += ",w=" + to_string_prec(this->w) + \
                    ",pb=" + std::to_string((int)this->add_parity_breaking);
                    // ",edge=" + std::to_string((int)this->add_edge_fields) + 
        #endif

		auto tmp = split_str(name, ",");
		std::string tmp_str = sep;
		for (int i = 0; i < tmp.size(); i++) {
			bool save = true;
			for (auto& skip_param : skip)
			{
				// skip the element if we don't want it to be included in the info
				if (split_str(tmp[i], "=")[0] == skip_param)
					save = false;
			}
			if (save) tmp_str += tmp[i] + ",";
		}
		tmp_str.pop_back();
		return tmp_str;
}

/// @brief Compare energie spactra for full model and all symmetry sectors combined
void ui::compare_energies()
{                
    v_1d<double> Esym;
	v_1d<std::string> symms;
    auto kernel = [&](int k, int p, int zx)
    {
        auto symmetric_model = std::make_unique<QHS::QHamSolver<XXZsym>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->hz, k, p, zx, this->syms.Sz);
        if(symmetric_model->get_hilbert_size() > 0){
            symmetric_model->diagonalization(false);
            arma::vec E = symmetric_model->get_eigenvalues();
            
            Esym.insert(Esym.end(), std::make_move_iterator(E.begin()), std::make_move_iterator(E.end()));
            v_1d<std::string> temp_str = v_1d<std::string>(E.size(), "k=" + std::to_string(k) + ",p=" + to_string(p) + ",zx=" + to_string(zx));
            symms.insert(symms.end(), std::make_move_iterator(temp_str.begin()), std::make_move_iterator(temp_str.end()));
        }
    };
    loopSymmetrySectors(kernel);

    auto full_model = std::make_unique<QHS::QHamSolver<XXZ>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->hz, this->syms.Sz);
    full_model->diagonalization(false);
    arma::vec E_dis = full_model->get_eigenvalues();
    
    auto permut = sort_permutation(Esym, [](const double a, const double b)
								   { return a < b; });
	apply_permutation(Esym, permut);
	apply_permutation(symms, permut);
	std::cout << std::endl << Esym.size() << std::endl << E_dis.size() << std::endl;
	printSeparated(std::cout, "\t", 20, true, "symmetry sector", "Energy sym", "Energy total", "difference");
	for (int k = 0; k < min((int)E_dis.size(), (int)Esym.size()); k++)
        if(std::abs(Esym[k] - E_dis(k)) > 1e-14)
		    printSeparated(std::cout, "\t", 20, true, symms[k], Esym[k], E_dis(k), Esym[k] - E_dis(k));
}

/// @brief Compaer full hamiltonian to the reconstructed one from symmetry sectors
void ui::compare_hamiltonian()
{   
    auto full_model = std::make_unique<QHS::QHamSolver<XXZ>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->hz, this->syms.Sz);
    arma::SpMat<ui::element_type> Hfull = full_model->get_hamiltonian();
    const u64 dim = full_model->get_hilbert_size();
    arma::sp_cx_mat H(dim, dim);
    auto kernel = [&](int k, int p, int zx)
    {
        auto symmetric_model = std::make_unique<QHS::QHamSolver<XXZsym>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->hz, k, p, zx, this->syms.Sz);
        auto U = symmetric_model->get_model_ref().get_hilbert_space().symmetry_rotation();
        arma::sp_cx_mat Hsym = cast_cx_sparse(symmetric_model->get_hamiltonian());
        H += U * Hsym * U.t();
    };
    loopSymmetrySectors(kernel);
    arma::sp_cx_mat res = cast_cx_sparse(Hfull) - cast_cx_sparse(H);
	printSeparated(std::cout, "\t", 20, true, "col", "row", "diff", "\t", "sym H");
    for(int i = 0; i < dim; i++){
        for(int j = 0; j < dim; j++){
            cpx val = res(i, j);
            if(std::abs(val) > 1e-14)
                printSeparated(std::cout, "\t", 15, true, i, j, val, "\t", H(i,j));
        }
    }
}

/// @brief 
void ui::check_symmetry_generators()
{
    v_1d<QOps::genOp> sym_group;
    // parity symmetry
    sym_group.emplace_back(QOps::_parity_symmetry(this->L, this->syms.p_sym));

    // spin flips
    if(this->hz == 0 && this->syms.Sz == 0.0)
        sym_group.emplace_back(QOps::_spin_flip_x_symmetry(this->L, this->syms.zx_sym));
    
    QHS::point_symmetric hilbert( this->L, sym_group, this->boundary_conditions, this->syms.k_sym, 0);
    auto group = hilbert.get_symmetry_group();
    for(auto& idx : {1, 130, 33, 71, 756}){
        for(auto& G : group){
            auto [state, val] = G(idx);
            printSeparated(std::cout, "\t", 16, true, std::vector<bool>(this->L, idx), std::vector<bool>(this->L, state), val);
        }
        std::cout << std::endl;
    }
}

/// @brief Create energy current for this specific model
arma::sp_cx_mat ui::spin_current(){

    const size_t dim_max = ULLPOW(this->L);
    auto check_spin = QOps::__builtins::get_digit(this->L);
    
    arma::sp_cx_mat js(dim_max, dim_max);
    for(int i = 0; i < this->L; i++)
    {
        int nei = (this->boundary_conditions)? i + 1 : (i + 1)%this->L;
        // printSeparated(std::cout, "\t", 20, true, "site", i, nei, nei2);
        if(nei < this->L){
            for(long k = 0; k < dim_max; k++)
            {
                double Si = double(check_spin(k, i)) - 0.5;
                double Snei = double(check_spin(k, nei)) - 0.5;
                {
                    auto [val, state_tmp]   = operators::sigma_x<cpx>(k, this->L, i);
                    auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei);
                    js(new_idx, k) -= (val * val2);
                }{
                    auto [val, state_tmp]   = operators::sigma_x<cpx>(k, this->L, nei);
                    auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, i);
                    js(new_idx, k) += (val * val2);
                }
            }
        }
    }
    return js;
}

/// @brief Create energy current for this specific model
arma::sp_mat ui::energy_current(){

    const size_t dim_max = ULLPOW(this->L);
    auto check_spin = QOps::__builtins::get_digit(this->L);
    // _assert_(this->J2 == 0.0 && this->delta2 == 0, "Energy current implemented only for integrable case, no nearest neighbour terms yet!");
    double Jx = this->J1;
    double Jy = this->J1;
    double Jz = this->delta1;
    arma::sp_mat jE(dim_max, dim_max);
    printSeparated(std::cout, "\t", 20, true, "Start Current", Jx, Jy, Jz);
    auto start = std::chrono::system_clock::now();
    for(int i = 0; i < this->L; i++)
    {
        int nei = (this->boundary_conditions)? i + 1 : (i + 1)%this->L;
        int nei2 = (this->boundary_conditions)? i + 2 : (i + 2)%this->L;
        // printSeparated(std::cout, "\t", 20, true, "site", i, nei, nei2);
        if(nei < this->L && nei2 < this->L){
            for(long k = 0; k < dim_max; k++)
            {
                double Si = double(check_spin(k, i)) - 0.5;
                double Snei = double(check_spin(k, nei)) - 0.5;
                double Snei2 = double(check_spin(k, nei2)) - 0.5;
                {
                    // + Sx Sz Sy
                    auto [val, state_tmp]   = operators::sigma_x<cpx>(k, this->L, i);
                    auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei2);
                    jE(new_idx, k) += std::imag(Jx * Jy * Snei * val * val2);
                }{
                    // - Sy Sz Sx
                    auto [val, state_tmp]   = operators::sigma_x<cpx>(k, this->L, nei2);
                    auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, i);
                    jE(new_idx, k) -= std::imag(Jx * Jy * Snei * val * val2);
                }{
                    // + Sy Sx Sz
                    auto [val, state_tmp]   = operators::sigma_x<cpx>(k, this->L, nei);
                    auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, i);
                    jE(new_idx, k) += std::imag(Jz * Jy * Snei2 * val * val2);
                }{
                    // - Sx Sy Sz
                    auto [val, state_tmp]   = operators::sigma_x<cpx>(k, this->L, i);
                    auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei);
                    jE(new_idx, k) -= std::imag(Jz * Jx * Snei2 * val * val2);
                }{
                    // Sz Sy Sx
                    auto [val, state_tmp]   = operators::sigma_x<cpx>(k, this->L, nei2);
                    auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei);
                    jE(new_idx, k) += std::imag(Jz * Jx * Si * val * val2);
                }{
                    // - Sz Sx Sy
                    auto [val, state_tmp]   = operators::sigma_x<cpx>(k, this->L, nei);
                    auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei2);
                    jE(new_idx, k) -= std::imag(Jz * Jy * Si * val * val2);
                }
            }
        }
    }
    std::cout << " - - - - - - finished energy current in : " << tim_s(start) << " s - - - - - - " << std::endl; // simulation end

    return jE / double(this->L);
}


/// @brief 
void ui::ErgodicNonDiffusive(){
    std::string dir = this->saving_dir + "ErgodicNonDiffusive" + kPSep;
	createDirs(dir);
	
    const size_t dim_max = 1e5;
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info();
	const size_t size = dim > dim_max? this->l_steps : dim;

	int Ll = this->L;
	int counter = 0;
	
	const double _bandwidth_def = std::sqrt(this->L);
	const double _tH = double(dim) / _bandwidth_def;
	
	const arma::vec omegax = arma::logspace(std::log10(1.0/dim) - 1, std::log10( _bandwidth_def ) + 1, 10 * this->L);
	const arma::vec energy_density = arma::regspace(0.05, 0.02, 0.95);
	
	arma::vec times = arma::logspace(-2, (std::log10(100 * _tH)), this->num_of_points);

	double window_width = 0.04;
	auto _hilbert_space = this->ptr_to_model->get_model_ref().get_hilbert_space();

#ifdef USE_SYMMETRIES
    int RRR  = 1;
#else
    int RRR = this->realisations;
#endif
	for(int realis = 0; realis < RRR; realis++)
	{
		clk::time_point start_re = std::chrono::system_clock::now();
		if(realis > 0)
			this->ptr_to_model->generate_hamiltonian();
		
		clk::time_point start = std::chrono::system_clock::now();
		if(dim > dim_max){
			this->ptr_to_model->diag_sparse(this->l_steps, this->l_bundle, this->tol, this->seed);	
		}
		else{
        	this->ptr_to_model->diagonalization();
		}
		std::cout << " - - - - - - finished diagonalization in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();
		
		const arma::vec E = this->ptr_to_model->get_eigenvalues();
		const auto& V = this->ptr_to_model->get_eigenvectors();
		double E_av = arma::trace(E) / double(dim);

		auto i = std::min_element(std::begin(E), std::end(E), [=](double x, double y) {
			return abs(x - E_av) < abs(y - E_av);
		});
		const long Eav_idx = i - std::begin(E);

        arma::Col<element_type> Hdiagonal = arma::diagvec( this->ptr_to_model->get_dense_hamiltonian() );

		auto i2 = min_element(begin(Hdiagonal), end(Hdiagonal), [=](element_type x, element_type y) {
			return std::abs(x - E_av) < std::abs(y - E_av);
		});
		const u64 idx = i2 - begin(Hdiagonal);
        
		long int E_min = dim > dim_max? 1 : Eav_idx - long(dim / 4);
		long int E_max = dim > dim_max? size-1 : Eav_idx + long(dim / 4);

		double wH = 0, r = 0, count = 0;
		for (long int i = E_min; i < E_max; i++){
            double dE1 = E(i+1) - E(i);
            double dE2 = E(i) - E(i-1);
			wH += E(i+1) - E(i);
            r += min(dE1, dE2) / max(dE1, dE2);
            count += 1;
        }
		wH /= double(count);
        r /= double(count);

		start = std::chrono::system_clock::now();
		
        #ifdef USE_SYMMETRIES
            std::string dir_realis = dir;
        #else
            std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
        #endif
        createDirs(dir_realis);
        E.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "energies"));
        omegax.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "omegas",   arma::hdf5_opts::append));
        energy_density.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "energy_density",   arma::hdf5_opts::append));
        arma::vec( {wH} ).save(   arma::hdf5_name(dir_realis + info + ".hdf5", "wH",   arma::hdf5_opts::append));
        arma::vec( {r} ).save(   arma::hdf5_name(dir_realis + info + ".hdf5", "gap_ratio",   arma::hdf5_opts::append));

        times.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "times",   arma::hdf5_opts::append));

        for(int q = 1; q <= this->L/2; q++)
        {
            auto _hilbert_space_sym_q = tensor(QHS::point_symmetric( this->L, v_1d<QOps::generic_operator<cpx>>(), 
                                            this->boundary_conditions, q, -1), U1Hilbert(this->L, this->syms.Sz) );
            // auto maps = _hilbert_space_sym_q.get_mapping();
            const std::size_t dim_sym = _hilbert_space_sym_q.get_hilbert_space_size();
            const double ksym = constants<double>::two_pi * q / double(this->L);

            QOps::generic_operator<cpx> T = QOps::_translation_symmetry<QOps::particle::boson>(this->L, q);
            arma::vec initial_state(dim, arma::fill::zeros);
            u64 state = _hilbert_space_sym_q(dim_sym/4);// ULLPOW(this->L / 2) - 1;// _hilbert_space(idx);
            cpx val = 1.0;
            for(int ell = 0; ell < this->L; ell++){
                u64 idx_in_basis = _hilbert_space.find(state);
                // initial_state(idx_in_basis) = std::real(val * std::exp(-1i * ksym));
                double phase = ksym * ell;
                initial_state(idx_in_basis) = std::cos(phase) / std::sqrt(double(this->L));
                std::tie(state, val) = T(state);
            }
            initial_state = arma::normalise(initial_state);
            // initial_state /= std::sqrt(this->L);
            std::cout << "Norm = " << arma::norm(initial_state) << std::endl;
            // std::cout << initial_state.t() << std::endl;
            std::cout << " - - - - - - finished state preparation for q = " << q << " in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
            start = std::chrono::system_clock::now();

            arma::sp_mat MomMode(dim, dim);
            auto check_spin = QOps::__builtins::get_digit(this->L);

            arma::vec exponents(this->L, arma::fill::zeros);
            for (int j = 0; j < this->L; j++)
                exponents(j) = std::cos(double(j) * constants<double>::two_pi * q / double(this->L));
        #pragma omp parallel for
            for (u64 k = 0; k < dim; k++) 
            {
                double s_i;
                u64 base_state = _hilbert_space(k);
                for (int j = 0; j < this->L; j++) 
                {
                    s_i = check_spin(base_state, j) ? 0.5 : -0.5;				// true - spin up, false - spin down
                    MomMode(k, k) += s_i * exponents(j);
                }
            }
            MomMode = MomMode * 1. / std::sqrt(this->L);
            double _operator_HSnorm = arma::trace(MomMode.t() * MomMode) / double(dim);
            MomMode = MomMode / std::sqrt(_operator_HSnorm);

            std::cout << "Hilbert-Schmidt Norm\t\t" << _operator_HSnorm << "\t\tNew Norm\t\t" << arma::trace(MomMode.t() * MomMode) / double(dim) << std::endl;

            arma::Mat<ui::element_type> mat_elem = V.t() * MomMode * V;
            double Sq_state = arma::dot(initial_state, MomMode * MomMode.t() * initial_state);
            
            std::cout << "⟨ψ_k|(S^z_q)²|ψ_k⟩ = " << Sq_state << std::endl;
            std::cout << " - - - - - - finished matrix elements of q = " << q << " in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
            start = std::chrono::system_clock::now();
            auto [_susc, _susc_r] = adiabatics::gauge_potential_save((mat_elem), E, this->L, wH);

            std::cout << " - - - - - - finished AGP in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
            start = std::chrono::system_clock::now();
            arma::mat _integrated_spectral_fun(omegax.size()-1, energy_density.size(), arma::fill::zeros);
            arma::mat _spectral_fun(omegax.size()-1, energy_density.size(), arma::fill::zeros);
            arma::mat _spectral_fun_typ(omegax.size()-1, energy_density.size(), arma::fill::zeros);
            arma::mat _element_count(omegax.size()-1, energy_density.size(), arma::fill::zeros);
            
            const double bandwidth = E(E.size() - 1) - E(0);
        #pragma omp parallel for
            for(int ii = 0; ii < energy_density.size(); ii++){
                const double eps = energy_density(ii);
                const double energyx = eps * bandwidth + E(0);
                spectrals::preset_omega set_omega(E, window_width, energyx);
                arma::vec omegas_i, matter;
                std::tie(omegas_i, matter) = set_omega.get_matrix_elements(mat_elem);
                for(int k = 0; k < omegax.size() - 1; k++){
                    arma::uvec indices = arma::find(omegas_i >= omegax[k] && omegas_i < omegax[k+1]);
                    if(indices.size() > 0){
                        _element_count(k, ii) = indices.size();
                        arma::vec x = arma::vec( omegas_i.elem(indices) );
                        arma::vec y = arma::vec( matter.elem(indices) );
                        _spectral_fun(k, ii) = arma::accu( y );
                        _spectral_fun_typ(k, ii) = arma::accu( arma::log(y) );
                    }
                    indices = arma::find(omegas_i < omegax[k+1]);
                    if(indices.size() > 0){
                        arma::vec y = arma::vec( matter.elem(indices) );
                        _integrated_spectral_fun(k, ii) = arma::accu(y);
                    }
                }
            }

            // Precompute overlap of initial state with eigenstates
            arma::vec coeffs(dim);
            for(int n = 0; n < E.size(); n++)
                coeffs(n) = arma::cdot(initial_state, this->ptr_to_model->get_eigenState(n));


            // Precompute α = M^T · β  (single O(D²) operation outside time loop)
            arma::vec alpha = mat_elem * coeffs;

            arma::vec _dynamical_structure_factor(omegax.size()-1, arma::fill::zeros);
            arma::vec _spectral_fun_all(omegax.size()-1, arma::fill::zeros);
            arma::vec _element_count_all(omegax.size()-1, arma::fill::zeros);
            const double dw_log = std::log10(omegax[1]) - std::log10(omegax[0]);
            const double w0_log = std::log10(omegax[0]);

            {
                for(int n = 0; n < E.size(); n++){
                    for(int m = n+1; m < E.size(); m++)
                    {
                        double wnm = E(m) - E(n);
                        const auto idx = int( (std::log10(wnm) - w0_log) / dw_log);
                        if(idx < omegax.size() && idx >= 0){
                            _spectral_fun_all(idx) += 2*std::abs(mat_elem(n, m) * mat_elem(m, n));
                            _element_count_all(idx) += 2;

                            double A_nm = coeffs(m) * mat_elem(m,n) * alpha(n);
                            double B_nm = coeffs(n) * mat_elem(n,m) * alpha(m);
                            _dynamical_structure_factor(idx) += A_nm + B_nm;
                        }
                    }	
                }
                std::cout << " - - - - - - finished spectral function at all energy density for realis = " << realis << " in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
            }
            std::cout << " - - - - - - finished \hat{V} matrix elements in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end


            arma::vec autocorr_inf(times.size(), arma::fill::zeros);
            arma::vec autocorr_psi(times.size(), arma::fill::zeros);
            arma::vec lohschmidt(times.size(), arma::fill::zeros);

            double DE = 0;
            double DE_psi = 0;
            double ipr = 0.0;
            // arma::vec quench(times.size(), arma::fill::zeros);
            if(dim < dim_max){
                start = std::chrono::system_clock::now();

                // Precompute diagonal and off-diagonal prefactors
                // to avoid redundant multiplications inside time loop
                int Ns = E.size();

                // Diagonal contributions (time-independent)
                double diag_inf = 0.0;
                double diag_psi = 0.0;
                for(int n = 0; n < Ns; n++){
                    diag_inf += mat_elem(n,n) * mat_elem(n,n);
                    diag_psi += coeffs(n) * mat_elem(n,n) * alpha(n);
                    ipr += std::abs(coeffs(n)*coeffs(n)) * std::abs(coeffs(n)*coeffs(n));
                }
                DE = diag_inf / double(dim);
                DE_psi = diag_psi;
                // Off-diagonal: precompute A_mn, B_mn, ω_mn  
                // Store as flat arrays for cache efficiency
                int n_pairs = Ns * (Ns - 1) / 2;
                arma::vec omega_nm(n_pairs);
                arma::vec coeff_nm(n_pairs);   // |α_m * α_n|^2
                arma::vec A_nm(n_pairs);   // β_m * M_{mn} * α_n
                arma::vec B_nm(n_pairs);   // β_n * M_{nm} * α_m
                arma::vec inf_nm(n_pairs); // M_{nm} * M_{mn}  for infinite T

                int idx = 0;
                for(int n = 0; n < Ns; n++){
                    for(int m = n+1; m < Ns; m++){
                        omega_nm(idx) = E(m) - E(n);
                        A_nm(idx)     = coeffs(m) * mat_elem(m,n) * alpha(n);
                        B_nm(idx)     = coeffs(n) * mat_elem(n,m) * alpha(m);
                        inf_nm(idx)   = mat_elem(n,m) * mat_elem(m,n);
                        coeff_nm(idx) = std::abs(coeffs(n)*coeffs(n)) * std::abs(coeffs(m)*coeffs(m));
                        idx++;
                    }
                }

            #pragma omp parallel for schedule(dynamic)
                for(long t_idx = 0; t_idx < times.size(); t_idx++){

                    double val_inf = diag_inf;
                    double val_psi = diag_psi;
                    double val_loh = ipr;

                    for(int p = 0; p < n_pairs; p++){
                        double cos_t = std::cos(omega_nm(p) * times(t_idx));

                        // Infinite temperature: 2 * M_{nm} M_{mn} * cos(ω t)
                        val_inf += 2.0 * inf_nm(p) * cos_t;

                        // Initial state: (A + B) * cos(ω t)
                        // [sin terms cancel when taking real part]
                        val_psi += (A_nm(p) + B_nm(p)) * cos_t;

                        val_loh += 2 * coeff_nm(p) * cos_t;
                    }

                    autocorr_inf(t_idx) = val_inf / double(dim);
                    autocorr_psi(t_idx) = val_psi;
                    lohschmidt(t_idx)   = val_loh;
                }
                autocorr_psi /= Sq_state;
                
                // // ---- t=0 DIAGNOSTIC ----
                // double C_t0_direct   = arma::dot(initial_state, MomMode * MomMode * initial_state);
                // double C_t0_eigenstate = diag_psi;
                // for(int p = 0; p < n_pairs; p++) 
                //     C_t0_eigenstate += A_nm(p) + B_nm(p);  // all cos(0) = 1

                // std::cout << "=== t=0 CHECK ===" << std::endl;
                // std::cout << "Direct ⟨ψ_k|(S^z_q)²|ψ_k⟩     = " << C_t0_direct    << std::endl;
                // std::cout << "Eigenstate sum C(q,0)      = " << C_t0_eigenstate << std::endl;
                // std::cout << "Sq_state (normalization)   = " << Sq_state        << std::endl;
                // std::cout << "Ratio C(q,0)/Sq_state      = " << C_t0_eigenstate / Sq_state << std::endl;
                // std::cout << "||coeffs||²                = " << arma::dot(coeffs, coeffs)  << std::endl;
                // std::cout << "M symmetry ||M-Mᵀ||        = " << arma::norm(mat_elem - mat_elem.t(), "fro") << std::endl;
                    // quench(t_idx) = std::real( arma::cdot(psi.col(t_idx), kinetic * psi.col(t_idx)) );
                
                std::cout << " - - - - - - finished time evolution for q = " << q << "  in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
            }
            omp_set_num_threads(this->thread_number);
            
            
            _integrated_spectral_fun.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "q=" + std::to_string(q) + "/integrated_spectral_fun", arma::hdf5_opts::append));
            _spectral_fun.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "q=" + std::to_string(q) + "/spectral_fun", arma::hdf5_opts::append));
            _spectral_fun_typ.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "q=" + std::to_string(q) + "/log(_spectral_fun_typ)", arma::hdf5_opts::append));
            _element_count.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "q=" + std::to_string(q) + "/element_count", arma::hdf5_opts::append));
            _spectral_fun_all.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "q=" + std::to_string(q) + "/spectral_fun_all", arma::hdf5_opts::append));
            _element_count_all.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "q=" + std::to_string(q) + "/element_count_all", arma::hdf5_opts::append));
            _dynamical_structure_factor.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "q=" + std::to_string(q) + "/dynamical_structure_factor", arma::hdf5_opts::append));

            _susc.save(	 arma::hdf5_name(dir_realis + info + ".hdf5", "q=" + std::to_string(q) + "/susc",     arma::hdf5_opts::append));
            _susc_r.save(arma::hdf5_name(dir_realis + info + ".hdf5", "q=" + std::to_string(q) + "/susc_reg", arma::hdf5_opts::append));

            lohschmidt.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "q=" + std::to_string(q) + "/lohschmidt",   arma::hdf5_opts::append));
            autocorr_inf.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "q=" + std::to_string(q) + "/autocorrelation",   arma::hdf5_opts::append));
            autocorr_psi.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "q=" + std::to_string(q) + "/quench_psi_q",   arma::hdf5_opts::append));
            arma::vec( {DE} ).save(   arma::hdf5_name(dir_realis + info + ".hdf5", "q=" + std::to_string(q) + "/DE",   arma::hdf5_opts::append));
            arma::vec( {DE_psi} ).save(   arma::hdf5_name(dir_realis + info + ".hdf5", "q=" + std::to_string(q) + "/DE_psi",   arma::hdf5_opts::append));
            arma::vec( {ipr} ).save(   arma::hdf5_name(dir_realis + info + ".hdf5", "q=" + std::to_string(q) + "/ipr",   arma::hdf5_opts::append));

        }
		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start_re) << " s - - - - - - " << std::endl; // simulation end
	}
}

/// @brief Calculate matrix element of energy current <state1|jE|state2> at site i and basis state k
/// @param state1 <state1| left-hand state in matrix element
/// @param state2 |state2> right-hand state in matrix element
/// @param i site
/// @param k basis state id
/// @param check_spin function to check current spin value
ui::element_type 
ui::jE_mat_elem_kernel(
            const arma::Col<element_type>& state1, 
            const arma::Col<element_type>& state2,
            int i, u64 k, const QOps::_ifun& check_spin
            )
{
    ui::element_type result = ui::element_type(0);
    _assert_(this->J2 == 0.0 && this->delta2 == 0, "Energy current implemented only for integrable case, no nearest neighbour terms yet!");
    double Jx = this->J1;
    double Jy = this->J1;
    double Jz = this->delta1;
   
    int nei = (this->boundary_conditions)? i + 1 : (i + 1)%this->L;
    int nei2 = (this->boundary_conditions)? i + 2 : (i + 2)%this->L;
    if(nei < this->L && nei2 < this->L){
        double Si = double(check_spin(k, i)) - 0.5;
        double Snei = double(check_spin(k, nei)) - 0.5;
        double Snei2 = double(check_spin(k, nei2)) - 0.5;
        {
            auto [val, state_tmp]   = operators::sigma_x<cpx>(k, this->L, i);
            auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei2);
            // jE(new_idx, k) += std::imag(Jx * Jy * Snei * val * val2);
            result += my_conjungate(state1(new_idx)) * std::imag(Jx * Jy * Snei * val * val2) * state2(k);
        }{
            auto [val, state_tmp]   = operators::sigma_x<cpx>(k, this->L, nei2);
            auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, i);
            // jE(new_idx, k) -= std::imag(Jx * Jy * Snei * val * val2);
            result -= my_conjungate(state1(new_idx)) * std::imag(Jx * Jy * Snei * val * val2) * state2(k);
        }{
            auto [val, state_tmp]   = operators::sigma_x<cpx>(k, this->L, nei);
            auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, i);
            // jE(new_idx, k) += std::imag(Jz * Jy * Snei2 * val * val2);
            result += my_conjungate(state1(new_idx)) * std::imag(Jz * Jy * Snei2 * val * val2) * state2(k);
        }{
            auto [val, state_tmp]   = operators::sigma_x<cpx>(k, this->L, i);
            auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei);
            // jE(new_idx, k) -= std::imag(Jz * Jx * Snei2 * val * val2);
            result -= my_conjungate(state1(new_idx)) * std::imag(Jz * Jx * Snei2 * val * val2) * state2(k);
        }{
            auto [val, state_tmp]   = operators::sigma_x<cpx>(k, this->L, nei2);
            auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei);
            // jE(new_idx, k) += std::imag(Jz * Jx * Si * val * val2);
            result += my_conjungate(state1(new_idx)) * std::imag(Jz * Jx * Si * val * val2) * state2(k);
        }{
            auto [val, state_tmp]   = operators::sigma_x<cpx>(k, this->L, nei);
            auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei2);
            // jE(new_idx, k) -= std::imag(Jz * Jy * Si * val * val2);
            result -= my_conjungate(state1(new_idx)) * std::imag(Jz * Jy * Si * val * val2) * state2(k);
        }
    }

    return result;
}

// -------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------- IMPLEMENTATION OF UI

/// @brief Create unique pointer to model with current parameters in class
typename ui::model_pointer ui::create_new_model_pointer(){
    #ifdef USE_SYMMETRIES
        return std::make_unique<QHS::QHamSolver<XXZsym>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->hz, this->syms.k_sym, this->syms.p_sym, this->syms.zx_sym, this->syms.Sz);
    #else
        return std::make_unique<QHS::QHamSolver<XXZ>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->hz, this->syms.Sz, this->add_parity_breaking, this->w, this->seed);
    #endif
}

/// @brief Reset member unique pointer to model with current parameters in class
void ui::reset_model_pointer(){
    #ifdef USE_SYMMETRIES
        return this->ptr_to_model.reset(new QHS::QHamSolver<XXZsym>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->hz, this->syms.k_sym, this->syms.p_sym, this->syms.zx_sym, this->syms.Sz));
    #else
        return this->ptr_to_model.reset(new QHS::QHamSolver<XXZ>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->hz, this->syms.Sz, this->add_parity_breaking, this->w, this->seed));
    #endif
}

/// @brief 
/// @param argc 
/// @param argv 
ui::ui(int argc, char **argv)
{
    auto input = change_input_to_vec_of_str(argc, argv);			// change standard input to vec of strings
	input = std::vector<std::string>(input.begin()++, input.end()); // skip the first element which is the name of file
	// plog::init(plog::info, "log.txt");						    // initialize logger
	
    if (std::string option = this->getCmdOption(input, "-f"); option != "")
	    input = this->parse_input_file(option); // parse input from file
	
	this->parse_cmd_options((int)input.size(), input); // parse input from CMD directly
}


/// @brief 
/// @param argc 
/// @param argv 
void ui::parse_cmd_options(int argc, std::vector<std::string> argv)
{
    //<! set all general UI parameters
    XXZUIparent::parse_cmd_options(argc, argv);

    //<! set the remaining UI parameters
	std::string choosen_option = "";	

	#define set_param(name) choosen_option = "-" #name;                                 \
	                        this->set_option(this->name, argv, choosen_option);         \
                                                                                        \
	                        choosen_option = "-" #name "s";                             \
	                        this->set_option(this->name##s, argv, choosen_option);      \
                                                                                        \
	                        choosen_option = "-" #name "n";                             \
	                        this->set_option(this->name##n, argv, choosen_option, true);
    set_param(J1);
    set_param(J2);
    set_param(delta1);
    set_param(delta2);
    set_param(hz);
    set_param(w);

    choosen_option = "-pb";
    this->set_option(this->add_parity_breaking, argv, choosen_option);

    // choosen_option = "-edge";
    // this->set_option(this->add_edge_fields, argv, choosen_option);

    //<! SYMMETRIES
    choosen_option = "-k";
    this->set_option(this->syms.k_sym, argv, choosen_option);

    choosen_option = "-p";
    this->set_option(this->syms.p_sym, argv, choosen_option);
    
    choosen_option = "-zx";
    this->set_option(this->syms.zx_sym, argv, choosen_option);
    
    choosen_option = "-Sz";
    this->set_option(this->syms.Sz, argv, choosen_option);
    if(this->L % 2 == 1 && this->syms.Sz == 0.0)
        this->syms.Sz = 0.5;

    //<! FOLDER
    #ifdef USE_EXP_COUPLING
        #ifdef ADD_CURRENT
        std::string folder = "." + kPSep + "results_fgr2" + kPSep;
        #else
        std::string folder = "." + kPSep + "results_fgr" + kPSep;
        #endif
    #else
        std::string folder = "." + kPSep + "results" + kPSep;
    #endif
    #ifdef USE_SYMMETRIES
        folder += "symmetries" + kPSep;
    #else
        folder += "disorder" + kPSep;
    #endif
    switch(this->boundary_conditions){
        case 0: folder += "PBC" + kPSep; break;
        case 1: folder += "OBC" + kPSep; break;
        case 2: folder += "ABC" + kPSep; break;
        default:
            folder += "PBC" + kPSep; 
            break;
        
    }

	folder = this->dir_prefix + folder;

    if (fs::create_directories(folder) || fs::is_directory(folder)) // creating the directory for saving the files with results
    	this->saving_dir = folder;									// if can create dir this is is
}


/// @brief 
void ui::set_default(){
    XXZUIparent::set_default();
    this->J1 = 1.0;
	this->J1s = 0.0;
	this->J1n = 1;
    this->J2 = 0.0;
	this->J2s = 0.0;
	this->J2n = 1;

    this->delta1 = 1.0;
	this->delta1s = 0.0;
	this->delta1n = 1;
    this->delta2 = 0.0;
	this->delta2s = 0.0;
	this->delta2n = 1;

    this->hz = 0.0;
	this->hzs = 0.0;
	this->hzn = 1;
	this->w = 0.0;
	this->ws = 0.0;
	this->wn = 1;


    // this->add_edge_fields = 0;
    this->add_parity_breaking = 0;

    this->syms.k_sym = 0;
    this->syms.p_sym = 1;
    this->syms.zx_sym = 1;
    this->syms.Sz = 0.0;
}

/// @brief 
void ui::print_help() const {
    XXZUIparent::print_help();

    printf(" Flags for XXZ model:\n");
    printSeparated(std::cout, "\t", 20, true, "-J1", "(double)", "nearest neighbour coupling strength");
    printSeparated(std::cout, "\t", 20, true, "-J1s", "(double)", "step in nearest neighbour coupling strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-J1n", "(int)", "number of nearest neighbour couplings in the sweep");
    printSeparated(std::cout, "\t", 20, true, "-J2", "(double)", "next-nearest neighbour coupling strength");
    printSeparated(std::cout, "\t", 20, true, "-J2s", "(double)", "step in next-nearest neighbour coupling strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-J2n", "(int)", "number of next-nearest neighbour couplings in the sweep");
    
    printSeparated(std::cout, "\t", 20, true, "-delta1", "(double)", "nearest neighbour interaction strength");
    printSeparated(std::cout, "\t", 20, true, "-delta1s", "(double)", "step in nearest neighbour interaction strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-delta1n", "(int)", "number of nearest neighbour interaction in the sweep");
    printSeparated(std::cout, "\t", 20, true, "-delta2", "(double)", "next-nearest neighbour interaction strength");
    printSeparated(std::cout, "\t", 20, true, "-delta2s", "(double)", "step in next-nearest neighbour interaction strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-delta2n", "(int)", "number of next-nearest neighbour interaction in the sweep");

    printSeparated(std::cout, "\t", 20, true, "-hz", "(double)", "uniform longitudinal field on spins");
    printSeparated(std::cout, "\t", 20, true, "-hzs", "(double)", "step in Z-field strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-hzn", "(int)", "number of Z-field values in the sweep");
    #ifdef USE_SYMMETRIES
        printSeparated(std::cout, "\t", 20, true, "-k", "(int)", "quasimomentum symmetry sector");
        printSeparated(std::cout, "\t", 20, true, "-p", "(int)", "parity symmetry sector");
        printSeparated(std::cout, "\t", 20, true, "-zx", "(int)", "spin flip in X direction symmetry sector");
        printSeparated(std::cout, "\t", 20, true, "-Sz", "(float)", "magnetization sector");

    #else
        printSeparated(std::cout, "\t", 20, true, "-w", "(double)", "disorder strength from uniform distribution");
        printSeparated(std::cout, "\t", 20, true, "-ws", "(double)", "step in disorder strength sweep");
        printSeparated(std::cout, "\t", 20, true, "-wn", "(int)", "number of disorder in the sweep");

        printSeparated(std::cout, "\t", 20, true, "-seed", "(u64)", "randomness in position for coupling to grain");
        // printSeparated(std::cout, "\t", 20, true, "-edge", "(boolean)", "add edge fields for SUSY (when no disorder, i.e. w=0)");
        printSeparated(std::cout, "\t", 20, true, "-pb", "(boolean)", "add parity breaking term on edge (when no disorder, i.e. w=0)");
    #endif
	std::cout << std::endl;
}

/// @brief 
void ui::printAllOptions() const{
    XXZUIparent::printAllOptions();
    std::cout << "H = \u03A3_r J_r\u03A3_i [ (1-\u03B7_r) S^x_i S^x_i+1 + (1+\u03B7_r) S^y_i S^y_i+1 + \u0394_r S^z_iS^z_i+1] + \u03A3_i h^z_i S^z_i + h^x\u03A3_i S^x_i" << std::endl << std::endl;
	std::cout << "h_i \u03B5 [hz - w, hz + w]" << std::endl;

	std::cout << "------------------------------ CHOSEN XXZ OPTIONS:" << std::endl;
    std::cout 
		  << "J1  = " << this->J1 << std::endl
		  << "J1n = " << this->J1n << std::endl
		  << "J1s = " << this->J1s << std::endl

		  << "J2  = " << this->J2 << std::endl
		  << "J2n = " << this->J2n << std::endl
		  << "J2s = " << this->J2s << std::endl

		  << "\u03941  = " << this->delta1 << std::endl
		  << "\u03941n = " << this->delta1n << std::endl
		  << "\u03941s = " << this->delta1s << std::endl
		  
		  << "\u03942  = " << this->delta2 << std::endl
		  << "\u03942n = " << this->delta2n << std::endl
		  << "\u03942s = " << this->delta2s << std::endl

		  << "hz  = " << this->hz << std::endl
		  << "hzn = " << this->hzn << std::endl
		  << "hzs = " << this->hzs << std::endl;
    #ifdef USE_SYMMETRIES
		  if(this->boundary_conditions == 0)        std::cout << "k  = " << this->syms.k_sym << std::endl;
		  if(this->k_real_sec(this->syms.k_sym))    std::cout << "p  = " << this->syms.p_sym << std::endl;
		  if(this->use_flip_X())                    std::cout << "zx  = " << this->syms.zx_sym << std::endl;
		  std::cout << "Sz = " << this->syms.Sz << std::endl;
    #else
		 std::cout  << "seed  = " << this->seed << std::endl
		  << "realisations  = " << this->realisations << std::endl
		  << "honid  = " << this->jobid << std::endl
		  << "w  = " << this->w << std::endl
		  << "ws = " << this->ws << std::endl
		  << "wn = " << this->wn << std::endl
		  << "add parity breaking term = " << this->add_parity_breaking << std::endl;
		//   << "add edge fields = " << this->add_edge_fields << std::endl;
    #endif
          std::cout << std::endl;
    printSeparated(std::cout, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
}   





    // arma::vec empty;
    // empty.save(arma::hdf5_name(this->saving_dir + this->set_info({"k", "p", "zx", "zz"}) + ".hdf5", "(empty)"));
    // auto kernel = [&](int k, int p, int zx, int zz)
    // {
    //     auto symmetric_model = std::make_unique<QHamSolver<XXZsym>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
    //                                                                     this->hx, this->hz, k, p, zz, zx, this->add_edge_fields);
    //     symmetric_model->diagonalization();
    //     arma::vec E = (symmetric_model->get_eigenvalues());
    //     arma::cx_mat V = symmetric_model->get_eigenvectors();
    //     std::string _suff = "_k=" + std::to_string(k) + "_p=" + std::to_string(p) + "_zx=" + std::to_string(zx) + "_zz=" + std::to_string(zz);
    //     E.save(arma::hdf5_name(this->saving_dir + this->set_info({"k", "p", "zx", "zz"}) + ".hdf5", "energies/" + _suff, arma::hdf5_opts::append));
    //     V.save(arma::hdf5_name(this->saving_dir + this->set_info({"k", "p", "zx", "zz"}) + ".hdf5", "eigenstates/" + _suff, arma::hdf5_opts::append));
    //     //symmetric_model->get_dense_hamiltonian().save(arma::hdf5_name(this->saving_dir + this->set_info({"k", "p", "zx", "zz"}) + ".hdf5", "Hamiltonian/" + _suff));
    // };
    // loopSymmetrySectors(kernel);

    // return;




};