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

	// auto Hamil = this->ptr_to_model->get_hamiltonian();
	// this->l_steps = 0.05 * Hamil.n_cols;
	// if(this->l_steps > 200)
	// 	this->l_steps = 200;
	// auto polfed = polfed::POLFED<ui::element_type>(Hamil, this->l_steps, this->l_bundle, -1, this->tol, 0.2, this->seed, true);
	// auto [E, V] = polfed.eig();
	// return;
//     this->ptr_to_model = create_new_model_pointer();
//     this->ptr_to_model->diagonalization(false);
//     arma::vec E_ED = this->ptr_to_model->get_eigenvalues();

//     auto H = this->ptr_to_model->get_hamiltonian();
//     u64 dim = this->ptr_to_model->get_hilbert_size();
//     // this->l_steps = int(dim / 20.0);

//     auto tol_arr = arma::logspace(-15, -2, 50);
//     // std::cout << tol_arr.t() << std::endl;
//     arma::vec steps(tol_arr.size(), arma::fill::zeros);
//     arma::vec E_err(tol_arr.size(), arma::fill::zeros);
//     arma::vec V_err(tol_arr.size(), arma::fill::zeros);
// // #pragma omp parallel
//     for(int ii = 0; ii < tol_arr.size(); ii++)
//     {
//         this->tol = tol_arr(ii);
//         auto polfed = polfed::POLFED<ui::element_type>(H, this->l_steps, this->l_bundle, -1, tol_arr(ii), 0.2, this->seed, this->reorthogonalize);
//         arma::vec E; arma::mat V;
//         std::tie(E, V) = polfed.eig();
//         steps(ii) = polfed.get_convergence_steps();

//         double Emin = arma::min(E);
//         auto i = std::min_element(std::begin(E_ED), std::end(E_ED), [=](double x, double y) {
//             return std::abs(x - Emin) < std::abs(y - Emin);
//             });
//         u64 idx = i - std::begin(E_ED);

//         arma::vec error(E.size(), arma::fill::zeros);
//         for(int k = 0; k < E.size(); k++){
//             error(k) = arma::norm(H * V.col(k) - E(k) * V.col(k));
//         }
//         V_err(ii) = arma::max(error);
//         auto permut = sort_permutation(E, [](const double a, const double b)
// 								   { return a < b; });
//         apply_permutation(E, permut);
//         error  = arma::vec(E.size(), arma::fill::zeros);
//         for(int k = 0; k < E.size(); k++){
//             error(k) = std::abs(E_ED(idx + k) - E(k));
//         }
//         E_err(ii) = arma::max(error);
//         printSeparated(std::cout, "\t", 16, true, this->tol, E_err(ii), V_err(ii), steps(ii));
//     }
//     std::string name = this->saving_dir + this->set_info() + "_jobid=" + std::to_string(this->jobid);
//     steps.save(arma::hdf5_name(name + ".hdf5", "steps"));
//     E_err.save(arma::hdf5_name(name + ".hdf5", "energy error", arma::hdf5_opts::append));
//     V_err.save(arma::hdf5_name(name + ".hdf5", "state error", arma::hdf5_opts::append));
//     tol_arr.save(arma::hdf5_name(name + ".hdf5", "tolerance", arma::hdf5_opts::append));
//     // compare_energies();
//     return;


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
            spectrals();
            continue;

            diagonal_matrix_elements();
            std::cout << "\t\t - - - - - - FINISHED ITERATION IN : " << tim_s(start_loop) << " seconds\n\t\t\t Total time : " << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
        }}}}}}
            }
        std::cout << "Add default function" << std::endl;
	}
	std::cout << " - - - - - - FINISHED CALCULATIONS IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
}

void ui::fractality_in_clean_basis(){

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
    arma::mat V0(dim, dim);
    u64 col_start = 0;
    auto kernel = [&](int ks, int ps, int zxs)
        {
            auto model_sym = std::make_unique<QHS::QHamSolver<XXZsym>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->hz, ks, ps, zxs, this->syms.Sz);
            model_sym->diagonalization();
            u64 dim_sector = model_sym->get_hilbert_size();
            const arma::vec Esym = model_sym->get_eigenvalues();
            const auto& Vsym = model_sym->get_eigenvectors();
            const auto U = model_sym->get_model_ref().get_hilbert_space().symmetry_rotation(_hilbert_space);

            if(ks > 0 && ks < this->L / 2.){
                E0 = arma::join_cols(E0, Esym, Esym);
                V0.cols(col_start, col_start + dim_sector - 1) = arma::real(U * Vsym);
                col_start += dim_sector;
                V0.cols(col_start, col_start + dim_sector - 1) = arma::real(U * Vsym);
                col_start += dim_sector;
            } else {
                E0 = arma::join_cols(E0, Esym);
                V0.cols(col_start, col_start + dim_sector - 1) = arma::real(U * Vsym);
                col_start += dim_sector;
            }
            
        };
    loopSymmetrySectors(kernel);

    // std::cout << "DIM_TOT = " << dim_tot << std::endl;
    auto permut = sort_permutation(E0, [](const double a, const double b)
                            { return a < b; });
    apply_permutation(E0, permut);
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
              
            apply_permutation(overlaps, permut);
            
            if(n >= (num_of_states - num_of_states_for_Cn) / 2 && n < (num_of_states + num_of_states_for_Cn) / 2)
                 coefficients.col(n - (num_of_states - num_of_states_for_Cn) / 2) = arma::square(overlaps);
            
            for(int iiq = 0; iiq < qs.size(); iiq++)
                for(int n0 = 0; n0 < dim; n0++)
                    part_ratio_d2(n, iiq) += std::pow(overlaps(n0), qs(iiq));
                    
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
        coefficients /= double(num_of_states_for_Cn);
		
        std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
        
        createDirs(dir_realis);
        E.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "energies"));
        E0.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "E0",   arma::hdf5_opts::append));
        coefficients.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "coefficients",   arma::hdf5_opts::append));
        
        ldos.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "LDOS",   arma::hdf5_opts::append));
        energy_density.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "energy_density",   arma::hdf5_opts::append));
        
        qs.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "qs",   arma::hdf5_opts::append));
        part_ratio_d2.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "P2",   arma::hdf5_opts::append));
        part_ratio_d2_comp.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "P2_comp",   arma::hdf5_opts::append));
        // energy_density.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "energy_density",   arma::hdf5_opts::append));
        // energy_density.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "energy_density",   arma::hdf5_opts::append));
    }
}

void ui::spectrals()
{
	std::string dir = this->saving_dir + "Spectrals_SzSz" + kPSep;
    // std::string dir = this->saving_dir + "energy_current" + kPSep;
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
    // arma::sp_mat U_U1(dim_full, dim);
    auto check_spin = QOps::__builtins::get_digit(this->L);

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
            int nei = j + 1;
            if(nei >= this->L)
                nei = (this->boundary_conditions>0)? -1 : nei % this->L;
            
            double s_j = check_spin(base_state, nei) ? 0.5 : -0.5;				// true - spin up, false - spin down
            if(nei >= 0){
                // auto [val, state_tmp]   = operators::sigma_minus<cpx>(base_state, this->L, nei);
                // auto [val2, state]      = operators::sigma_plus<cpx>(state_tmp, this->L, j);
                // u64 idx = _hilbert_space.find(state);
                // try {
                //     kinetic(idx, k) += 1.0;
                //     kinetic(k, idx) += 1.0;
                // } 
                // catch (const std::exception& err) {
                //     // std::cout << "Exception:\t" << err.what() << "\n";
                //     // std::cout << "SHit ehhh..." << std::endl;
                //     // printSeparated(std::cout, "\t", 14, true, new_idx, idx, this->_hilbert_space(k), 1.0/);
                // }
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

    double _operator_HSnorm = arma::trace(kinetic * kinetic) / double(dim);
	kinetic = kinetic / std::sqrt(_operator_HSnorm);

    // const auto U = this->ptr_to_model->get_model_ref().get_hilbert_space().symmetry_rotation();
    // arma::sp_cx_mat kinetic2 = U.t() * energy_current() * U;
    // // // arma::sp_cx_mat kinetic2 = U_U1.t() * spin_current() * U_U1;
	// cpx _operator_HSnorm = arma::trace(kinetic2 * kinetic2.t()) / double(dim);
    // std::cout << "Hilbert-Schmidt Norm\t\t" << _operator_HSnorm << std::endl;
	// kinetic2 = kinetic2 / std::sqrt(_operator_HSnorm);
    // kinetic = arma::imag(kinetic2);
	std::cout << "Hilbert-Schmidt Norm\t\t" << _operator_HSnorm << "\t\tNew Norm\t\t" << arma::trace(kinetic * kinetic.t()) / dim << std::endl;

    auto subsystem_sizes = arma::conv_to<arma::Col<int>>::from(arma::linspace(0, this->L, this->L + 1));
	std::cout << subsystem_sizes.t() << std::endl;
	std::vector<QOps::generic_operator<element_type>> permutation_op;
	for(int LA_idx = 0; LA_idx < subsystem_sizes.size() - 1; LA_idx++)
	{	
		int LA = subsystem_sizes[LA_idx];
		auto start_LA = std::chrono::system_clock::now();
		std::vector<int> p(this->L);
		p[LA % this->L] = 0;
		for(int l = 0; l < this->L; l++){
			if(l != LA % this->L){
				p[l] = (l < (LA % this->L) )? l + 1 : l;
			}
		}
		// std::cout << LA << "\t\t" << p << "\t\t" << p2 << std::endl;
		auto permutation = QOps::_permutation_generator<element_type>(this->L, p);
		permutation_op.push_back(permutation);

		std::cout << " - - - - - - set permutation matrix for LA = " << LA << " in : " << tim_s(start_LA) << " s - - - - - - " << std::endl;
	}
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

            auto i2 = min_element(begin(Hdiagonal), end(Hdiagonal), [=](element_type x, element_type y) {
                return abs(x - E_av) < abs(y - E_av);
            });
            const u64 idx = i2 - begin(Hdiagonal);
            double quench_E = std::real( Hdiagonal(idx) );
            double tot_spin_init = kinetic(idx, idx);

            coeff = V.row(idx).t();

            quench = arma::vec(times.size(), arma::fill::zeros);
            psi = arma::cx_mat(dim, times.size(), arma::fill::zeros);
            start = std::chrono::system_clock::now();
            std::cout << " - - - - - - finished finding product state with energy E = " << quench_E << " compared to mean energy <H> = " << E_av << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
            

            start = std::chrono::system_clock::now();
        #pragma omp parallel for
            for(long t_idx = 0; t_idx < times.size(); t_idx++)
            {
                double time = times(t_idx);
                for(long alfa = 0; alfa < size; alfa++)
                {
                    auto state = V.col(alfa);
                    psi.col(t_idx) += std::exp(-1i * time * E(alfa)) * state * state(idx);
                }
            }

            std::cout << " - - - - - - finished preparing initial states for all times in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
        }
		arma::Mat<element_type> mat_elem = V.t() * kinetic * V;
		arma::Col<element_type> diag_mat_elem = arma::diagvec(mat_elem);
		// arma::mat xx = arma::abs(mat_elem);
		// xx.save(   arma::hdf5_name("MAT_ELEM" + info + ".hdf5", "mat_elem"));
		// xx = ( arma::mat(total_spin) );
		// xx.save(   arma::hdf5_name("MAT_ELEM" + info + ".hdf5", "sparse", arma::hdf5_opts::append));

		std::cout << " - - - - - - finished matrix elements in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();
		auto [_susc, _susc_r] = adiabatics::gauge_potential_save(mat_elem, E, this->L, wH);

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
            for(long t_idx = 0; t_idx < times.size(); t_idx++)
                quench(t_idx) = std::real( arma::cdot(psi.col(t_idx), kinetic * psi.col(t_idx)) );
            
        std::cout << " - - - - - - finished time evolution for Sz_L in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
        }
        start = std::chrono::system_clock::now();

		E_min = dim > dim_max? 0 : Eav_idx - std::min(50, int(dim/50));
		E_max = dim > dim_max? size : Eav_idx + std::min(50, int(dim/50));
        u64 num = E_max - E_min;
        arma::mat S(num, this->L + 1, arma::fill::zeros);
		arma::mat S_site = S;
		arma::vec participation_entropy(num, arma::fill::zeros);
		
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
        std::cout << " - - - - - - finished diagonalization of unperturbed H in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
        start = std::chrono::system_clock::now();

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
        mat_elem = V0.t() * kinetic * V0;
		arma::Col<element_type> diag_mat_elem0 = arma::diagvec(mat_elem);
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

			_susc.save(	 arma::hdf5_name(dir_realis + info + ".hdf5", "susc",     arma::hdf5_opts::append));
			_susc_r.save(arma::hdf5_name(dir_realis + info + ".hdf5", "susc_reg", arma::hdf5_opts::append));

			coeff.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "coefficients", arma::hdf5_opts::append));
			arma::vec x = arma::real(diag_mat_elem);  x.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "diag_mat",   arma::hdf5_opts::append));
            x = arma::imag(diag_mat_elem);  x.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "diag_mat_im",   arma::hdf5_opts::append));
			times.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "times",   arma::hdf5_opts::append));
			quench.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench",   arma::hdf5_opts::append));
			arma::vec( {quench_E} ).save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench_energy",   arma::hdf5_opts::append));
			arma::vec( {tot_spin_init} ).save(   arma::hdf5_name(dir_realis + info + ".hdf5", "tot_spin_init",   arma::hdf5_opts::append));
			// arma::vec( {_operator_HSnorm} ).save(   arma::hdf5_name(dir_realis + info + ".hdf5", "HSnorm",   arma::hdf5_opts::append));

			arma::vec( {wH} ).save(   arma::hdf5_name(dir_realis + info + ".hdf5", "wH",   arma::hdf5_opts::append));
			arma::vec( {r} ).save(   arma::hdf5_name(dir_realis + info + ".hdf5", "gap_ratio",   arma::hdf5_opts::append));

			S.save(arma::hdf5_name(dir_realis + info + ".hdf5", "entropy", arma::hdf5_opts::append));
			S_site.save(arma::hdf5_name(dir_realis + info + ".hdf5", "single_site_entropy", arma::hdf5_opts::append));
			subsystem_sizes.save(arma::hdf5_name(dir_realis + info + ".hdf5", "subsystem sizes", arma::hdf5_opts::append));
			participation_entropy.save(arma::hdf5_name(dir_realis + info + ".hdf5", "von Neumann participation entropy", arma::hdf5_opts::append));

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
    arma::sp_mat Hfull = full_model->get_hamiltonian();
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
        std::string folder = "." + kPSep + "results_fgr" + kPSep;
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