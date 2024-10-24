#include "includes/QuadraticUI.hpp"

int outer_threads = 1;
int num_of_threads = 1;


namespace QuadraticUI{

void ui::make_sim(){
    printAllOptions();
	
	this->ptr_to_model = this->create_new_model_pointer();
	
	clk::time_point start = std::chrono::system_clock::now();
    switch (this->fun)
	{
	case 0: 
		diagonalize(); 
		break;
	case 1:
		spectral_form_factor();
		break;
	case 2:
		eigenstate_entanglement();
		break;
	case 3:
		eigenstate_entanglement_degenerate();
		break;
	case 4:
		diagonal_matrix_elements();
		break;
	case 5:
		spectrals();
		break;
	case 6:
		eigenstate_entanglement_manybody();
		break;
	case 7:
		quench();
		break;
	default:
		#define generate_scaling_array(name) arma::linspace(this->name, this->name + this->name##s * (this->name##n - 1), this->name##n);
		auto L_list = generate_scaling_array(L);
		auto J_list = generate_scaling_array(J);
		auto w_list = generate_scaling_array(w);
		auto g_list = generate_scaling_array(g);

		for (auto& Lx : L_list){
			for(auto& Jx : J_list){
				for(auto& wx : w_list){
					for(auto& gx : g_list){
						this->L = Lx;
						set_volume();

						this->J = Jx;
						this->w = wx;
						this->g = gx;
						this->site = this->L / 2.;
						this->reset_model_pointer();
						const auto start_loop = std::chrono::system_clock::now();
						std::cout << " - - START NEW ITERATION:\t\t par = "; // simuVAtion end
						printSeparated(std::cout, "\t", 16, true, this->L, this->J, this->w, this->g);
						
						eigenstate_entanglement_manybody(); continue;
						spectrals(); continue;
						spectral_form_factor(); continue;
						eigenstate_entanglement(); continue;
						// eigenstate_entanglement_degenerate();
						// average_sff();
						std::cout << "\t\t - - - - - - FINISHED ITERATION IN : " << tim_s(start_loop) << " seconds\n\t\t\t Total time : " << tim_s(start) << " s - - - - - - " << std::endl; // simuVAtion end
		}}}}
        std::cout << "Add default function" << std::endl;
	}
	std::cout << " - - - - - - FINISHED CALCUVATIONS IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simuVAtion end
}

// -------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------- USER DEFINED ROUTINES
void ui::spectrals()
{
	std::string dir = this->saving_dir + "SpectralFunctions" + kPSep;
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info();

	const size_t size = dim > 1e5? this->l_steps : dim;

	arma::vec energies(size, arma::fill::zeros);

	int Ll = this->L;
	int counter = 0;
	
	const double _bandwidth_def = RP_data::default_pars::getBandwidth(this->g, this->L);
	
	const arma::vec betas = arma::logspace(-2, 2, 100);
	const arma::vec omegax = arma::logspace(std::log10(1.0/dim) - 2, std::log10( _bandwidth_def ) + 1, 40 * this->L);
	const arma::vec energy_density = arma::regspace(0.05, 0.02, 0.95);

	arma::Mat<element_type> spectral_fun(omegax.size()-1, energy_density.size(), arma::fill::zeros);
	arma::Mat<element_type> spectral_fun_typ(omegax.size()-1, energy_density.size(), arma::fill::zeros);
	arma::Mat<element_type> element_count(omegax.size()-1, energy_density.size(), arma::fill::zeros);
	
	double window_width = 0.04;

// #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
	for(int realis = 0; realis < this->realisations; realis++)
	{
		clk::time_point start_re = std::chrono::system_clock::now();
		if(realis > 0)
			this->ptr_to_model->generate_hamiltonian();
		
		clk::time_point start = std::chrono::system_clock::now();
		if(dim > 1e5){
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

		std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
		createDirs(dir_realis);
		E.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "energies"));
		
		long int E_min = dim < 0? 0 : Eav_idx - long(dim / 4);
		long int E_max = dim > 1e5? dim : Eav_idx + long(dim / 4);

		double wH = 0;
		for (long int i = E_min; i < E_max; i++)
			wH += E(i+1) - E(i);

		start = std::chrono::system_clock::now();
		// arma::Mat<element_type> mat_elem = V * Sz_ops[i] * V.t();
		auto kernel = [Ll](u64 state){ 
			auto [val1, tmp22] = operators::sigma_z(state, Ll, Ll - 1 );
			return std::make_pair(state, val1);
			};
		auto _operator = QOps::generic_operator<>(this->L, std::move(kernel), 1.0);
		arma::sp_mat opmat = arma::real(_operator.to_matrix(dim));
		arma::Mat<element_type> mat_elem = V.t() * opmat * V;
		std::cout << " - - - - - - finished matrix elements in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();

		// auto [_Z, _count, _count_proj,AGP_T, AGP_T_reg, AGP_E, AGP_E_proj] = adiabatics::gauge_potential_finite_T(mat_elem, E, betas, energy_density);
		auto [_susc, _susc_r] = adiabatics::gauge_potential_save(mat_elem, E, this->L, wH);

		std::cout << " - - - - - - finished AGP in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();
		arma::Mat<element_type> _integrated_spectral_fun(omegax.size()-1, energy_density.size(), arma::fill::zeros);
		arma::Mat<element_type> _spectral_fun(omegax.size()-1, energy_density.size(), arma::fill::zeros);
		arma::Mat<element_type> _spectral_fun_typ(omegax.size()-1, energy_density.size(), arma::fill::zeros);
		arma::Mat<element_type> _element_count(omegax.size()-1, energy_density.size(), arma::fill::zeros);
		
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
						if(indices.size() > 1)
							_integrated_spectral_fun(k, ii) = simpson_rule(x, y);
					}
				}
		}
		std::cout << " - - - - - - finished Sz_L matrix elements at finite energy density in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
	// 	start = std::chrono::system_clock::now();

	// 	arma::Mat<element_type> _spectral_fun_beta(omegax.size()-1, betas.size(), arma::fill::zeros);
	// 	arma::vec _partition_fun(betas.size(), arma::fill::zeros);
		
	// 	double _delta_log_omegax = std::log(omegax(1)) - std::log(omegax(0));
	// #pragma omp parallel for
	// 	for(int iiB = 0; iiB < betas.size(); iiB++){
	// 		double beta = betas(iiB);
	// 		double _Z = 0;
	// 		for(int n = 0; n < dim; n++){
	// 			double _thermal_weight = std::exp(-beta * (E(n) - E(0)) );
	// 			_Z += _thermal_weight;
	// 			for(int m = n + 1; m < dim; m++){
	// 				double omega = E(m) - E(n);
	// 				u64 idx_w;
	// 				if(omega < omegax(0))
	// 					idx_w = 0;
	// 				else
	// 					idx_w = 1 + int( (std::log(omega) - std::log(omegax(0))) / _delta_log_omegax );
	// 				_spectral_fun_beta(idx_w, iiB) += 2 * _thermal_weight * mat_elem(n, m);
	// 			}
	// 		}
	// 		_spectral_fun_beta(iiB) = _spectral_fun_beta(iiB) / _Z;
	// 		_partition_fun(iiB) = _Z;
	// 	}
	// 	std::cout << " - - - - - - finished Sz_L matrix elements at finite temperature in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		// #ifndef MY_MAC
		{
			omegax.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "omegas",   arma::hdf5_opts::append));
			_integrated_spectral_fun.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "integrated_spectral_fun",   arma::hdf5_opts::append));
			energy_density.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "energy_density",   arma::hdf5_opts::append));
			_spectral_fun.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "spectral_fun",   arma::hdf5_opts::append));
			_spectral_fun_typ.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "log(_spectral_fun_typ)",   arma::hdf5_opts::append));
			_element_count.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "element_count",   arma::hdf5_opts::append));

			// betas.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "betas",   arma::hdf5_opts::append));
			// _partition_fun.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "partition_fun",   arma::hdf5_opts::append));
			// _spectral_fun_beta.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "spectral_fun_beta",   arma::hdf5_opts::append));
			
			_susc.save(	 arma::hdf5_name(dir_realis + info + ".hdf5", "susc",     arma::hdf5_opts::append));
			_susc_r.save(arma::hdf5_name(dir_realis + info + ".hdf5", "susc_reg", arma::hdf5_opts::append));
		}
		spectral_fun += _spectral_fun;
		element_count += _element_count;
		spectral_fun_typ += _spectral_fun_typ;
		// #endif
		
		energies += E;
		counter++;
		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start_re) << " s - - - - - - " << std::endl; // simulation end
	}
	if(counter == 0) return;
	
	// #ifdef MY_MAC
	// 	energies /= double(counter);
	// 	spectral_fun = spectral_fun / element_count;
	// 	spectral_fun_typ = arma::exp(spectral_fun_typ / element_count);

	// 	energies.save(		arma::hdf5_name(dir + info + ".hdf5", "energies"));
	// 	energy_density.save(   arma::hdf5_name(dir + info + ".hdf5", "energy_density",   arma::hdf5_opts::append));
	// 	spectral_fun.save(	arma::hdf5_name(dir + info + ".hdf5", "spectral_fun",   arma::hdf5_opts::append));
	// 	spectral_fun_typ.save(	arma::hdf5_name(dir + info + ".hdf5", "spectral_fun_typ",   arma::hdf5_opts::append));
	// 	element_count.save(	arma::hdf5_name(dir + info + ".hdf5", "element_count",   arma::hdf5_opts::append));
	// 	omegax.save(   		arma::hdf5_name(dir + info + ".hdf5", "omegax",   arma::hdf5_opts::append));
	// 	arma::vec({(double)counter}).save(	arma::hdf5_name(dir + info + ".hdf5", "realisations",   arma::hdf5_opts::append));
	// #endif
}

void ui::quench()
{
	std::string dir = this->saving_dir + "Quench" + kPSep;
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info();

	const size_t size = dim > 1e5? this->l_steps : dim;
	
	double tH = dim;
	// int time_end = (int)std::ceil(std::log10(10 * tH));
	// time_end = (time_end / std::log10(tH) < 10 ) ? time_end + 2 : time_end;
	arma::vec times = arma::logspace(-2, (std::log10(1000 * tH)), this->num_of_points);

	int Ll = this->L;

	int counter = 0;
// #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
	for(int realis = 0; realis < this->realisations; realis++)
	{
		clk::time_point start_re = std::chrono::system_clock::now();
		if(realis > 0)
			this->ptr_to_model->generate_hamiltonian();
		
		clk::time_point start = std::chrono::system_clock::now();
		if(dim > 1e5){
			this->ptr_to_model->diag_sparse(this->l_steps, this->l_bundle, this->tol, this->seed);	
		}
		else{
        	this->ptr_to_model->diagonalization();
		}
		std::cout << " - - - - - - finished diagonalization in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();
		
		const arma::vec E = this->ptr_to_model->get_eigenvalues();
		const auto& V = this->ptr_to_model->get_eigenvectors();
		std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
		createDirs(dir_realis);
		E.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "energies"));
		
		arma::vec Hdiagonal = arma::diagvec( this->ptr_to_model->get_dense_hamiltonian() );

		double E_av = arma::trace(E) / double(dim);
		auto i = min_element(begin(Hdiagonal), end(Hdiagonal), [=](double x, double y) {
			return abs(x - E_av) < abs(y - E_av);
		});
		const u64 idx = i - begin(Hdiagonal);
		double quench_E = Hdiagonal(idx);

		arma::vec coeff = V.row(idx).t();
		coeff.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "coefficients", arma::hdf5_opts::append));

		arma::vec quench(times.size(), arma::fill::zeros);
		arma::cx_mat psi(dim, times.size(), arma::fill::zeros);

		start = std::chrono::system_clock::now();
		std::cout << " - - - - - - finished finding product state with energy E = " << quench_E << " compared to mean energy <H> = " << E_av << tim_s(start) << " s - - - - - - " << std::endl; // simulation end

		start = std::chrono::system_clock::now();
	#pragma omp parallel for
		for(long t_idx = 0; t_idx < times.size(); t_idx++)
		{
			double time = times(t_idx);
			for(long alfa = 0; alfa < dim; alfa++)
			{
				auto state = V.col(alfa);
				psi.col(t_idx) += std::exp(-1i * time * E(alfa)) * state * state(idx);
			}
		}

		std::cout << " - - - - - - finished preparing initial states for all times in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		
		start = std::chrono::system_clock::now();
		auto kernel = [Ll](u64 state){ 
			auto [val1, tmp22] = operators::sigma_z(state, Ll, Ll - 1 );
			return std::make_pair(state, val1);
			};
		auto _operator = QOps::generic_operator<>(this->L, std::move(kernel), 1.0);
		arma::sp_mat op = arma::real(_operator.to_matrix(dim));
		arma::Mat<element_type> mat_elem = V.t() * op * V;
		arma::vec diag_mat_elem = arma::diagvec(mat_elem);
		std::cout << " - - - - - - finished Sz_L matrix elements in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end

		start = std::chrono::system_clock::now();
	#pragma omp parallel for
		for(long t_idx = 0; t_idx < times.size(); t_idx++)
			quench(t_idx) = std::real( arma::cdot(psi.col(t_idx), op * psi.col(t_idx)) );
		
		std::cout << " - - - - - - finished time evolution for Sz_L in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		
		// start = std::chrono::system_clock::now();
		// auto [autocorr_Sz, LTA_Sz] = spectrals::autocorrelation_function(mat_elem, E, times);
		// std::cout << " - - - - - - finished auto correlator time evolution for Sz_L in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		// #ifndef MY_MAC
		{
			diag_mat_elem.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "diag_mat",   arma::hdf5_opts::append));
			times.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "times",   arma::hdf5_opts::append));
			quench.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench",   arma::hdf5_opts::append));
			arma::vec( {quench_E} ).save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench_energy",   arma::hdf5_opts::append));
		}
		// #endif
		
		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start_re) << " s - - - - - - " << std::endl; // simulation end
	}
}


// -------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------- IMPLEMENTATION OF UI

void ui::set_volume(){
	#if defined(RP) || defined(SYK) || defined(PLRB)
		this->V = ULLPOW(this->L);
	// #elif 
	// 	this->V = this->L;
	#else
		this->V = std::pow(this->L, DIM);
	#endif
}

/// @brief Create unique pointer to model with current parameters in cVAss
typename ui::model_pointer ui::create_new_model_pointer(){
    return std::make_unique<QHS::QHamSolver<Quadratic>>(this->V, this->J, this->w, this->seed, this->g, this->boundary_conditions); 
}

/// @brief Reset member unique pointer to model with current parameters in cVAss
void ui::reset_model_pointer(){
    this->ptr_to_model.reset(new QHS::QHamSolver<Quadratic>(this->V, this->J, this->w, this->seed, this->g, this->boundary_conditions)); 
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
    user_interface_dis<Quadratic>::parse_cmd_options(argc, argv);

    //<! set the remaining UI parameters
	std::string choosen_option = "";	

	#define _set_param_(name, g_eq0) choosen_option = "-" #name;                               \
	                        this->set_option(this->name, argv, choosen_option, g_eq0);         \
                                                                                        \
	                        choosen_option = "-" #name "s";                             \
	                        this->set_option(this->name##s, argv, choosen_option, g_eq0);      \
                                                                                        \
	                        choosen_option = "-" #name "n";                             \
	                        this->set_option(this->name##n, argv, choosen_option, true);
	#define set_param(name) _set_param_(name, false);
    
	set_param(J);
	#if defined(ANDERSON) || defined(AUBRY_ANDRE)
		set_param(w);
	#endif
	#if defined(AUBRY_ANDRE) || defined(PLRB) || defined(RP)
		set_param(g);
	#endif

	set_volume();
	
    //<! FOLDER
    std::string folder = "results" + kPSep;
	
	// #if defined(ANDERSON)
	// 	folder += "Anderson" + kPSep;
	// #elif defined(SYK)
	// 	folder += "SYK2" + kPSep;
	// #elif defined(PLRB)
	// 	folder += "PLRB" + kPSep;
	// #elif defined(AUBRY_ANDRE)
	// 	folder += "AubryAndre" + kPSep;
	// #elif defined(RP)
	// 	folder += "RP" + kPSep;
	// #else
	// 	folder += "FreeFermions" + kPSep;
	// #endif
	folder += model + kPSep;
	#if !defined(SYK) && !defined(PLRB) && !defined(RP)
		folder += "dim=" + std::to_string(DIM) + kPSep;
	#endif
	#if !defined(SYK) && !defined(RP)
		switch(this->boundary_conditions){
			case 0: folder += "PBC" + kPSep; break;
			case 1: folder += "OBC" + kPSep; break;
			case 2: folder += "ABC" + kPSep; break;
			default:
				folder += "PBC" + kPSep; 
				break;
			
		}
	#endif

	folder = this->dir_prefix + folder;
	
    if (fs::create_directories(folder) || fs::is_directory(folder)) // creating the directory for saving the files with results
    	this->saving_dir = folder;									// if can create dir this is is
}


/// @brief 
void ui::set_default(){
    user_interface_dis<Quadratic>::set_default();
    this->J = 1.0;
	this->Js = 0.0;
	this->Jn = 1;

	this->w = 0.0;
	this->ws = 0.0;
	this->wn = 1;

	this->g = 0.0;
	this->gs = 0.0;
	this->gn = 1;
}

/// @brief 
void ui::print_help() const {
    user_interface_dis<Quadratic>::print_help();
    
    printf(" Flags for Quadratic model:\n");
    printSeparated(std::cout, "\t", 20, true, "-J", "(double)", "coupling strength (hopping)");
    printSeparated(std::cout, "\t", 20, true, "-Js", "(double)", "step in coupling strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-Jn", "(int)", "number of couplings in the sweep");

    printSeparated(std::cout, "\t", 20, true, "-w", "(double)", "Anderson: disorder bandwidth on localized spins\n Aubry-Andre: strength of potential");
    printSeparated(std::cout, "\t", 20, true, "-ws", "(double)", "step in disorder strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-wn", "(int)", "number of disorder in the sweep");

    printSeparated(std::cout, "\t", 20, true, "-g", "(double)", "Aubry-Andre: periodicity of the potential");
    printSeparated(std::cout, "\t", 20, true, "-gs", "(double)", "step in periodicity");
    printSeparated(std::cout, "\t", 20, true, "-gn", "(int)", "number of periodicities in sweep");

	std::cout << std::endl;
}

/// @brief 
void ui::printAllOptions() const{
    user_interface_dis<Quadratic>::printAllOptions();
    std::cout << "Quadratic:\n\t\t" << "H = J\u03A3_i,j A_{i,j} c^+_i c_j + h.c + \u03A3_j h_j n_j" << std::endl << std::endl;
	#if defined(ANDERSON)
		std::cout << "Anderson:\th_j \u03B5 [- w, w]" << std::endl;
	#elif defined(SYK)
		std::cout << "SYK2\th_j = 0" << std::endl;
	#elif defined(AUBRY_ANDRE)
		std::cout << "Aubry-Andre\th_j = w*cos(2\u03C0j*g + \u03C6)\t\u03C6=0 - random phase (0 for now)" << std::endl;
	#elif defined(RP)
		std::cout << "Rozezweig-Porter\th_j = N(0,1); A_{i,j} = GOE(N) / N^{-g/2}" << std::endl;
	#else
		std::cout << "Free-Fermions\th_j = 0" << std::endl;
	#endif
	std::cout << "------------------------------ CHOSEN Quadratic OPTIONS:" << std::endl;
    std::cout 
		  << "V = " << this->V << std::endl
		  << "J  = " << this->J << std::endl
		  << "Jn = " << this->Jn << std::endl
		  << "Js = " << this->Js << std::endl;
	#if defined(ANDERSON) || defined(AUBRY_ANDRE)
		std::cout 
		  << "w  = " << this->w << std::endl
		  << "ws = " << this->ws << std::endl
		  << "wn = " << this->wn << std::endl;
	#endif
	#if defined(AUBRY_ANDRE) || defined(PLRB) || defined(RP)
		std::cout
		  << "g  = " << this->g << std::endl
		  << "gs = " << this->gs << std::endl
		  << "gn = " << this->gn << std::endl;
	#endif
}   

/// @brief 
/// @param skip 
/// @param sep 
/// @return 
std::string ui::set_info(std::vector<std::string> skip, std::string sep) const
{
        std::string name = "L=" + std::to_string(this->L) + ",J=" + to_string_prec(this->J);
		#if defined(ANDERSON) || defined(AUBRY_ANDRE)
			name += ",w=" + to_string_prec(this->w);
		#endif
		#if defined(AUBRY_ANDRE) || defined(PLRB) || defined(RP)
        	name += ",g=" + to_string_prec(this->g);
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










};


// 	std::string dir = this->saving_dir + "Spectrals" + kPSep;
// 	createDirs(dir);
	
// 	const int Lhalf = this->L / 2;
// 	const int Ll = this->L;
// 	size_t dim = ULLPOW(Ll);
// 	std::string info = this->set_info();
// 	this->ptr_to_model.reset(new QHS::QHamSolver<Quadratic>(dim, this->J, this->w, this->seed, this->g, this->boundary_conditions)); 

// 	const size_t size = dim > 1e5? this->l_steps : dim;

// 	auto disorder_generator = disorder<double>(this->seed);
// 	std::cout << disorder_generator.uniform(dim, 0.5).t() << std::endl;
// 	int counter = 0;
// // #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
// 	for(int realis = 0; realis < this->realisations; realis++)
// 	{
// 		clk::time_point start_re = std::chrono::system_clock::now();
// 		if(realis > 0)
// 			this->ptr_to_model->generate_hamiltonian();
		
// 		clk::time_point start = std::chrono::system_clock::now();
// 		if(dim > 1e5){
// 			this->ptr_to_model->diag_sparse(this->l_steps, this->l_bundle, this->tol, this->seed);	
// 		}
// 		else{
//         	// this->ptr_to_model->diagonalization();
// 		}
// 		std::cout << " - - - - - - finished diagonalization in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
// 		start = std::chrono::system_clock::now();
		
// 		// const arma::vec E = this->ptr_to_model->get_eigenvalues();
// 		// const auto& V = this->ptr_to_model->get_eigenvectors();
// 		// double E_av = arma::trace(E) / double(dim);
// 		// auto i = min_element(begin(E), end(E), [=](double x, double y) {
// 		// 	return abs(x - E_av) < abs(y - E_av);
// 		// });
// 		// const long Eav_idx = i - begin(E);

// 		// std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
// 		// createDirs(dir_realis);
// 		// E.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "energies"));
		

// 		arma::vec LTA_r(4, arma::fill::zeros);
// 		arma::vec agp_norm_r(4, arma::fill::zeros);
// 		arma::vec typ_susc_r(4, arma::fill::zeros);

// 		// const double window_width = 0.0001 * ( E(dim-1) - E(0) );
// 		// spectrals::preset_omega set_omega(E, window_width, E(Eav_idx));
// 		// arma::vec omegas(set_omega.num_of_omegas, arma::fill::zeros);
// 		// arma::Col<element_type> (set_omega.num_of_omegas, arma::fill::zeros);

// 		arma::sp_mat Sz, Sq(dim, dim), nq(dim, dim), nr(dim, dim);

// 		std::cout << " - - - - - - CREATING MANY-BODY OPERATORS" << std::endl;
// 		{
// 			start = std::chrono::system_clock::now();
// 			auto kernel = [Ll, Lhalf](u64 state){ 
// 					auto [val, temporary] = operators::sigma_z(state, Ll, Lhalf );
// 					return std::make_pair(state, val);
// 					};
// 			auto _operator = QOps::generic_operator<>(this->L, std::move(kernel), 1.0);
// 			Sz = 2 * arma::real(_operator.to_matrix(dim));
// 			std::cout << " - - - - - - finished setting Sz operator with norm ||S_z||^2= " << arma::trace(Sz * Sz) / double(dim)  << "; in time: " << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
// 		}
// 		{
// 			double q = constants<double>::two_pi / double(this->L);
// 			start = std::chrono::system_clock::now();
// 			for(int site = 0; site < this->L; site++){
// 				auto kernel = [Ll, site, q](u64 state){ 
// 						auto [val, temporary] = operators::sigma_z(state, Ll, site );
// 						return std::make_pair( state, val * std::cos(q * site) );
// 						};
// 				auto _operator = QOps::generic_operator<>(this->L, std::move(kernel), 1.0);
// 				Sq += arma::real(_operator.to_matrix(dim));
// 			}
// 			Sq = 2 * std::sqrt(2 / double(this->L) ) * Sq;
// 			std::cout << " - - - - - - finished setting Sq operator with norm ||S_q||^2= " << arma::trace(Sq * Sq) / double(dim) << "; in time: " << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
// 		}
// 		std::cout << " - - - - - -CREATING  SINGLE-PARTICLE OPERATORS" << std::endl;
// 		{
// 			auto disorder = disorder_generator.uniform(dim, 0.5);
// 			std::cout << disorder.t() << std::endl;
// 			double q = constants<double>::two_pi / double(dim);
// 			for(int site = 0; site < dim; site++){
// 				nq(site, site) = std::cos(q * site);
// 				nr(site, site) = disorder(site);
// 			}
// 			nq = nq - arma::trace(nq) / dim * arma::eye<arma::sp_mat>(dim, dim);
// 			nr = nr - arma::trace(nr) / dim * arma::eye<arma::sp_mat>(dim, dim);
// 			// nq = nq / ( arma::trace(nq * nq) / double(dim)  - arma::trace(nq) / dim * arma::trace(nq) / dim);
// 			// nr = nr / ( arma::trace(nr * nr) / double(dim)  - arma::trace(nr) / dim * arma::trace(nr) / dim);
// 			std::cout << " - - - - - - finished setting Sq operator with norm ||n_q||_sp= " << arma::trace(nq * nq) / double(dim) << "; in time: " << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
// 			std::cout << " - - - - - - finished setting Sq operator with norm ||n_r||_sp= " << arma::trace(nr * nr) / double(dim) << "; in time: " << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
// 		}

// 		// start = std::chrono::system_clock::now();
// 		// double _agp, _typ_susc, _susc;
// 		// {
// 		// 	arma::vec tmp;
// 		// 	start = std::chrono::system_clock::now();
// 		// 	// arma::Mat<element_type> mat_elem = V * Sz_ops[i] * V.t();
// 		// 	auto kernel = [Ll, ](u64 state){ 
// 		// 		auto [val1, tmp22] = operators::sigma_z(state, Ll, site_1 );
// 		// 		auto [val2, tmp33] = operators::sigma_z(state, Ll, site_2 );
// 		// 		return std::make_pair(state, val1 * val2);
// 		// 		};
// 		// 	auto _operator = QOps::generic_operator<>(this->L, std::move(kernel), 1.0);
// 		// 	arma::sp_mat op = arma::real(_operator.to_matrix(dim));
// 		// 	arma::Mat<element_type> mat_elem = V.t() * op * V;
// 		// 	std::tie(_agp, _typ_susc, _susc, tmp) = adiabatics::gauge_potential(mat_elem, E, this->L);
// 		// 	agp_norm_r(i) = _agp;
// 		// 	typ_susc_r(i) = _typ_susc;
// 		// 	diag_mat_elem_r.col(i) = arma::diagvec(mat_elem);
			
// 		// 	auto [omegas_i, matter] = set_omega.get_matrix_elements(mat_elem);
// 		// 	omegas = omegas_i;
// 		// 	spectral_funs.col(i) = matter;

//     	// 	std::cout << " - - - - - - finished matrix elements for sites: i=" << site_1 << ", j=" << site_2 << " in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
// 		// }
// 		// start = std::chrono::system_clock::now();
// 		// // arma::Mat<element_type> mat_elem = V * Sz_ops[i] * V.t();
// 		// auto kernel = [Ll, N](u64 state){ 
// 		// 	auto [val1, tmp22] = operators::sigma_z(state, Ll, Ll - 1 );
// 		// 	return std::make_pair(state, val1);
// 		// 	};
// 		// auto _operator = QOps::generic_operator<>(this->L, std::move(kernel), 1.0);
// 		// arma::sp_mat op = arma::real(_operator.to_matrix(dim));
// 		// arma::Mat<element_type> mat_elem = V.t() * op * V;
// 		// auto [_agp, _typ_susc, _susc, tmp] = adiabatics::gauge_potential(mat_elem, E, this->L);
// 		// agp_norm_r(site_pairs.size()) = _agp;
// 		// typ_susc_r(site_pairs.size()) = _typ_susc;
// 		// arma::vec diag_mat_elem_Sz_r = arma::diagvec(mat_elem);
// 		// auto [omegas_i, matter] = set_omega.get_matrix_elements(mat_elem);

// 		// std::cout << " - - - - - - finished Sz_L matrix elements in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
// 		// {
// 		// 	agp_norm_r.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "agp",   arma::hdf5_opts::append));
// 		// 	typ_susc_r.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "typ_susc",   arma::hdf5_opts::append));
// 		// 	diag_mat_elem_r.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "diag_mat",   arma::hdf5_opts::append));
// 		// 	omegas.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "omegas",   arma::hdf5_opts::append));
// 		// 	spectral_funs.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "spectral_funs",   arma::hdf5_opts::append));

// 		// 	matter.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "spectral_Sz_L",   arma::hdf5_opts::append));
// 		// 	diag_mat_elem_Sz_r.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "diag_mat_Sz_L",   arma::hdf5_opts::append));
			
// 		// }
// 		// std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start_re) << " s - - - - - - " << std::endl; // simulation end
	// }