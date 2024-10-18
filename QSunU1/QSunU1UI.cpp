#include "includes/QSunU1UI.hpp"

int outer_threads = 1;
int num_of_threads = 1;

bool normalize_grain = 1;

namespace QSunU1UI{

void ui::make_sim(){
    printAllOptions();
	clk::time_point start = std::chrono::system_clock::now();
    
	this->ptr_to_model = this->create_new_model_pointer();
	
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
		multifractality();
		break;
	case 4:
		entanglement_evolution();
		break;
	case 5:
		survival_probability();
		break;
	case 6:
		matrix_elements();
		break;
	case 7:
		agp_save();
		break;
	case 8:
		spectral_function();
		break;
	default:
		#define generate_scaling_array(name) arma::linspace(this->name, this->name + this->name##s * (this->name##n - 1), this->name##n);
		
		auto J_list = generate_scaling_array(J);
		auto alfa_list = generate_scaling_array(alfa);
		auto h_list = generate_scaling_array(h);
		auto w_list = generate_scaling_array(w);
		auto gamma_list = generate_scaling_array(gamma);

		auto L_list = arma::linspace(this->L_loc, this->L_loc + this->Ls * (this->Ln - 1), this->Ln);
		std::cout << L_list.t() << std::endl;
		
		for (auto& L_locx : L_list){
			for (auto& alfax : alfa_list){
				for (auto& hx : h_list){
					for(auto& Jx : J_list){
						for(auto& wx : w_list){
							for(auto& gammax : gamma_list)
							{
								this->L_loc = L_locx;	
								this->L = L_locx + this->grain_size;
								if(this->L % 2 == 1 && this->Sz == 0.0) this->Sz = 0.5;
								if(this->L % 2 == 0 && std::abs(this->Sz) == 0.5) this->Sz = 0.0;

								this->alfa = alfax;
								this->h = hx;
								this->J = Jx;
								this->w = wx;
								this->gamma = gammax;
								this->site = this->L / 2.;
								
								this->reset_model_pointer();
								const auto start_loop = std::chrono::system_clock::now();
								std::cout << " - - START NEW ITERATION:\t\t par = "; // simulation end
								printSeparated(std::cout, "\t", 16, true, this->L_loc, this->J, this->alfa, this->h, this->w, this->gamma);
								
								survival_probability();
								//entanglement_evolution();
								//average_sff();
								std::cout << "\t\t - - - - - - FINISHED ITERATION IN : " << tim_s(start_loop) << " seconds\n\t\t\t Total time : " << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
						}}}}}}
        std::cout << "Add default function" << std::endl;
	}
	std::cout << " - - - - - - FINISHED CALCULATIONS IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
}


// ------------------------------------------------ OVERRIDEN METHODS
/// @brief Cast state from U(1) basis to full Hilbert basis
/// @param state input state in U(1) basis
/// @return state in full basis
arma::Col<ui::element_type> ui::cast_state(const arma::Col<ui::element_type>& state)
{
    auto U1sector = this->ptr_to_model->get_mapping();
    arma::Col<ui::element_type> full_state(ULLPOW(this->L), arma::fill::zeros);
    for(int i = 0; i < U1sector.size(); i++)
        full_state(U1sector[i]) = state(i);
    return full_state;
}

/// @brief Calculate matrix elements of local operators
void ui::matrix_elements()
{
	std::string dir = this->saving_dir + "MatrixElements" + kPSep;
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info();

	// arma::vec sites = arma::linspace(0, this->L-1, this->L);
	arma::Col<int> sites = arma::Col<int>({this->grain_size, (int)this->L / 2, (int)this->L - 1});

	arma::vec agp_norm_Sz(sites.size(), arma::fill::zeros);
	arma::vec typ_susc_Sz(sites.size(), arma::fill::zeros);
	arma::mat diag_mat_elem_Sz(dim, sites.size(), arma::fill::zeros);

	arma::vec agp_norm_SzSz(sites.size(), arma::fill::zeros);
	arma::vec typ_susc_SzSz(sites.size(), arma::fill::zeros);
	arma::mat diag_mat_elem_SzSz(dim, sites.size(), arma::fill::zeros);

	arma::vec agp_norm_kin(sites.size(), arma::fill::zeros);
	arma::vec typ_susc_kin(sites.size(), arma::fill::zeros);
	arma::mat diag_mat_elem_kin(dim, sites.size(), arma::fill::zeros);
	arma::vec energies(dim, arma::fill::zeros);

	int Ll = this->L;
	int N = this->grain_size;

	int counter = 0;
	auto U1Hilbert = this->ptr_to_model->get_model_ref().get_hilbert_space();
	auto neighbor_generator = disorder<int>(this->seed);
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
		
		arma::vec agp_norm_Sz_r(sites.size(), arma::fill::zeros);
		arma::vec typ_susc_Sz_r(sites.size(), arma::fill::zeros);
		arma::Mat<element_type> diag_mat_elem_Sz_r(dim, sites.size(), arma::fill::zeros);

		arma::vec agp_norm_SzSz_r(sites.size(), arma::fill::zeros);
		arma::vec typ_susc_SzSz_r(sites.size(), arma::fill::zeros);
		arma::Mat<element_type> diag_mat_elem_SzSz_r(dim, sites.size(), arma::fill::zeros);

		arma::vec agp_norm_kin_r(sites.size(), arma::fill::zeros);
		arma::vec typ_susc_kin_r(sites.size(), arma::fill::zeros);
		arma::Mat<element_type> diag_mat_elem_kin_r(dim, sites.size(), arma::fill::zeros);
		
		for(int i = 0; i < sites.size(); i++)
		{
			int site = sites(i);
			double _agp, _typ_susc, _susc;
			arma::vec tmp;
			start = std::chrono::system_clock::now();
			// arma::Mat<element_type> mat_elem = V * Sz_ops[i] * V.t();
			auto kernel_Sz = [Ll, site](u64 state){ 
				auto [val, tmp11] = operators::sigma_z(state, Ll, site ); 
				return std::make_pair(state, val); 
				};
			auto _operator = QOps::generic_operator<>(this->L, std::move(kernel_Sz), 1.0);
			arma::sp_mat op = arma::real(_operator.to_reduced_matrix(U1Hilbert));
			arma::Mat<element_type> mat_elem = V.t() * op * V;

			std::tie(_agp, _typ_susc, _susc, tmp) = adiabatics::gauge_potential(mat_elem, E, this->L);
			agp_norm_Sz_r(i) = _agp;
			typ_susc_Sz_r(i) = _typ_susc;
			diag_mat_elem_Sz_r.col(i) = arma::diagvec(mat_elem); 
			
    		std::cout << " - - - - - - finished Sz matrix elements for site i = " << sites(i) << "in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
			{
				start = std::chrono::system_clock::now();
				auto kernel_SzSz = [Ll, N, site, &neighbor_generator](u64 state){ 
					int nei = neighbor_generator.uniform_dist<int>(0, N-1);
					auto [val1, tmp22] = operators::sigma_z(state, Ll, site );
					auto [val2, tmp33] = operators::sigma_z(state, Ll, nei );
					return std::make_pair(state, val1 * val2);
					};
				_operator = QOps::generic_operator<>(this->L, std::move(kernel_SzSz), 1.0);
				op = arma::real(_operator.to_reduced_matrix(U1Hilbert));
				mat_elem = V.t() * op * V;

				std::tie(_agp, _typ_susc, _susc, tmp) = adiabatics::gauge_potential(mat_elem, E, this->L);
				agp_norm_SzSz_r(i) = _agp;
				typ_susc_SzSz_r(i) = _typ_susc;
				diag_mat_elem_SzSz_r.col(i) = arma::diagvec(mat_elem); 
				
				std::cout << " - - - - - - finished SzSz matrix elements for site i = " << sites(i) << "in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
				start = std::chrono::system_clock::now();
				auto kernel_kin = [Ll, N, site, &neighbor_generator](u64 state){ 
					int nei = neighbor_generator.uniform_dist<int>(0, N-1);
					auto [spin1, tmp11] = operators::sigma_z(state, Ll, site );
					auto [spin2, tmp22] = operators::sigma_z(state, Ll, nei );
					if(std::real(spin1 * spin2) < 0){
						auto [val1, num] = operators::sigma_x(state, Ll, site );
						auto [val2, num2] = operators::sigma_x(num, Ll, nei );
						return std::make_pair(num2, val1 * val2); 
					} else 
						return std::make_pair(state, cpx(0.0));
					};
				_operator = QOps::generic_operator<>(this->L, std::move(kernel_kin), 1.0);
				op = arma::real(_operator.to_reduced_matrix(U1Hilbert));
				mat_elem = V.t() * op * V;

				std::tie(_agp, _typ_susc, _susc, tmp) = adiabatics::gauge_potential(mat_elem, E, this->L);
				agp_norm_kin_r(i) = _agp;
				typ_susc_kin_r(i) = _typ_susc;
				diag_mat_elem_kin_r.col(i) = arma::diagvec(mat_elem); 
				std::cout << " - - - - - - finished kinetic matrix elements for site i = " << sites(i) << "in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
			}
		}
		// #ifndef MY_MAC
		{
			std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
			createDirs(dir_realis);
			sites.save(arma::hdf5_name(dir_realis + info + ".hdf5", "sites"));
			E.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "energies",   arma::hdf5_opts::append));

			agp_norm_Sz_r.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "AGP/Sz",   arma::hdf5_opts::append));
			agp_norm_SzSz_r.save( arma::hdf5_name(dir_realis + info + ".hdf5", "AGP/SzSz", arma::hdf5_opts::append));
			agp_norm_kin_r.save(  arma::hdf5_name(dir_realis + info + ".hdf5", "AGP/kin",  arma::hdf5_opts::append));

			typ_susc_Sz_r.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "TYP_SUSC/Sz",   arma::hdf5_opts::append));
			typ_susc_SzSz_r.save( arma::hdf5_name(dir_realis + info + ".hdf5", "TYP_SUSC/SzSz", arma::hdf5_opts::append));
			typ_susc_kin_r.save(  arma::hdf5_name(dir_realis + info + ".hdf5", "TYP_SUSC/kin",  arma::hdf5_opts::append));

			diag_mat_elem_Sz_r.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "DIAG_MAT/Sz",   arma::hdf5_opts::append));
			diag_mat_elem_SzSz_r.save( arma::hdf5_name(dir_realis + info + ".hdf5", "DIAG_MAT/SzSz", arma::hdf5_opts::append));
			diag_mat_elem_kin_r.save(  arma::hdf5_name(dir_realis + info + ".hdf5", "DIAG_MAT/kin",  arma::hdf5_opts::append));
		}
		// #endif
		
		agp_norm_Sz += agp_norm_Sz_r;
		typ_susc_Sz += arma::log(typ_susc_Sz_r);
		diag_mat_elem_Sz += diag_mat_elem_Sz_r;

		agp_norm_SzSz += agp_norm_SzSz_r;
		typ_susc_SzSz += arma::log(typ_susc_SzSz_r);
		diag_mat_elem_SzSz += diag_mat_elem_SzSz_r;

		agp_norm_kin += agp_norm_kin_r;
		typ_susc_kin += arma::log(typ_susc_kin_r);
		diag_mat_elem_kin += diag_mat_elem_kin_r;

		energies += E;
		counter++;
		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start_re) << " s - - - - - - " << std::endl; // simulation end
	}
	if(counter == 0) return;
	
	#ifdef MY_MAC
		agp_norm_Sz /= double(counter);
		typ_susc_Sz = arma::exp(typ_susc_Sz / double(counter));
		diag_mat_elem_Sz /= double(counter);

		agp_norm_SzSz /= double(counter);
		typ_susc_SzSz = arma::exp(typ_susc_SzSz / double(counter));
		diag_mat_elem_SzSz /= double(counter);

		agp_norm_kin /= double(counter);
		typ_susc_kin = arma::exp(typ_susc_kin / double(counter));
		diag_mat_elem_kin /= double(counter);

		energies /= double(counter);
		sites.save(arma::hdf5_name(dir + info + ".hdf5", "sites"));
		// agp_norm.save(arma::hdf5_name(dir + info + ".hdf5", "agp norm", arma::hdf5_opts::append));
		// typ_susc.save(arma::hdf5_name(dir + info + ".hdf5", "typical susceptibility", arma::hdf5_opts::append));
		// susc.save(arma::hdf5_name(dir + info + ".hdf5", "susceptibility", arma::hdf5_opts::append));
		energies.save(		arma::hdf5_name(dir + info + ".hdf5", "energies",   arma::hdf5_opts::append));
		agp_norm_Sz.save(	arma::hdf5_name(dir + info + ".hdf5", "AGP/Sz",   arma::hdf5_opts::append));
		agp_norm_SzSz.save( arma::hdf5_name(dir + info + ".hdf5", "AGP/SzSz", arma::hdf5_opts::append));
		agp_norm_kin.save(  arma::hdf5_name(dir + info + ".hdf5", "AGP/kin",  arma::hdf5_opts::append));

		typ_susc_Sz.save(	arma::hdf5_name(dir + info + ".hdf5", "TYP_SUSC/Sz",   arma::hdf5_opts::append));
		typ_susc_SzSz.save( arma::hdf5_name(dir + info + ".hdf5", "TYP_SUSC/SzSz", arma::hdf5_opts::append));
		typ_susc_kin.save(  arma::hdf5_name(dir + info + ".hdf5", "TYP_SUSC/kin",  arma::hdf5_opts::append));

		diag_mat_elem_Sz.save(   arma::hdf5_name(dir + info + ".hdf5", "DIAG_MAT/Sz",   arma::hdf5_opts::append));
		diag_mat_elem_SzSz.save( arma::hdf5_name(dir + info + ".hdf5", "DIAG_MAT/SzSz", arma::hdf5_opts::append));
		diag_mat_elem_kin.save(  arma::hdf5_name(dir + info + ".hdf5", "DIAG_MAT/kin",  arma::hdf5_opts::append));
	#endif
}

/// @brief Calculate AGPs from matrix elements of local operators
void ui::agp_save()
{
	std::string dir = this->saving_dir + "AGP_SAVE" + kPSep;
	// if(this->op > 0) dir += "OtherObservables" + kPSep;
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info();
	const size_t size = dim > 1e5? this->l_steps : dim;

	arma::vec energies(size, arma::fill::zeros);
	arma::vec susc(    size, arma::fill::zeros);
	arma::vec susc_r(  size, arma::fill::zeros);

	int Ll = this->L;
	int N = this->grain_size;
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
		E.save(arma::hdf5_name(dir_realis + info + ".hdf5", "energies"));
		energies += E;

		start = std::chrono::system_clock::now();
		auto kernel_def = [Ll, N](u64 state){ 
					auto [val1, tmp22] = operators::sigma_z(state, Ll, Ll - 1 );
					return std::make_pair(state, val1);
					};
		auto _operator = QOps::generic_operator<>(this->L, std::move(kernel_def), 1.0);
		arma::sp_mat oper = arma::real(_operator.to_matrix(dim));
		
		arma::Mat<element_type> mat_elem = V.t() * oper * V;
		auto [_susc, _susc_r] = adiabatics::gauge_potential_save(mat_elem, E, this->L);

		std::cout << " - - - - - - finished Sz_L matrix elements in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		// #ifndef MY_MAC
		{
			_susc.save(	 arma::hdf5_name(dir_realis + info + ".hdf5", "susc",     arma::hdf5_opts::append));
			_susc_r.save(arma::hdf5_name(dir_realis + info + ".hdf5", "susc_reg", arma::hdf5_opts::append));
		}
		// #endif
		susc += _susc;
		susc_r += _susc_r;
		counter++;
		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start_re) << " s - - - - - - " << std::endl; // simulation end
	}
	if(counter == 0) return;
	
	#ifdef MY_MAC
		susc /= double(counter);
		susc_r /= double(counter);
		energies /= double(counter);

		energies.save(arma::hdf5_name(dir + info + ".hdf5", "energies"));
		susc.save(    arma::hdf5_name(dir + info + ".hdf5", "susc", arma::hdf5_opts::append));
		susc_r.save(  arma::hdf5_name(dir + info + ".hdf5", "susc_reg", arma::hdf5_opts::append));
	#endif
}

/// @brief Calculate matrix elements of local operators
void ui::spectral_function()
{
	std::string dir = this->saving_dir + "SpectralFunctions" + kPSep;
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info();

	const size_t size = dim > 1e5? this->l_steps : dim;

	arma::vec energies(size, arma::fill::zeros);

	int Ll = this->L;
	int N = this->grain_size;

	int counter = 0;
	arma::vec omegax = arma::logspace((std::log10(0.1/dim)), (std::log10( 5 + this->L )), 30 * this->L);
	arma::vec energy_density = arma::regspace(0.05, 0.02, 0.95);

	arma::Mat<element_type> spectral_fun(omegax.size()-1, energy_density.size(), arma::fill::zeros);
	arma::Mat<element_type> spectral_fun_typ(omegax.size()-1, energy_density.size(), arma::fill::zeros);
	arma::Mat<element_type> element_count(omegax.size()-1, energy_density.size(), arma::fill::zeros);
	double window_width = 0.1;
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

		auto i = min_element(begin(E), end(E), [=](double x, double y) {
			return abs(x - E_av) < abs(y - E_av);
		});
		const long Eav_idx = i - begin(E);

		std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
		createDirs(dir_realis);
		E.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "energies"));
		energy_density.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "energy_density",   arma::hdf5_opts::append));

		start = std::chrono::system_clock::now();
		// arma::Mat<element_type> mat_elem = V * Sz_ops[i] * V.t();
		
		auto sites = std::vector<int>( {Ll / 2, Ll} );
		for(int il = 0; il < sites.size(); il++){
			int ell = sites[il];
			auto kernel = [Ll, N, ell](u64 state){ 
				auto [val1, tmp22] = operators::sigma_z(state, Ll, ell );
				return std::make_pair(state, val1);
				};
			auto _operator = QOps::generic_operator<>(this->L, std::move(kernel), 1.0);
			arma::sp_mat opmat = arma::real(_operator.to_matrix(dim));
			arma::Mat<element_type> mat_elem = V.t() * opmat * V;
			std::cout << " - - - - - - finished matrix elements in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
			
			double cutoff = std::sqrt(Ll) / double(dim);
			auto [_susc, _susc_r] = adiabatics::gauge_potential_save(mat_elem, E, this->L, cutoff);

			std::cout << " - - - - - - finished AGP in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
			
			arma::Mat<element_type> _spectral_fun(omegax.size()-1, energy_density.size(), arma::fill::zeros);
			arma::Mat<element_type> _spectral_fun_typ(omegax.size()-1, energy_density.size(), arma::fill::zeros);
			arma::Mat<element_type> _element_count(omegax.size()-1, energy_density.size(), arma::fill::zeros);
			
			const double bandwidth = E(E.size() - 1) - E(0);	
			for(int ii = 0; ii < energy_density.size(); ii++){
				const double eps = energy_density(ii);
				const double energyx = eps * bandwidth + E(0);
				spectrals::preset_omega set_omega(E, window_width, energyx);
				auto [omegas_i, matter] = set_omega.get_matrix_elements(mat_elem);

				for(int k = 0; k < omegax.size() - 1; k++){
					arma::uvec indices = arma::find(omegas_i >= omegax[k] && omegas_i < omegax[k+1]);
					_element_count(k, ii) = indices.size();
					_spectral_fun(k, ii) = arma::accu( matter.rows(indices));
					_spectral_fun_typ(k, ii) = arma::accu( arma::log(matter.rows(indices)) );
				}
			}
			std::cout << " - - - - - - finished Sz_L matrix elements in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
			// #ifndef MY_MAC
			{
				omegax.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "omegas_l=" + std::to_string(ell),   arma::hdf5_opts::append));
				_spectral_fun.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "spectral_fun_l=" + std::to_string(ell),   arma::hdf5_opts::append));
				_spectral_fun_typ.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "log(_spectral_fun_typ)_l=" + std::to_string(ell),   arma::hdf5_opts::append));
				_element_count.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "element_count_l=" + std::to_string(ell),   arma::hdf5_opts::append));

				_susc.save(	 arma::hdf5_name(dir_realis + info + ".hdf5", "susc_l=" + std::to_string(ell),     arma::hdf5_opts::append));
				_susc_r.save(arma::hdf5_name(dir_realis + info + ".hdf5", "susc_reg_l=" + std::to_string(ell), arma::hdf5_opts::append));
			}
		}
		// spectral_fun += _spectral_fun;
		// element_count += _element_count;
		// spectral_fun_typ += _spectral_fun_typ;
		// #endif
		
		// energies += E;
		// counter++;
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


// -------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------- IMPLEMENTATION OF UI

/// @brief Create unique pointer to model with current parameters in class
typename ui::model_pointer ui::create_new_model_pointer(){
    return std::make_unique<QHS::QHamSolver<QuantumSunU1>>(this->L_loc, this->J, this->alfa, this->gamma, this->w, this->h, this->Sz,
																	this->seed, this->grain_size, this->zeta, this->initiate_avalanche, normalize_grain); 
}

/// @brief Reset member unique pointer to model with current parameters in class
void ui::reset_model_pointer(){
    this->ptr_to_model.reset(new QHS::QHamSolver<QuantumSunU1>(this->L_loc, this->J, this->alfa, this->gamma, this->w, this->h, this->Sz,
																	this->seed, this->grain_size, this->zeta, this->initiate_avalanche, normalize_grain)); 
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
    user_interface_dis<QuantumSunU1>::parse_cmd_options(argc, argv);

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
    set_param(h);
    set_param(w);
    set_param(gamma);
    _set_param_(alfa, true); // set always positive

    choosen_option = "-zeta";
    this->set_option(this->zeta, argv, choosen_option);
    
    choosen_option = "-ini_ave";
    this->set_option(this->initiate_avalanche, argv, choosen_option);
    
	choosen_option = "-L";
    this->set_option(this->L_loc, argv, choosen_option, true);

    choosen_option = "-N";
    this->set_option(this->grain_size, argv, choosen_option, true);
	this->L = this->L_loc + this->grain_size;

    choosen_option = "-Sz";
    this->set_option(this->Sz, argv, choosen_option);
    if(this->L % 2 == 1 && this->Sz == 0.0)
        this->Sz = 0.5;

	this->saving_dir = this->dir_prefix + "." + kPSep + "results" + kPSep;
}


/// @brief 
void ui::set_default(){
    user_interface_dis<QuantumSunU1>::set_default();
    this->J = 1.0;
	this->Js = 0.0;
	this->Jn = 1;

	this->zeta = 0.2;
	
	this->gamma = 1.0;
	this->gammas = 0.2;
	this->gamman = 1;

	this->h = 0.0;
	this->hs = 0.1;
	this->hn = 1;

	this->w = 0.01;
	this->ws = 0.0;
	this->wn = 1;

	this->alfa = 1.0;
	this->alfas = 0.02;
	this->alfan = 1;
	
	this->L_loc = 1;
	this->grain_size = 1;
	this->L = this->L_loc + this->grain_size;
	this->Sz = 0.0;
	
    this->initiate_avalanche = 0;
}

/// @brief 
void ui::print_help() const {
    user_interface_dis<QuantumSunU1>::print_help();
    
    printf(" Flags for U(1) Quantum Sun model:\n");
    printSeparated(std::cout, "\t", 20, true, "-Sz", "(float)", "magnetization sector");
    printSeparated(std::cout, "\t", 20, true, "-L", "(int)", "number of localised spins (override above)");
    printSeparated(std::cout, "\t", 20, true, "-J", "(double)", "coupling strength");
    printSeparated(std::cout, "\t", 20, true, "-Js", "(double)", "step in coupling strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-Jn", "(int)", "number of couplings in the sweep");
    printSeparated(std::cout, "\t", 20, true, "-gamma", "(double)", "strength of ergodic bubble");

    printSeparated(std::cout, "\t", 20, true, "-alfa", "(double)", "decay control of coupling with distance");
    printSeparated(std::cout, "\t", 20, true, "-alfas", "(double)", "step in decay strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-alfan", "(int)", "number of values in the sweep");

    printSeparated(std::cout, "\t", 20, true, "-h", "(double)", "uniform field on spins");
    printSeparated(std::cout, "\t", 20, true, "-hs", "(double)", "step in field strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-hn", "(int)", "number of field values in the sweep");

    printSeparated(std::cout, "\t", 20, true, "-w", "(double)", "disorder bandwidth on localized spins");
    printSeparated(std::cout, "\t", 20, true, "-ws", "(double)", "step in disorder strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-wn", "(int)", "number of disorder in the sweep");

    printSeparated(std::cout, "\t", 20, true, "-zeta", "(double)", "randomness in position for coupling to grain");
    printSeparated(std::cout, "\t", 20, true, "-N", "(int)", "size of random grain (number of spins inside grain)");
    printSeparated(std::cout, "\t", 20, true, "-ini_ave", "(boolean)", "initiate avalanche by hand");
	std::cout << std::endl;
}

/// @brief 
void ui::printAllOptions() const{
    user_interface_dis<QuantumSunU1>::printAllOptions();
	std::cout << "U(1) QUANTUM SUN:\n\t\t" << "H = \u03B3R + J \u03A3_i \u03B1^{u_i} S^x_ni S^x_i+1 + S^y_ni S^y_i+1 +";
	std::cout << "\u03A3_i h_i S^z_i" << std::endl << std::endl;
	std::cout << "u_i \u03B5 [j - \u03B6, j + \u03B6]"  << std::endl;
	if constexpr (scaled_disorder == 1)
    	std::cout << "h_i \u03B5 [h - W', h + W']\t W'=2w/L" << std::endl;
	else
		std::cout << "h_i \u03B5 [h - w, h + w]" << std::endl;
	

	std::cout << "------------------------------ CHOSEN U(1) QuantumSun OPTIONS:" << std::endl;
    std::cout 
		  << "total Sz = " << this->Sz << std::endl
		  << "num of spins = " << this->L_loc << std::endl
		  << "grain size = " << this->grain_size << std::endl
		  << "J  = " << this->J << std::endl
		  << "Jn = " << this->Jn << std::endl
		  << "Js = " << this->Js << std::endl
		  << "\u03B3 = " << this->gamma << std::endl
		  << "h  = " << this->h << std::endl
		  << "hs = " << this->hs << std::endl
		  << "hn = " << this->hn << std::endl;
	if constexpr (scaled_disorder == 1)
    	std::cout << "W'=2w/L= " << this->w << std::endl;
	else
		std::cout << "w = " << this->w << std::endl;
		
	std::cout << "ws = " << this->ws << std::endl
		  << "wn = " << this->wn << std::endl
		  << "\u03B1  = " << this->alfa << std::endl
		  << "\u03B1s = " << this->alfas << std::endl
		  << "\u03B1n = " << this->alfan << std::endl
		  << "\u03B6 = " << this->zeta << std::endl
		  << "initialize avelanche = " << this->initiate_avalanche << std::endl;
}   

/// @brief 
/// @param skip 
/// @param sep 
/// @return 
std::string ui::set_info(std::vector<std::string> skip, std::string sep) const
{
        std::string name = "L=" + std::to_string(this->L_loc) + \
            ",N=" + std::to_string(this->grain_size) + \
            ",J=" + to_string_prec(this->J) + \
            ",g=" + to_string_prec(this->gamma);
        if(this->alfa < 1.0) name += ",zeta=" + to_string_prec(this->zeta);
        
		name += ",alfa=" + to_string_prec(this->alfa) + \
            ",h=" + to_string_prec(this->h);
        if constexpr (scaled_disorder == 1)
			name += ",W'=" + to_string_prec(this->w);
		else
			name += ",w=" + to_string_prec(this->w);
		
		name += ",Sz=" + to_string_prec(this->Sz);
		
        if(this->initiate_avalanche) name += ",ini_ave";

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