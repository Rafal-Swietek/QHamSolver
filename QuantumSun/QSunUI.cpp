#include "includes/QSunUI.hpp"

int outer_threads = 1;
int num_of_threads = 1;

bool normalize_grain = 1;

namespace QSunUI{

void ui::make_sim(){
    printAllOptions();
	clk::time_point start = std::chrono::system_clock::now();
    
	std::cout << "Using num  of threads = " << omp_get_num_threads() << std::endl;
	this->ptr_to_model = this->create_new_model_pointer();
	auto Hamil = this->ptr_to_model->get_hamiltonian();
	this->l_steps = 0.05 * Hamil.n_cols;
	if(this->l_steps > 200)
		this->l_steps = 200;
	auto polfed = polfed::POLFED<ui::element_type>(Hamil, this->l_steps, this->l_bundle, -1, this->tol, 0.2, this->seed, true);
	auto [E, V] = polfed.eig();
	return;


	// auto Hamil = this->ptr_to_model->get_hamiltonian();
	// arma::sp_mat H = Hamil;
	// auto polfed = polfed::POLFED<ui::element_type>(H, this->l_steps, this->l_bundle, -1, this->tol, 0.2, this->seed, true);
	// arma::vec E;
	// arma::mat V;
	// std::tie(E, V) = polfed.eig();
	// E = arma::sort(E);
	
	// std::cout << "-------> POLFED finished in " << tim_s(start) << " s" << std::endl;
	// arma::vec E_ED;
	// arma::eig_sym(E_ED, V, arma::mat(Hamil));
	// double Emin = arma::min(E);
	// auto i = std::min_element(std::begin(E_ED), std::end(E_ED), [=](double x, double y) {
	// 	return std::abs(x - Emin) < std::abs(y - Emin);
	// 	});
	// u64 idx = i - std::begin(E_ED);

	// start = std::chrono::system_clock::now();
	// arma::eigs_opts _opts;
	// _opts.tol = 0;
	// _opts.maxiter = 30 * this->l_steps;
	// arma::vec E2;
	// arma::eigs_sym(E2, V, Hamil, this->l_steps, arma::trace(Hamil) / double(Hamil.n_cols), _opts);
	// std::cout << "-------> ARMA::LU finished in " << tim_s(start) << " s" << std::endl;
	// for(int i =0; i < E.size(); i++){
	// 	printSeparated(std::cout, "\t", 16, true, E(i), E2(i), E_ED(i + idx), std::abs(E(i) - E2(i)));
	// }

	// return;

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
		correlators();
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

								auto Hamil = this->ptr_to_model->get_hamiltonian();
								this->l_steps = 0.05 * Hamil.n_cols;
								if(this->l_steps > 200)
									this->l_steps = 200;
								auto polfed = polfed::POLFED<ui::element_type>(Hamil, this->l_steps, this->l_bundle, -1, this->tol, 0.2, this->seed, true);
								auto [E, V] = polfed.eig();
								// eigenstate_entanglement();
								// matrix_elements();
								// spectral_form_factor();
								// diagonalize();
								// survival_probability();
								//entanglement_evolution();
								//average_sff();
								std::cout << "\t\t - - - - - - FINISHED ITERATION IN : " << tim_s(start_loop) << " seconds\n\t\t\t Total time : " << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
						}}}}}}
        std::cout << "Add default function" << std::endl;
	}
	std::cout << " - - - - - - FINISHED CALCULATIONS IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
}



// ------------------------------------------------ OVERRIDEN METHODS

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
		
		u64 num = (u64)std::min(0.1 * dim, 500.0);
		auto Eav_idx = this->ptr_to_model->E_av_idx;
		const u64 idx_min = Eav_idx - (u64)num / 2.0;
		const u64 idx_max = Eav_idx + (u64)num / 2.0;

		std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
		createDirs(dir_realis);
		sites.save(arma::hdf5_name(dir_realis + info + ".hdf5", "sites"));
		E.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "energies",   arma::hdf5_opts::append));
		
		arma::vec agp_norm_Sz_r(sites.size(), arma::fill::zeros);
		arma::vec typ_susc_Sz_r(sites.size(), arma::fill::zeros);
		arma::Mat<element_type> diag_mat_elem_Sz_r(dim, sites.size(), arma::fill::zeros);

		arma::vec agp_norm_Sx_r(sites.size(), arma::fill::zeros);
		arma::vec typ_susc_Sx_r(sites.size(), arma::fill::zeros);
		arma::Mat<element_type> diag_mat_elem_Sx_r(dim, sites.size(), arma::fill::zeros);

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
			arma::sp_mat op = arma::real(_operator.to_matrix(dim));
			arma::Mat<element_type> mat_elem = V.t() * op * V;
			arma::Mat<element_type> _submat_ = mat_elem.submat(idx_min, idx_min, idx_max -1, idx_max - 1);
			_submat_.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "MAT_ELEM/Sz_i=" + std::to_string(site),   arma::hdf5_opts::append));
			std::tie(_agp, _typ_susc, _susc, tmp) = adiabatics::gauge_potential(mat_elem, E, this->L);
			agp_norm_Sz_r(i) = _agp;
			typ_susc_Sz_r(i) = _typ_susc;
			diag_mat_elem_Sz_r.col(i) = arma::diagvec(mat_elem);
			
    		std::cout << " - - - - - - finished Sz matrix elements for site i = " << sites(i) << "in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
			
			start = std::chrono::system_clock::now();
			// arma::Mat<element_type> mat_elem = V * Sz_ops[i] * V.t();
			auto kernel_Sx = [Ll, site](u64 state){ 
				auto [val, num] = operators::sigma_x(state, Ll, site ); 
				return std::make_pair(num, val); 
				};
			_operator = QOps::generic_operator<>(this->L, std::move(kernel_Sx), 1.0);
			op = arma::real(_operator.to_matrix(dim));
			mat_elem = V.t() * op * V;
			_submat_ = mat_elem.submat(idx_min, idx_min, idx_max -1, idx_max - 1);
			_submat_.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "MAT_ELEM/Sx_i=" + std::to_string(site),   arma::hdf5_opts::append));
			std::tie(_agp, _typ_susc, _susc, tmp) = adiabatics::gauge_potential(mat_elem, E, this->L);
			agp_norm_Sx_r(i) = _agp;
			typ_susc_Sx_r(i) = _typ_susc;
			diag_mat_elem_Sx_r.col(i) = arma::diagvec(mat_elem);
			
    		std::cout << " - - - - - - finished Sx matrix elements for site i = " << sites(i) << "in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
			{
				start = std::chrono::system_clock::now();
				auto kernel_SzSz = [Ll, N, site, &neighbor_generator](u64 state){ 
					int nei = neighbor_generator.uniform_dist<int>(0, N-1);
					auto [val1, tmp22] = operators::sigma_z(state, Ll, site );
					auto [val2, tmp33] = operators::sigma_z(state, Ll, nei );
					return std::make_pair(state, val1 * val2);
					};
				_operator = QOps::generic_operator<>(this->L, std::move(kernel_SzSz), 1.0);
				op = arma::real(_operator.to_matrix(dim));
				mat_elem = V.t() * op * V;
				_submat_ = mat_elem.submat(idx_min, idx_min, idx_max -1, idx_max - 1);
				_submat_.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "MAT_ELEM/SzSz_i=" + std::to_string(site),   arma::hdf5_opts::append));
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
				op = arma::real(_operator.to_matrix(dim));
				mat_elem = V.t() * op * V;
				_submat_ = mat_elem.submat(idx_min, idx_min, idx_max -1, idx_max - 1);
				_submat_.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "MAT_ELEM/kin_i=" + std::to_string(site),   arma::hdf5_opts::append));
				std::tie(_agp, _typ_susc, _susc, tmp) = adiabatics::gauge_potential(mat_elem, E, this->L);
				agp_norm_kin_r(i) = _agp;
				typ_susc_kin_r(i) = _typ_susc;
				diag_mat_elem_kin_r.col(i) = arma::diagvec(mat_elem); 
				std::cout << " - - - - - - finished kinetic matrix elements for site i = " << sites(i) << "in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
			}
		}
		// #ifndef MY_MAC
		{
			agp_norm_Sz_r.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "AGP/Sz",   arma::hdf5_opts::append));
			agp_norm_Sx_r.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "AGP/Sx",   arma::hdf5_opts::append));
			agp_norm_SzSz_r.save( arma::hdf5_name(dir_realis + info + ".hdf5", "AGP/SzSz", arma::hdf5_opts::append));
			agp_norm_kin_r.save(  arma::hdf5_name(dir_realis + info + ".hdf5", "AGP/kin",  arma::hdf5_opts::append));

			typ_susc_Sz_r.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "TYP_SUSC/Sz",   arma::hdf5_opts::append));
			typ_susc_Sx_r.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "TYP_SUSC/Sx",   arma::hdf5_opts::append));
			typ_susc_SzSz_r.save( arma::hdf5_name(dir_realis + info + ".hdf5", "TYP_SUSC/SzSz", arma::hdf5_opts::append));
			typ_susc_kin_r.save(  arma::hdf5_name(dir_realis + info + ".hdf5", "TYP_SUSC/kin",  arma::hdf5_opts::append));

			diag_mat_elem_Sz_r.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "DIAG_MAT/Sz",   arma::hdf5_opts::append));
			diag_mat_elem_Sx_r.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "DIAG_MAT/Sx",   arma::hdf5_opts::append));
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

/// @brief Calculate matrix elements of local operators
void ui::correlators()
{
	std::string dir = this->saving_dir + "Correlators" + kPSep;
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info();

	// arma::vec sites = arma::linspace(0, this->L-1, this->L);
	const int Lhalf = this->L / 2;
	std::vector<std::pair<int,int>> site_pairs = std::vector<std::pair<int,int>>(
			{std::make_pair(this->grain_size, this->grain_size + 1), std::make_pair(this->grain_size, this->grain_size + 2),
			std::make_pair(this->grain_size, this->L - 1), std::make_pair(this->grain_size + 1, this->L - 1),
			std::make_pair(Lhalf, Lhalf + 1), std::make_pair(Lhalf, this->L - 1)}
			);
	std::cout << "site pairs:" << std::endl;
	for(auto& pair : site_pairs)
		std::cout << pair.first << " " << pair.second << std::endl;

	const double chi = 0.341345;

	const double wH = std::sqrt(this->L) / (chi * dim);
	double tH = 1. / wH;
	double r1 = 0.0, r2 = 0.0;
	int time_end = (int)std::ceil(std::log10(5 * tH));
	time_end = (time_end / std::log10(tH) < 1.5) ? time_end + 1 : time_end;

	arma::vec times = arma::logspace(-2, time_end, this->num_of_points);

	arma::vec agp_norm(site_pairs.size(), arma::fill::zeros);
	arma::vec typ_susc(site_pairs.size(), arma::fill::zeros);
	arma::mat diag_mat_elem(dim, site_pairs.size(), arma::fill::zeros);

	arma::vec energies(dim, arma::fill::zeros);

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
    	this->ptr_to_model->diagonalization();
		auto Eav_idx = this->ptr_to_model->E_av_idx;

		std::cout << " - - - - - - finished diagonalization in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();
		
		const arma::vec E = this->ptr_to_model->get_eigenvalues();
		const auto& V = this->ptr_to_model->get_eigenvectors();
		

		std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
		createDirs(dir_realis);
		E.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "energies",   arma::hdf5_opts::append));
		
		arma::vec LTA_r(site_pairs.size() + 1, arma::fill::zeros);
		arma::vec agp_norm_r(site_pairs.size() + 1, arma::fill::zeros);
		arma::vec typ_susc_r(site_pairs.size() + 1, arma::fill::zeros);
		arma::Mat<element_type> diag_mat_elem_r(dim, site_pairs.size(), arma::fill::zeros);

		const double window_width = 0.001 * this->L;
		spectrals::preset_omega set_omega(E, window_width, E(Eav_idx));
		arma::vec omegas(set_omega.num_of_omegas, arma::fill::zeros);
		arma::Mat<element_type> spectral_funs(set_omega.num_of_omegas, site_pairs.size(), arma::fill::zeros);

		arma::mat quench_FM(times.size(), site_pairs.size(), arma::fill::zeros);
		arma::mat quench_AFM(times.size(), site_pairs.size(), arma::fill::zeros);
		arma::mat quench_spiral(times.size(), site_pairs.size(), arma::fill::zeros);
		arma::mat quench_random(times.size(), site_pairs.size(), arma::fill::zeros);
		arma::mat autocorr(times.size(), site_pairs.size(), arma::fill::zeros);

		arma::cx_mat psi_FM(dim, times.size(), arma::fill::zeros);
		arma::cx_mat psi_AFM(dim, times.size(), arma::fill::zeros);
		arma::cx_mat psi_spiral(dim, times.size(), arma::fill::zeros);
		arma::cx_mat psi_random(dim, times.size(), arma::fill::zeros);
		u64 idx = (dim - 1) / 3;

		start = std::chrono::system_clock::now();
		arma::cx_vec random_state = this->random_product_state();
		arma::mat R(2, 2);
		arma::vec spiral_state = up;
		for (int j = 1; j < this->L; j++)
		{
			auto the = pi / this->L * double(j);
			R(0, 0) = std::cos(the); R(1, 1) = std::cos(the);
			R(0, 1) = std::sin(the); R(1, 0) = -std::sin(the);
			spiral_state = arma::kron(spiral_state, R * up);
		}
		spiral_state = arma::normalise(spiral_state);
		arma::vec coeff_spiral(dim);
		arma::cx_vec coeff_random(dim);
	#pragma omp parallel for
		for(long alfa = 0; alfa < dim; alfa++)
		{
			auto state = V.col(alfa);
			coeff_spiral(alfa) = dot_prod(state, spiral_state);
			coeff_random(alfa) = dot_prod(state, random_state);
		}
		std::cout << " - - - - - - finished preparing initial states FM, AFM, spiral, random product in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();
	#pragma omp parallel for
		for(long t_idx = 0; t_idx < times.size(); t_idx++)
		{
			double time = times(t_idx);
			for(long alfa = 0; alfa < dim; alfa++)
			{
				auto state = V.col(alfa);
				psi_FM.col(t_idx) += std::exp(-1i * time * E(alfa)) * state * state(0);
				psi_AFM.col(t_idx) += std::exp(-1i * time * E(alfa)) * state * state(idx);
				psi_spiral.col(t_idx) += std::exp(-1i * time * E(alfa)) * state * coeff_spiral(alfa);
				psi_random.col(t_idx) += std::exp(-1i * time * E(alfa)) * state * coeff_random(alfa);
			}
		}
		arma::vec quench_E(4);
		auto H = this->ptr_to_model->get_hamiltonian();
		quench_E(0) = H(0, 0);
		quench_E(1) = H(idx, idx);
		quench_E(2) = arma::cdot(spiral_state, H * spiral_state);
		quench_E(3) = std::real(arma::cdot(random_state, H * random_state));

		std::cout << " - - - - - - finished preparing initial states for all times in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		for(int i = 0; i < site_pairs.size(); i++)
		{
			int site_1 = site_pairs[i].first;
			int site_2 = site_pairs[i].second;
			double _agp, _typ_susc, _susc;
			arma::vec tmp;
			start = std::chrono::system_clock::now();
			// arma::Mat<element_type> mat_elem = V * Sz_ops[i] * V.t();
			auto kernel = [Ll, N, site_1, site_2](u64 state){ 
				auto [val1, tmp22] = operators::sigma_z(state, Ll, site_1 );
				auto [val2, tmp33] = operators::sigma_z(state, Ll, site_2 );
				return std::make_pair(state, val1 * val2);
				};
			auto _operator = QOps::generic_operator<>(this->L, std::move(kernel), 1.0);
			arma::sp_mat op = arma::real(_operator.to_matrix(dim));
			arma::Mat<element_type> mat_elem = V.t() * op * V;
			std::tie(_agp, _typ_susc, _susc, tmp) = adiabatics::gauge_potential(mat_elem, E, this->L);
			agp_norm_r(i) = _agp;
			typ_susc_r(i) = _typ_susc;
			diag_mat_elem_r.col(i) = arma::diagvec(mat_elem);
			
			auto [omegas_i, matter] = set_omega.get_matrix_elements(mat_elem);
			omegas = omegas_i;
			spectral_funs.col(i) = matter;

    		std::cout << " - - - - - - finished matrix elements for sites: i=" << site_1 << ", j=" << site_2 << " in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
			start = std::chrono::system_clock::now();
		#pragma omp parallel for
			for(long t_idx = 0; t_idx < times.size(); t_idx++)
			{
				quench_FM(t_idx, i) = std::real(arma::cdot(psi_FM.col(t_idx), op * psi_FM.col(t_idx)));
				quench_AFM(t_idx, i) = std::real(arma::cdot(psi_AFM.col(t_idx), op * psi_AFM.col(t_idx)));
				quench_spiral(t_idx, i) = std::real(arma::cdot(psi_spiral.col(t_idx), op * psi_spiral.col(t_idx)));
				quench_random(t_idx, i) = std::real(arma::cdot(psi_random.col(t_idx), op * psi_random.col(t_idx)));
			}
    		std::cout << " - - - - - - finished time evolution for sites: i=" << site_1 << ", j=" << site_2 << " in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
			start = std::chrono::system_clock::now();
			auto [_autocorr, LTA] = spectrals::autocorrelation_function(mat_elem, E, times);
			autocorr.col(i) = _autocorr;
			LTA_r(i) = LTA;
    		std::cout << " - - - - - - finished auto correlator time evolution for sites: i=" << site_1 << ", j=" << site_2 << " in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		}
		start = std::chrono::system_clock::now();
		// arma::Mat<element_type> mat_elem = V * Sz_ops[i] * V.t();
		auto kernel = [Ll, N](u64 state){ 
			auto [val1, tmp22] = operators::sigma_z(state, Ll, Ll - 1 );
			return std::make_pair(state, val1);
			};
		auto _operator = QOps::generic_operator<>(this->L, std::move(kernel), 1.0);
		arma::sp_mat op = arma::real(_operator.to_matrix(dim));
		arma::Mat<element_type> mat_elem = V.t() * op * V;
		auto [_agp, _typ_susc, _susc, tmp] = adiabatics::gauge_potential(mat_elem, E, this->L);
		agp_norm_r(site_pairs.size()) = _agp;
		typ_susc_r(site_pairs.size()) = _typ_susc;
		arma::vec diag_mat_elem_Sz_r = arma::diagvec(mat_elem);
		auto [omegas_i, matter] = set_omega.get_matrix_elements(mat_elem);

		std::cout << " - - - - - - finished Sz_L matrix elements in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();
		arma::vec quench_FM_Sz(times.size(), arma::fill::zeros);
		arma::vec quench_AFM_Sz(times.size(), arma::fill::zeros);
		arma::vec quench_spiral_Sz(times.size(), arma::fill::zeros);
		arma::vec quench_random_Sz(times.size(), arma::fill::zeros);
	#pragma omp parallel for
		for(long t_idx = 0; t_idx < times.size(); t_idx++)
		{
			quench_FM_Sz(t_idx) = std::real(arma::cdot(psi_FM.col(t_idx), op * psi_FM.col(t_idx)));
			quench_AFM_Sz(t_idx) = std::real(arma::cdot(psi_AFM.col(t_idx), op * psi_AFM.col(t_idx)));
			quench_spiral_Sz(t_idx) = std::real(arma::cdot(psi_spiral.col(t_idx), op * psi_spiral.col(t_idx)));
			quench_random_Sz(t_idx) = std::real(arma::cdot(psi_random.col(t_idx), op * psi_random.col(t_idx)));
		}
		std::cout << " - - - - - - finished time evolution for Sz_L in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();
		auto [autocorr_Sz, LTA_Sz] = spectrals::autocorrelation_function(mat_elem, E, times);
		LTA_r(site_pairs.size()) = LTA_Sz;
		std::cout << " - - - - - - finished auto correlator time evolution for Sz_L in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		// #ifndef MY_MAC
		{
			agp_norm_r.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "agp",   arma::hdf5_opts::append));
			typ_susc_r.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "typ_susc",   arma::hdf5_opts::append));
			diag_mat_elem_r.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "diag_mat",   arma::hdf5_opts::append));
			omegas.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "omegas",   arma::hdf5_opts::append));
			spectral_funs.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "spectral_funs",   arma::hdf5_opts::append));

			matter.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "spectral_Sz_L",   arma::hdf5_opts::append));
			diag_mat_elem_Sz_r.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "diag_mat_Sz_L",   arma::hdf5_opts::append));
			quench_FM_Sz.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench_FM_Sz_L",   arma::hdf5_opts::append));
			quench_AFM_Sz.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench_AFM_Sz_L",   arma::hdf5_opts::append));
			quench_spiral_Sz.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench_spiral_Sz_L",   arma::hdf5_opts::append));
			quench_random_Sz.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench_random_Sz_L",   arma::hdf5_opts::append));
			autocorr_Sz.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "autocorr_Sz_L",   arma::hdf5_opts::append));

			times.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "times",   arma::hdf5_opts::append));
			quench_FM.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench_FM",   arma::hdf5_opts::append));
			quench_AFM.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench_AFM",   arma::hdf5_opts::append));
			quench_spiral.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench_spiral",   arma::hdf5_opts::append));
			quench_random.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench_random",   arma::hdf5_opts::append));
			autocorr.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "autocorr",   arma::hdf5_opts::append));
			LTA_r.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "LTA",   arma::hdf5_opts::append));

			quench_E.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench_energy",   arma::hdf5_opts::append));
		}
		// #endif
		
		// agp_norm += agp_norm_r;
		// typ_susc += arma::log(typ_susc_r);
		// diag_mat_elem += diag_mat_elem_r;

		// energies += E;
		// counter++;
		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start_re) << " s - - - - - - " << std::endl; // simulation end
	}
	if(counter == 0) return;
	
	// #ifdef MY_MAC
	// 	agp_norm /= double(counter);
	// 	typ_susc = arma::exp(typ_susc / double(counter));
	// 	diag_mat_elem /= double(counter);

	// 	energies /= double(counter);
	// 	energies.save(		arma::hdf5_name(dir + info + ".hdf5", "energies",   arma::hdf5_opts::append));
	// 	agp_norm.save(	arma::hdf5_name(dir + info + ".hdf5", "agp",   arma::hdf5_opts::append));
	// 	typ_susc.save(	arma::hdf5_name(dir + info + ".hdf5", "typ_susc",   arma::hdf5_opts::append));
	// 	diag_mat_elem.save(   arma::hdf5_name(dir + info + ".hdf5", "diag_mat",   arma::hdf5_opts::append));
	// #endif
}


// -------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------- IMPLEMENTATION OF UI

/// @brief Create unique pointer to model with current parameters in class
typename ui::model_pointer ui::create_new_model_pointer(){
    return std::make_unique<QHS::QHamSolver<QuantumSun>>(this->L_loc, this->J, this->alfa, this->gamma, this->w, this->h, 
																	this->seed, this->grain_size, this->zeta, this->initiate_avalanche, normalize_grain); 
}

/// @brief Reset member unique pointer to model with current parameters in class
void ui::reset_model_pointer(){
    this->ptr_to_model.reset(new QHS::QHamSolver<QuantumSun>(this->L_loc, this->J, this->alfa, this->gamma, this->w, this->h, 
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
    user_interface_dis<QuantumSun>::parse_cmd_options(argc, argv);

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
	if constexpr (conf_disorder == 1)
    	this->saving_dir = this->dir_prefix + "results_conf_dis" + kPSep;
	else
    	this->saving_dir = this->dir_prefix + "results" + kPSep;
}


/// @brief 
void ui::set_default(){
    user_interface_dis<QuantumSun>::set_default();
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
	
    this->initiate_avalanche = 0;
}

/// @brief 
void ui::print_help() const {
    user_interface_dis<QuantumSun>::print_help();
    
    printf(" Flags for Quantum Sun model:\n");
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
    user_interface_dis<QuantumSun>::printAllOptions();
	std::cout << "QUANTUM SUN:\n\t\t" << "H = \u03B3R + J \u03A3_i \u03B1^{u_i} S^x_i S^x_i+1 + ";
	if constexpr (conf_disorder == 1)
    	std::cout << "\u03A3_n h_n |n><n|" << std::endl << std::endl;
	else
		std::cout << "\u03A3_i h_i S^z_i" << std::endl << std::endl;
	std::cout << "u_i \u03B5 [j - \u03B6, j + \u03B6]"  << std::endl;
	if constexpr (scaled_disorder == 1)
    	std::cout << "h_i \u03B5 [h - W', h + W']\t W'=2w/L" << std::endl;
	else
		std::cout << "h_i \u03B5 [h - w, h + w]" << std::endl;
	

	std::cout << "------------------------------ CHOSEN QuantumSun OPTIONS:" << std::endl;
    std::cout 
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
// #deinfe for greek alfabet

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