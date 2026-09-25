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
	// std::string dir = this->saving_dir + "MatrixElements" + kPSep;
	std::string dir = this->saving_dir + "SpectralsSiteResolved" + kPSep;
	// std::string dir = this->saving_dir + "Spectrals_IDK_what" + kPSep;

	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info();

	// arma::vec sites = arma::linspace(0, this->L-1, this->L);
	arma::vec sites = arma::linspace(0, this->L-1, this->L);
	// arma::Col<int> sites = arma::Col<int>({(int)this->L - 1});

	auto _hilbert_space = this->ptr_to_model->get_model_ref().get_hilbert_space();
	int Ll = this->L;
	int N = this->grain_size;

	arma::vec omegax = arma::logspace(int(std::log10(0.1/dim)), int(std::log10( 5 + this->L )), 10 * this->L);
	const arma::vec energy_density = arma::regspace(0.05, 0.02, 0.95);
	const double chi = 0.341345;

	const double wH = std::sqrt(this->L) / (chi * dim);
	double tH = 1. / wH;
	double r1 = 0.0, r2 = 0.0;
	int time_end = (int)std::ceil(std::log10(50 * tH));
	time_end = (time_end / std::log10(tH) < 2.5) ? time_end + 3 : time_end;

	arma::vec times = arma::logspace(-2, time_end, 2000);

	auto calculate_ratio_variance = [this](arma::mat mat_in){ 
			arma::vec diag = mat_in.diag();
			arma::uword D = mat_in.n_rows;
			arma::vec offdiag(D * (D - 1));
			arma::uword k = 0;
			for (arma::uword i = 0; i < D; ++i)
				for (arma::uword j = 0; j < D; ++j)
					if (i != j)
						offdiag(k++) = mat_in(i, j);

			return arma::var(diag) / arma::var(offdiag);
	};

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

		arma::vec dis_array = this->ptr_to_model->get_model_ref().get_disorder();
		
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
		dis_array.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "disorder",   arma::hdf5_opts::append));
		E.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "energies",   arma::hdf5_opts::append));
		
		arma::Mat<element_type> agp_norm_Sz_r(dim, sites.size(), arma::fill::zeros);
		arma::Mat<element_type> typ_susc_Sz_r(dim, sites.size(), arma::fill::zeros);
		arma::Mat<element_type> diag_mat_elem_Sz_r(dim, sites.size(), arma::fill::zeros);
		
		arma::Mat<element_type> agp_norm_Sx_r(dim, sites.size(), arma::fill::zeros);
		arma::Mat<element_type> typ_susc_Sx_r(dim, sites.size(), arma::fill::zeros);
		arma::Mat<element_type> diag_mat_elem_Sx_r(dim, sites.size(), arma::fill::zeros);
		
		arma::vec Hdiagonal = arma::diagvec( this->ptr_to_model->get_dense_hamiltonian() );

		double E_av = arma::trace(E) / double(dim);
		auto i = min_element(begin(Hdiagonal), end(Hdiagonal), [=](double x, double y) {
			return abs(x - E_av) < abs(y - E_av);
		});
		const u64 idx_state = i - begin(Hdiagonal);
		double quench_E = Hdiagonal(idx_state);

		arma::Col<element_type> coeff = V.row(idx_state).t();

        double E_min_pred = -2. * this->L;
        double E_max_pred =  2. * this->L;
        double _eta = 0.04;

        arma::vec energies = arma::linspace(E_min_pred - 5 * _eta, E_max_pred + 5 * _eta, 3000);
        arma::vec DOS(energies.size(), arma::fill::zeros);
        arma::vec LDOS(energies.size(), arma::fill::zeros);
        arma::vec GAP_RATIO(energies.size(), arma::fill::zeros);
		double norm_inv = 1. / std::sqrt(constants<double>::two_pi * _eta*_eta);

	#pragma omp for schedule(dynamic)
        for(long e_idx = 0; e_idx < energies.size(); e_idx++){
            // doubl
            // ldos(e_idx) += arma::sum(arma::square(coeff.t()) % _eta / ( arma::square(Esym))
            for(long alfa = 0; alfa < E.size(); alfa++){
                double om = E(alfa) - energies(e_idx);
                double gauss = std::exp( - om * om / (2.*_eta * _eta) ) * norm_inv;
                DOS(e_idx) += gauss;
                LDOS(e_idx) += gauss * std::norm(coeff(alfa));
            }
            double E_lower = energies(e_idx) - 2 * _eta;
            double E_upper = energies(e_idx) + 2 * _eta;
            arma::uvec idx = arma::find(E > E_lower && E < E_upper);
            if(idx.size() > 1){
                u64 idx_start = idx.front();
                u64 idx_stop  = idx.back();
                arma::vec gaps = arma::diff(E.rows(idx_start, idx_stop));
                int counter = 0;
                for(long id = 0; id < gaps.size()-1; id++){
                    GAP_RATIO(e_idx) += std::min(gaps(id), gaps(id+1)) / std::max(gaps(id), gaps(id+1));
                    counter++;
                }
                GAP_RATIO(e_idx) /= double(counter);
            }
        }
        coeff.save(arma::hdf5_name(dir_realis + info + ".hdf5", "coeff", arma::hdf5_opts::append));
        DOS.save(arma::hdf5_name(dir_realis + info + ".hdf5", "dos", arma::hdf5_opts::append));
        LDOS.save(arma::hdf5_name(dir_realis + info + ".hdf5", "LDOS", arma::hdf5_opts::append));
        GAP_RATIO.save(arma::hdf5_name(dir_realis + info + ".hdf5", "GAP_RATIO", arma::hdf5_opts::append));
		for(int i = 0; i < sites.size(); i++)
		{
			int site = sites(i);
			// double _agp, _typ_susc, _susc;
			arma::vec _susc, _susc_r;
			start = std::chrono::system_clock::now();
			// arma::Mat<element_type> mat_elem = V * Sz_ops[i] * V.t();
			auto kernel_Sz = [Ll, site](u64 state){ 
				auto [val, tmp11] = operators::sigma_z<double>(state, Ll, site ); 
				return std::make_pair(state, val); 
			};
			auto _operator = QOps::generic_operator<double>(this->L, std::move(kernel_Sz), 1.0);
			arma::sp_mat op_mat = _operator.to_reduced_matrix(_hilbert_space);
			double HSnorm = arma::trace(op_mat * op_mat) / double(dim);
			op_mat = op_mat / std::sqrt(HSnorm);

			arma::Mat<element_type> mat_elem = V.t() * op_mat * V;
			double variance_ratio = calculate_ratio_variance(mat_elem);
			arma::vec( {variance_ratio} ).save(   arma::hdf5_name(dir_realis + info + ".hdf5", "j=" + std::to_string(site) + "/variance_ratio_full",   arma::hdf5_opts::append));
			arma::Mat<element_type> _submat_ = mat_elem.submat(idx_min, idx_min, idx_max -1, idx_max - 1);
			variance_ratio = calculate_ratio_variance(_submat_);
			arma::vec( {variance_ratio} ).save(   arma::hdf5_name(dir_realis + info + ".hdf5", "j=" + std::to_string(site) + "/variance_ratio_submat",   arma::hdf5_opts::append));
			
			// _submat_.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "MAT_ELEM/Sz_i=" + std::to_string(site),   arma::hdf5_opts::append));
			// std::tie(_agp, _typ_susc, _susc, tmp) = adiabatics::gauge_potential(mat_elem, E, this->L);
			std::tie(_susc, _susc_r) = adiabatics::gauge_potential_save(mat_elem, E);
			agp_norm_Sz_r.col(i) = _susc;
			typ_susc_Sz_r.col(i) = _susc_r;
			diag_mat_elem_Sz_r.col(i) = arma::diagvec(mat_elem);
			
    		std::cout << " - - - - - - finished Sz matrix elements for site i = " << sites(i) << "in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
			start = std::chrono::system_clock::now();
			arma::Mat<element_type> _spectral_fun_eps(omegax.size()-1, energy_density.size(), arma::fill::zeros);
			arma::Mat<element_type> _spectral_fun_typ_eps(omegax.size()-1, energy_density.size(), arma::fill::zeros);
			arma::Mat<element_type> _element_count_eps(omegax.size()-1, energy_density.size(), arma::fill::zeros);
			
			const double dw_log = std::log10(omegax[1]) - std::log10(omegax[0]);
        	const double w0_log = std::log10(omegax[0]);
			const double window_width = 0.05;
			const double bandwidth = E(E.size() - 1) - E(0);
		#pragma omp parallel for
			for(int ii = 0; ii < energy_density.size(); ii++)
			{
				const double eps = energy_density(ii);
				const double energyx = eps * bandwidth + E(0);
				for(int n = 0; n < E.size() - 1; n++)
				{
					for(int m = n+1; m < E.size() - 1; m++){
						if (abs((E(n) + E(m)) / 2. - energyx) < window_width / 2.){
							double wnm = E(m) - E(n);
							const auto idx = int( (std::log10(wnm) - w0_log) / dw_log);
							if(idx < omegax.size()-1 && idx >= 0){
								const double _b_ = std::abs(mat_elem(n, m));
								_spectral_fun_eps(idx, ii) += 2 * _b_ * _b_;
								_spectral_fun_typ_eps(idx, ii) += 2 * std::log(_b_ * _b_);
								_element_count_eps(idx, ii) += 2;
							} else if(idx < 0) {
								const double _b_ = std::abs(mat_elem(n, m));
								_spectral_fun_eps(0, ii) += 2 * _b_ * _b_;
								_spectral_fun_typ_eps(0, ii) += 2 * std::log(_b_ * _b_);
								_element_count_eps(0, ii) += 2;
							} else {
								const double _b_ = std::abs(mat_elem(n, m));
								_spectral_fun_eps(omegax.size()-2, ii) += 2 * _b_ * _b_;
								_spectral_fun_typ_eps(omegax.size()-2, ii) += 2 * std::log(_b_ * _b_);
								_element_count_eps(omegax.size()-2, ii) += 2;
							}
						}
					}	
				}
			}
			_spectral_fun_eps.save(arma::hdf5_name(dir_realis + info + ".hdf5", "j=" + std::to_string(site) + "/spectral_fun_eps",   arma::hdf5_opts::append));
			_spectral_fun_typ_eps.save(arma::hdf5_name(dir_realis + info + ".hdf5", "j=" + std::to_string(site) + "/spectral_fun_typ_eps",   arma::hdf5_opts::append));
			_element_count_eps.save(arma::hdf5_name(dir_realis + info + ".hdf5", "j=" + std::to_string(site) + "/element_count_eps",   arma::hdf5_opts::append));
			arma::vec _spectral_fun(omegax.size()-1, arma::fill::zeros);
			arma::vec _spectral_fun_typ(omegax.size()-1, arma::fill::zeros);
			arma::vec _element_count(omegax.size()-1, arma::fill::zeros);
			
			for(int n = 0; n < E.size() - 1; n++){
				for(int m = n+1; m < E.size() - 1; m++){
					double wnm = E(m) - E(n);
					const auto idx = int( (std::log10(wnm) - w0_log) / dw_log);
					// if(idx < omegax.size()-1 && idx >= 0){
					// 	const double _b_ = std::abs(mat_elem(n, m));
					// 	_spectral_fun(idx) += 2 * _b_ * _b_;
					// 	_spectral_fun_typ(idx) += 2 * std::log(_b_ * _b_);
					// 	_element_count(idx) += 2;
					// }
					if(idx < omegax.size()-1 && idx >= 0) {
						const double _b_ = std::abs(mat_elem(n, m));
						_spectral_fun(idx) += 2 * _b_ * _b_;
						_spectral_fun_typ(idx) += 2 * std::log(_b_ * _b_);
						_element_count(idx) += 2;
					} else if(idx < 0) {
						const double _b_ = std::abs(mat_elem(n, m));
						_spectral_fun(0) += 2 * _b_ * _b_;
						_spectral_fun_typ(0) += 2 * std::log(_b_ * _b_);
						_element_count(0) += 2;
					} else {
						const double _b_ = std::abs(mat_elem(n, m));
						_spectral_fun(omegax.size()-2) += 2 * _b_ * _b_;
						_spectral_fun_typ(omegax.size()-2) += 2 * std::log(_b_ * _b_);
						_element_count(omegax.size()-2) += 2;
					}
				}	
			}
			_spectral_fun.save(arma::hdf5_name(dir_realis + info + ".hdf5", "j=" + std::to_string(site) + "/spectral_fun",   arma::hdf5_opts::append));
			_spectral_fun_typ.save(arma::hdf5_name(dir_realis + info + ".hdf5", "j=" + std::to_string(site) + "/spectral_fun_typ",   arma::hdf5_opts::append));
			_element_count.save(arma::hdf5_name(dir_realis + info + ".hdf5", "j=" + std::to_string(site) + "/element_count",   arma::hdf5_opts::append));
			std::cout << " - - - - - - finished spectral function for site i = " << sites(i) << "in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
			// start = std::chrono::system_clock::now();
			// // start = std::chrono::system_clock::now();
			// // arma::Mat<element_type> mat_elem = V * Sz_ops[i] * V.t();
			// auto kernel_Sx = [Ll, site](u64 state){ 
			// 	auto [val, num] = operators::sigma_x<double>(state, Ll, site ); 
			// 	return std::make_pair(num, val); 
			// 	};
			// _operator = QOps::generic_operator<double>(this->L, std::move(kernel_Sx), 1.0);
			// op_mat = _operator.to_matrix(dim);
			// HSnorm = arma::trace(op_mat * op_mat) / double(dim);
			// op_mat = op_mat / std::sqrt(HSnorm);

			// mat_elem = V.t() * op * V;
			// _submat_ = mat_elem.submat(idx_min, idx_min, idx_max -1, idx_max - 1);

			// std::tie(_susc, _susc_r) = adiabatics::gauge_potential_save(mat_elem, E);
			// agp_norm_Sx_r.col(i) = _susc;
			// typ_susc_Sx_r.col(i) = _susc_r;
			// diag_mat_elem_Sx_r.col(i) = arma::diagvec(mat_elem);
			
    		// std::cout << " - - - - - - finished Sx matrix elements for site i = " << sites(i) << "in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		}
		// #ifndef MY_MAC

		arma::mat occupations(this->L, dim, arma::fill::zeros);
		for(int n = 0; n < dim; n++)
		{
			auto state = this->ptr_to_model->get_eigenState(n);
			arma::mat rho(this->L, this->L, arma::fill::zeros);
			for(u64 k_idx = 0; k_idx < dim; k_idx++)
			{
				u64 base_state = _hilbert_space(k_idx);
				for(int i = 0; i < this->L; i++)
				{
					auto [_spin, _] = operators::sigma_z<double>(base_state, this->L, i);
					if( _spin > 0){
						rho(i, i) += (state(k_idx)) * state(k_idx);
					}
					for(int j = i+1; j < this->L; j++)
					{
						auto [_spin2, _] = operators::sigma_z<double>(base_state, this->L, i);
						if(_spin * _spin2 < 0){
							auto [val1, cm] = operators::sigma_minus<double>(base_state, this->L, j);
							auto [val2, cpcm] = operators::sigma_plus<double>(cm, this->L, i);
							if(std::abs(val1 * val2) > 0)
							{
								auto idx = _hilbert_space.find(cpcm);
								auto _val_ = (state(idx)) * state(k_idx) * val1 * val2;
								rho(i, j) += _val_;
								rho(j, i) += _val_;
							}
						}
					}
				}	
			}
			arma::vec occ = arma::eig_sym(rho);
			occupations.col(n) = occ;
			printSeparated(std::cout, "\t", 20, true, n, arma::sum(occ));
		}
		{
			occupations.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "occupations",   arma::hdf5_opts::append));
			agp_norm_Sz_r.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "susc",   arma::hdf5_opts::append));
			// agp_norm_Sx_r.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "SUSC/Sx",   arma::hdf5_opts::append));

			typ_susc_Sz_r.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "susc_r",   arma::hdf5_opts::append));
			// typ_susc_Sx_r.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "SUSC_R/Sx",   arma::hdf5_opts::append));

			diag_mat_elem_Sz_r.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "diag_mat",   arma::hdf5_opts::append));
			// diag_mat_elem_Sx_r.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "DIAG_MAT/Sx",   arma::hdf5_opts::append));
		}
		// #endif
		
		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start_re) << " s - - - - - - " << std::endl; // simulation end
	}
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
					auto [val1, tmp22] = operators::sigma_z<double>(state, Ll, Ll - 1 );
					return std::make_pair(state, val1);
					};
		auto _operator = QOps::generic_operator<double>(this->L, std::move(kernel_def), 1.0);
		arma::sp_mat oper = _operator.to_matrix(dim);
		
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
	U1Hilbert _hilbert_space = this->ptr_to_model->get_model_ref().get_hilbert_space();

	std::string info = this->set_info();
	
	const u64 dim_max = 1e3;
	const size_t size = dim > dim_max? this->l_steps : dim;

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
		if(dim > dim_max){
			double error = this->ptr_to_model->diag_sparse(this->l_steps, this->l_bundle, this->tol, this->seed);
			if( error > 1e-10 ) { std::cout << "POLFED FAILED: Maximal Error = " << error << std::endl; continue; }
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

		// arma::Mat<element_type> mat_elem = V * Sz_ops[i] * V.t();
		
		auto sites = std::vector<int>( {Ll / 2, Ll - 1} );
		for(int il = 0; il < sites.size(); il++){
			start = std::chrono::system_clock::now();
			int ell = sites[il];
			auto kernel = [Ll, N, ell](u64 state){ 
				auto [val1, tmp22] = operators::sigma_z<double>(state, Ll, ell );
				return std::make_pair(state, val1);
				};
			auto _operator = QOps::generic_operator<double>(this->L, std::move(kernel), 1.0);
			
			// KEEP PROPER HILBERT SPACE FOR OPERATORS
			arma::sp_mat opmat = _operator.to_reduced_matrix(_hilbert_space);
			
			arma::Mat<element_type> mat_elem = V.t() * opmat * V;
			std::cout << " - - - - - - finished matrix elements in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
			start = std::chrono::system_clock::now();
			double cutoff = std::sqrt(Ll) / double(dim);
			auto [_susc, _susc_r] = adiabatics::gauge_potential_save(mat_elem, E, this->L, cutoff);

			std::cout << " - - - - - - finished AGP in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
			start = std::chrono::system_clock::now();

			arma::Mat<element_type> _integrated_spectral_fun(omegax.size()-1, energy_density.size(), arma::fill::zeros);
			arma::Mat<element_type> _spectral_fun(omegax.size()-1, energy_density.size(), arma::fill::zeros);
			arma::Mat<element_type> _spectral_fun_typ(omegax.size()-1, energy_density.size(), arma::fill::zeros);
			arma::Mat<element_type> _element_count(omegax.size()-1, energy_density.size(), arma::fill::zeros);
			
			double bandwidth, E0;
			if(dim > dim_max){
				auto Hamil = this->ptr_to_model->get_hamiltonian();
				auto lancz = lanczos::Lanczos<element_type, converge::energies>(Hamil, 1, 10000, 1e-15, this->seed, 1);
				lancz.diagonalization();
				arma::vec ener = lancz.get_eigenvalues();
				E0 = ener(0);
				bandwidth = ener(ener.size()-1) - E0;
			} else{
				E0 = E(0);
				bandwidth = E(E.size() - 1) - E(0);	
			}
			std::cout << " - - - - - - dE = " << bandwidth << std::endl;
			for(int ii = 0; ii < energy_density.size(); ii++){
				const double eps = energy_density(ii);
				const double energyx = eps * bandwidth + E0;
				std::cout << " - - - - - - E = " << energyx << std::endl;
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
					if(indices.size() > 1){
						arma::vec x = arma::vec( omegas_i.elem(indices) );
						arma::vec y = arma::vec( matter.elem(indices) );
						_integrated_spectral_fun(k, ii) = arma::accu(y);
					}
				}
			}
			std::cout << " - - - - - - finished Sz_ell = " << ell << " matrix elements in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
			// #ifndef MY_MAC
			{
				omegax.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "omegas_l=" + std::to_string(ell),   arma::hdf5_opts::append));
				_integrated_spectral_fun.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "integrated_spectral_fun_l=" + std::to_string(ell),   arma::hdf5_opts::append));
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