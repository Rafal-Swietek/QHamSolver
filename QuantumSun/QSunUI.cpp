#include "includes/QSunUI.hpp"

int outer_threads = 1;
int num_of_threads = 1;

bool normalize_grain = 1;

namespace QSunUI{

void ui::make_sim(){
    printAllOptions();
	
	clk::time_point start = std::chrono::system_clock::now();
	this->seed = std::random_device{}();
	this->ptr_to_model = this->create_new_model_pointer();
	size_t dim = this->ptr_to_model->get_hilbert_size();
	// // hybrydization();
	// // return;

	// arma::vec alfas = arma::linspace(0.6, 1.5, 26);
	// arma::mat H_trace(this->realisations, alfas.size(), arma::fill::zeros);
	// arma::mat H_trace2(this->realisations, alfas.size(), arma::fill::zeros);
	// arma::mat av(this->realisations, alfas.size(), arma::fill::zeros);
	// arma::mat var(this->realisations, alfas.size(), arma::fill::zeros);
	// arma::mat av2(this->realisations, alfas.size(), arma::fill::zeros);
	// arma::mat var2(this->realisations, alfas.size(), arma::fill::zeros);
	// arma::mat av3(this->realisations, alfas.size(), arma::fill::zeros);
	// arma::mat var3(this->realisations, alfas.size(), arma::fill::zeros);
	// for(int r = 0; r < this->realisations; r++)
	// {
	// 	for(int iig = 0; iig < alfas.size(); iig++)
	// 	{
	// 		this->seed = std::random_device{}();
	// 		this->alfa = alfas(iig);
	// 		this->reset_model_pointer();
	// 		arma::sp_mat H = this->ptr_to_model->get_hamiltonian();
	// 		arma::sp_mat H2 = H*H;
	// 		double meanH = arma::trace(H) / double(dim);
	// 		double varH = arma::trace(H2) / double(dim) - meanH * meanH;
	// 		H_trace(r, iig) = meanH;
	// 		H_trace2(r, iig) = varH;
			
	// 		arma::vec gec(dim, arma::fill::zeros);
	// 	#pragma omp parallel for
	// 		for(u64 k = 0; k < dim; k++)
	// 			gec(k) = 2 * H2(k,k) - H(k,k) * H(k,k);   
	// 		av2(r, iig) = arma::mean(gec);
	// 		var2(r, iig) = arma::var(gec);
			
	// 		H = H - meanH * arma::eye<arma::sp_mat>(dim, dim);
	// 		H2 = H*H;

	// 		gec = arma::vec(dim, arma::fill::zeros);
	// 	#pragma omp parallel for
	// 		for(u64 k = 0; k < dim; k++)
	// 			gec(k) = 2 * H2(k,k) - H(k,k) * H(k,k);   
	// 		av3(r, iig) = arma::mean(gec);
	// 		var3(r, iig) = arma::var(gec);

	// 		gec = arma::vec(dim, arma::fill::zeros);
	// 	#pragma omp parallel for
	// 		for(u64 k = 0; k < dim; k++)
	// 			gec(k) = 2 * H2(k,k) - H(k,k) * H(k,k); 
	// 		gec = gec / varH;
	// 		av(r, iig) = arma::mean(gec);
	// 		var(r, iig) = arma::var(gec);
	// 	}
	// 	std::cout << " - - - - - - finished realization r=" << r << "\t in :" << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
	// }
	// // av /= double(this->realisations);
	// // av2 /= double(this->realisations);
	// // var /= double(this->realisations);
	// // var2 /= double(this->realisations);
	// // H_trace /= double(this->realisations);
	// // H_trace2 /= double(this->realisations);

	// // av2 = av2 / H_trace2;
	// // var2 = var2 / arma::square(H_trace2);
	// // for(int iig = 0; iig < alfas.size(); iig++)
	// // {
	// // 	printSeparated(std::cout, "\t\t", 20, true, "--", std::abs(av(iig) - mean_GEC(this->L_loc, alfas(iig))), "--", std::abs(av2(iig) - mean_GEC(this->L_loc, alfas(iig))), "--------", std::abs(var(iig) - var_GEC(this->L_loc, alfas(iig))), "--", std::abs(var2(iig) - var_GEC(this->L_loc, alfas(iig))));
	// // }
	// std::string dir = "GEC_data2" + kPSep;
	// createDirs(dir);
	// std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid) + kPSep;
	// createDirs(dir_realis);
	// alfas.save(	  arma::hdf5_name(dir_realis + "GEC_L=" + std::to_string(this->L) + ".hdf5", "alfas"));
	// av.save(	  arma::hdf5_name(dir_realis + "GEC_L=" + std::to_string(this->L) + ".hdf5", "av", arma::hdf5_opts::append));
	// var.save(	  arma::hdf5_name(dir_realis + "GEC_L=" + std::to_string(this->L) + ".hdf5", "var", arma::hdf5_opts::append));
	// av2.save(	  arma::hdf5_name(dir_realis + "GEC_L=" + std::to_string(this->L) + ".hdf5", "av2", arma::hdf5_opts::append));
	// var2.save(	  arma::hdf5_name(dir_realis + "GEC_L=" + std::to_string(this->L) + ".hdf5", "var2", arma::hdf5_opts::append));
	// av3.save(	  arma::hdf5_name(dir_realis + "GEC_L=" + std::to_string(this->L) + ".hdf5", "av3", arma::hdf5_opts::append));
	// var3.save(	  arma::hdf5_name(dir_realis + "GEC_L=" + std::to_string(this->L) + ".hdf5", "var3", arma::hdf5_opts::append));
	// H_trace.save(	  arma::hdf5_name(dir_realis + "GEC_L=" + std::to_string(this->L) + ".hdf5", "trace_H", arma::hdf5_opts::append));
	// H_trace2.save(	  arma::hdf5_name(dir_realis + "GEC_L=" + std::to_string(this->L) + ".hdf5", "trace_H2", arma::hdf5_opts::append));

	// std::cout << " - - - - - - FINISHED CALCULATIONS IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
	// return;
	
	// arma::Mat<element_type> H = this->ptr_to_model->get_dense_hamiltonian();
	// H.save(   arma::hdf5_name("HamiltonianQSM.hdf5", "H"));

	// auto Translate = QOps::__builtins::translation(this->L, 1);
	// auto flip = QOps::__builtins::spin_flip_x(this->L);
	// auto some_kernel = [&Translate, &flip](u64 n){
	// 	n = std::get<0>(flip(n));
	// 	return !( (n) & std::get<0>( Translate(n) ) );
	// };
	// auto _hilbert = QHS::constrained_hilbert_space(this->L, std::move(some_kernel));
	// auto my_map = _hilbert.get_mapping();
	// for(auto& item : my_map){
	// 	auto vec = boost::dynamic_bitset<>(this->L, item);
	// 	printSeparated(std::cout, "\t", 16, true, item, vec);
	// }
	// return;

	// auto Hamil = this->ptr_to_model->get_hamiltonian();
	// this->l_steps = 0.1 * Hamil.n_cols;
	// if(this->l_steps > 500)
	// 	this->l_steps = 500;
	// auto polfed = polfed::POLFED<ui::element_type>(Hamil, this->l_steps, this->l_bundle, -1, this->tol, 0.25, this->seed, true);
	// auto [E, V] = polfed.eig();
	// return;


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
	case 8:
		quench();
		break;
	case 9:
		agp();
		break;
	case 10:
		agp_save();
		break;
	case 11:
		agp_mu();
		break;
	case 12:
		spectral_function();
		break;
	case 13:
		ground_state();
		break;
	case 14:
		geometric_tensor();
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
								
								// this->reset_model_pointer();
								const auto start_loop = std::chrono::system_clock::now();
								std::cout << " - - START NEW ITERATION:\t\t par = "; // simulation end
								printSeparated(std::cout, "\t", 16, true, this->L_loc, this->J, this->alfa, this->h, this->w, this->gamma);
								this->reset_model_pointer();
								
								quench_fourier(); continue;

								multifractality(); continue;

								geometric_tensor(); continue;

								ground_state(); continue;
								agp(); continue;

	const int Ll = this->L;

	auto disorder_generator = disorder<double>(this->seed);
	this->ptr_to_model.reset(new QHS::QHamSolver<QuantumSun>(this->L_loc, this->J, this->alfa, this->gamma, 0, 0, 
																	this->seed, this->grain_size, this->zeta, this->initiate_avalanche, normalize_grain)); 
	
	u64 dim = this->ptr_to_model->get_hilbert_size();

	auto kernel = [Ll](u64 state){ 
			auto [val1, tmp22] = operators::sigma_z<cpx>(state, Ll, Ll - 1 );
			return std::make_pair(state, val1);
			};
	auto _operator = QOps::genOp(this->L, std::move(kernel), 1.0);
	arma::sp_mat Sz = arma::real(_operator.to_matrix(dim));
	
	arma::mat H = this->ptr_to_model->get_dense_hamiltonian();
	arma::vec disorder_base = disorder_generator.uniform(this->L_loc, this->h - this->w, this->h + this->w);
	for (u64 k = 0; k < dim; k++) {
		for (int j = this->grain_size; j < this->L; j++)  // sum over spin d.o.f
		{
			const int pos_in_array = j - this->grain_size;                // array index of localised spin
			auto [val, Sz_k] = operators::sigma_z<cpx>(k, this->L, j);
			H(k, k) += disorder_base(pos_in_array) * std::real(val);
		}
	}
	arma::vec E; arma::mat V;
	arma::eig_sym(E, V, H);
	arma::vec diag_mat_elem = arma::diagvec(V.t() * Sz * V);
	
	auto name = "L=" + std::to_string(this->L_loc) + ".hdf5";
	disorder_base.save(arma::hdf5_name(name, "disorder"));
	E.save(	  			  arma::hdf5_name(name, "E0",   arma::hdf5_opts::append));
	diag_mat_elem.save(	  arma::hdf5_name(name, "Sz0",   arma::hdf5_opts::append));

	for(int j : arma::ivec({this->grain_size, this->L / 2, this->L - 1})){
		auto disorder = disorder_base;
		disorder(j - this->grain_size) = -disorder(j - this->grain_size);
		
		arma::mat H = this->ptr_to_model->get_dense_hamiltonian();
		for (u64 k = 0; k < dim; k++) {
			u64 base_state = k;
			for (int ell = this->grain_size; ell < this->L; ell++)  // sum over spin d.o.f
			{
				const int pos_in_array = ell - this->grain_size;                // array index of localised spin
				auto [val, Sz_k] = operators::sigma_z<cpx>(base_state, this->L, ell);
			    H(k, k) += disorder(pos_in_array) * real(val);
			}
		}
		arma::eig_sym(E, V, H);
		arma::vec diag_mat_elem = arma::diagvec(V.t() * Sz * V);
		E.save(	  			  arma::hdf5_name(name, "E_j=" + std::to_string(j),   arma::hdf5_opts::append));
		diag_mat_elem.save(	  arma::hdf5_name(name, "Sz_j=" + std::to_string(j),   arma::hdf5_opts::append));
	}

	continue;
								quench(); continue;


								// auto Hamil = this->ptr_to_model->get_hamiltonian();
								// this->l_steps = 0.1 * Hamil.n_cols;
								// if(this->l_steps > 500)
								// 	this->l_steps = 500;
								// auto polfed = polfed::POLFED<ui::element_type>(Hamil, this->l_steps, this->l_bundle, -1, this->tol, 0.2, this->seed, true);
								// auto [E, V] = polfed.eig();
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

void ui::hybrydization(){
	clk::time_point start = std::chrono::system_clock::now();

	// std::string subdir = "ParticipationRatio" + kPSep;
	std::string dir = this->saving_dir + "Hybridization" + kPSep;
	createDirs(dir);

	std::string info = this->set_info();
	std::string filename = info;
	size_t dim = this->ptr_to_model->get_hilbert_size();

	int counter = 0;

	const int size = dim;	
	arma::vec omegax = arma::logspace(int(std::log10(0.1/dim)), int(std::log10( 5 + this->L )), 20 * this->L);

	disorder<double> disorder_generator = disorder<double>(this->seed);
	disorder<int> neigh_generator = disorder<int>(this->seed);
	GOE grain_generator(this->seed);
	for(int realis = 0; realis < this->realisations; realis++)
	{
		if(realis > 0)
			this->ptr_to_model->generate_hamiltonian();
		
    	clk::time_point start_loop = std::chrono::system_clock::now();
		// this->ptr_to_model->diagonalization();
		const size_t dim_loc = ULLPOW( (this->L_loc) );
		const size_t dim_erg = ULLPOW( (this->grain_size) );

		arma::mat H = arma::mat(dim, dim, arma::fill::zeros);
		arma::sp_mat SxSxfull(dim, dim);
		arma::sp_mat SxSx(dim, dim);
		auto _disorder = disorder_generator.uniform(this->L_loc, this->h - this->w, this->h + this->w);
		std::cout << "AAAAA: " << _disorder.t() << std::endl;
		
		/* Create random neighbours for coupling hamiltonian */
		auto random_neigh = neigh_generator.uniform(this->L_loc, 0, this->grain_size - 1);

		/* Create GOE Matrix */
		arma::mat H_grain = this->gamma * grain_generator.generate_matrix(dim_erg);
		// H_grain = H_grain - arma::trace(H_grain);
		// H_grain /= std::sqrt(ULLPOW(this->grain_size) + 1);
		H_grain /= std::sqrt( arma::trace(H_grain * H_grain) / double(dim_erg) );

		/* Create random couplings */
		auto _long_range_couplings = arma::vec(this->L_loc, arma::fill::zeros);
		if(this->alfa > 0){
			
			if( std::abs(this->alfa - 1) < 1e-10){
				_long_range_couplings = arma::vec(this->L_loc, arma::fill::ones);
			} else {
				double u_j = 1 + disorder_generator.uniform_dist<double>(-this->zeta, this->zeta);
				_long_range_couplings(0) = 1.0;
				for (int j = 1; j < this->L_loc; j++){
					int pos = j;
					double u_j = pos + disorder_generator.uniform_dist<double>(-this->zeta, this->zeta);
					_long_range_couplings(j) = std::pow(this->alfa, u_j);
				}
			}
		}
		_extra_debug(
			std::cout << "disorder: \t\t" << _disorder.t() << std::endl;   
			std::cout << "couplings: \t\t" << _long_range_couplings.t() << std::endl;
			std::cout << "random_neigh: \t\t" << random_neigh.t() << std::endl;
			std::cout << "Grain matrix: \t\t" << H_grain << std::endl;
		)

		/* Generate coupling and spin hamiltonian */
		clk::time_point start = std::chrono::system_clock::now();
		for (u64 k = 0; k < dim; k++) {
			u64 base_state = k;
			for (int j = 0; j < this->L - this->grain_size; j++)  // sum over spin d.o.f
			{
				const int pos_in_array = this->L - 1 - this->grain_size - j;                // array index of localised spin
				/* disorder on localised spins */
				auto [val, Sz_k] = operators::sigma_z<double>(base_state, this->L, j);
				H(k, k) += _disorder(pos_in_array) * (val);
				// H0(k, k) += _disorder(pos_in_array) * (val);
			
				/* coupling of localised spins to GOE grain */
				int nei = random_neigh(pos_in_array);
				auto [val1, state_Sx_k] = operators::sigma_x<double>(base_state, this->L, j);
				auto [val2, state_SxSx_k] = operators::sigma_x<double>(state_Sx_k, this->L, this->L - this->grain_size + nei);
				double mat_element = this->J * _long_range_couplings(pos_in_array) * (val1 * val2);
				H(state_SxSx_k, k) += mat_element;
				
				SxSxfull(state_SxSx_k, k) += mat_element;
				if(j == 0)
					SxSx(state_SxSx_k, k) += mat_element;
			}
		}
		std::cout << " - - - - - - finished Hamiltonian in : " << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		// H = H + arma::kron<arma::mat>(arma::mat(H_grain), arma::mat(arma::eye<arma::mat>(dim_loc, dim_loc)));
		// H0 = H0 + arma::kron<arma::mat>(arma::mat(H_grain), arma::mat(arma::eye<arma::mat>(dim_loc, dim_loc)));
		H = H + arma::kron<arma::mat>(arma::mat(arma::eye<arma::mat>(dim_loc, dim_loc)), arma::mat(H_grain));

		arma::vec E;
		arma::mat V;
		arma::eig_sym(E, V, H);
		
		double E_av = arma::mean(E);
		std::cout << " - - - - - - finished diagonalization in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();
		
		auto i = min_element(begin(E), end(E), [=](double x, double y) {
				return abs(x - E_av) < abs(y - E_av);
				});
		u64 E_av_idx = i - E.begin();

		u64 num_of_states = std::min( u64(this->l_steps), u64(0.02*dim) );
		u64	Emin = E_av_idx - num_of_states / 2;
		u64	Emax = E_av_idx + num_of_states / 2;

		std::string dir_realis = dir + "realisation=" + std::to_string(realis + this->jobid) + kPSep;
		createDirs(dir_realis);
		E.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "energies"));
		omegax.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "omegax", arma::hdf5_opts::append));
		std::cout << " - - - - - - finished diagonalization of L-1 sized matrix in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
		
		start = std::chrono::system_clock::now();
		arma::Mat<element_type> mat_elem = V.t() * SxSx * V;
		std::cout << " - - - - - - finished SxSx matrix elements in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		arma::vec hybdrydization_SxSx(size-1, arma::fill::zeros);
				
		for(int k = 0; k < size-1; k++){
			hybdrydization_SxSx(k) = mat_elem(k+1, k) / (E(k+1) - E(k));
		}
		arma::vec _spectral_fun(omegax.size()-1, arma::fill::zeros);
		arma::vec _spectral_fun_typ(omegax.size()-1, arma::fill::zeros);
		arma::vec _element_count(omegax.size()-1, arma::fill::zeros);
		
		const double bandwidth = E(E.size() - 1) - E(0);	
		spectrals::preset_omega set_omega(E, 1000, 0);
		auto [omegas_i, matter] = set_omega.get_matrix_elements(mat_elem);

		for(int k = 0; k < omegax.size() - 1; k++){
			arma::uvec indices = arma::find(omegas_i >= omegax[k] && omegas_i < omegax[k+1]);
			_element_count(k) = indices.size();
			_spectral_fun(k) = arma::accu( matter.rows(indices));
			_spectral_fun_typ(k) = arma::accu( arma::log(matter.rows(indices)) );
		}
		omegas_i.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "omegas_mel", arma::hdf5_opts::append));
		matter.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "mat_elem", arma::hdf5_opts::append));
		hybdrydization_SxSx.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "hybrid_SxSx", arma::hdf5_opts::append));
		_spectral_fun.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "_spectral_fun", arma::hdf5_opts::append));
		_element_count.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "_element_count", arma::hdf5_opts::append));
		_spectral_fun_typ.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "_spectral_fun_typ", arma::hdf5_opts::append));
		
		std::cout << " - - - - - - finished SxSx matrix elements in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		
		start = std::chrono::system_clock::now();
		mat_elem = V.t() * SxSxfull * V;
		std::cout << " - - - - - - finished SxSxfull matrix elements in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		
		arma::vec hybdrydization_SxSxfull(size, arma::fill::zeros);
		for(int k = 0; k < size-1; k++){
			hybdrydization_SxSxfull(k) = mat_elem(k+1, k) / (E(k+1) - E(k));
		}

		arma::vec _spectral_fun_sum(omegax.size()-1, arma::fill::zeros);
		arma::vec _spectral_fun_typ_sum(omegax.size()-1, arma::fill::zeros);
		arma::vec _element_count_sum(omegax.size()-1, arma::fill::zeros);
		
		auto [omegas_i2, matter2] = set_omega.get_matrix_elements(mat_elem);

		for(int k = 0; k < omegax.size() - 1; k++){
			arma::uvec indices = arma::find(omegas_i2 >= omegax[k] && omegas_i2 < omegax[k+1]);
			_element_count_sum(k) = indices.size();
			_spectral_fun_sum(k) = arma::accu( matter2.rows(indices));
			_spectral_fun_typ_sum(k) = arma::accu( arma::log(matter2.rows(indices)) );
		}
		
		omegas_i2.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "omegas_mel_sum", arma::hdf5_opts::append));
		matter2.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "mat_elem_sum", arma::hdf5_opts::append));
		hybdrydization_SxSxfull.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "hybrid_SxSxfull", arma::hdf5_opts::append));
		_spectral_fun_sum.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "_spectral_fun_sum", arma::hdf5_opts::append));
		_element_count_sum.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "_element_count_sum", arma::hdf5_opts::append));
		_spectral_fun_typ_sum.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "_spectral_fun_typ_sum", arma::hdf5_opts::append));
		// omegax

		std::cout << " - - - - - - finished SxSx_full matrix elements in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start_loop) << " s - - - - - - " << std::endl; // simulation end
	};
    std::cout << " - - - - - - FINISHED HYBDRIDIZATION CALCULATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
}
/// @brief Calculate ground state properties
void ui::ground_state(){
	std::string dir = this->saving_dir + "GroundState" + kPSep + "TESTS" + kPSep;
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info();

	u64 dim_cut = 1000;
	const size_t size = this->l_steps;

	arma::vec energies(size, arma::fill::zeros);

	int Ll = this->L;
	int N = this->grain_size;

	int counter = 0;
	
	auto subsystem_sizes = arma::conv_to<arma::Col<int>>::from(arma::linspace(0, this->L, this->L + 1));
	arma::vec q_ipr_list = {0.5, 1.0, 1.5, 2, 3.0};
	
	std::vector<QOps::genOp> permutation_op;
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
		auto permutation = QOps::_permutation_generator<cpx>(this->L, p);
		permutation_op.push_back(permutation);

		std::cout << " - - - - - - set permutation matrix for LA = " << LA << " in : " << tim_s(start_LA) << " s - - - - - - " << std::endl;
	}
	std::vector<QOps::genOp> Sx_list;
	std::vector<QOps::genOp> Sy_list;
	std::vector<QOps::genOp> Sz_list;
	std::vector<std::vector<QOps::genOp>> SxSx_list;
	std::vector<std::vector<QOps::genOp>> SySy_list;
	std::vector<std::vector<QOps::genOp>> SzSz_list;
	for(int i = 0; i < this->L; i++)
	{
		auto kernel_Sx = [Ll, N, i](u64 state){ 
					auto [val1, state_X] = operators::sigma_x<cpx>(state, Ll, i );
					return std::make_pair(state_X, val1);
					};
		QOps::genOp _operator = QOps::genOp(this->L, std::move(kernel_Sx), 1.0);
		Sx_list.push_back(_operator);

		auto kernel_Sy = [Ll, N, i](u64 state){ 
					auto [val1, state_Y] = operators::sigma_y(state, Ll, i );
					return std::make_pair(state_Y, val1);
					};
		_operator = QOps::genOp(this->L, std::move(kernel_Sy), 1.0);
		Sy_list.push_back(_operator);

		auto kernel_Sz = [Ll, N, i](u64 state){ 
					auto [val1, state_Z] = operators::sigma_z<cpx>(state, Ll, i );
					return std::make_pair(state_Z, val1);
					};
		_operator = QOps::genOp(this->L, std::move(kernel_Sz), 1.0);
		Sz_list.push_back(_operator);

		std::vector<QOps::genOp> Sxx_list_temp;
		std::vector<QOps::genOp> Syy_list_temp;
		std::vector<QOps::genOp> Szz_list_temp;
		for(int j = 0; j < this->L; j++)
		{
			auto kernel_SxSx = [Ll, N, i, j](u64 state){ 
					auto [val1, state_X] = operators::sigma_x<cpx>(state, Ll, i );
					auto [val2, state_XX] = operators::sigma_x<cpx>(state_X, Ll, j );
					return std::make_pair(state_XX, val1 * val2);
					};
			_operator = QOps::genOp(this->L, std::move(kernel_SxSx), 1.0);
			Sxx_list_temp.push_back(_operator);
			auto kernel_SySy = [Ll, N, i, j](u64 state){ 
					auto [val1, state_Y] = operators::sigma_y(state, Ll, i );
					auto [val2, state_YY] = operators::sigma_y(state_Y, Ll, j );
					return std::make_pair(state_YY, val1 * val2);
					};
			_operator = QOps::genOp(this->L, std::move(kernel_SySy), 1.0);
			Syy_list_temp.push_back(_operator);
			auto kernel_SzSz = [Ll, N, i, j](u64 state){ 
					auto [val1, state_Z] = operators::sigma_z<cpx>(state, Ll, i );
					auto [val2, state_ZZ] = operators::sigma_z<cpx>(state_Z, Ll, j );
					return std::make_pair(state_ZZ, val1 * val2);
					};
			_operator = QOps::genOp(this->L, std::move(kernel_SzSz), 1.0);
			Szz_list_temp.push_back(_operator);
		}
		SxSx_list.push_back(Sxx_list_temp);
		SySy_list.push_back(Syy_list_temp);
		SzSz_list.push_back(Szz_list_temp);
	}
	for(int realis = 0; realis < this->realisations; realis++)
	{
		clk::time_point start_re = std::chrono::system_clock::now();
		if(realis > 0)
			this->ptr_to_model->generate_hamiltonian();
		clk::time_point start = std::chrono::system_clock::now();
		if(dim > dim_cut){
			double error = this->ptr_to_model->diag_lanczos(this->l_steps, this->tol, this->seed);
			if( error > 1e-10 ) { std::cout << "POLFED FAILED: Maximal Error = " << error << std::endl; }
		}
		else{
			this->ptr_to_model->diagonalization();
		}

		const arma::vec E = this->ptr_to_model->get_eigenvalues().rows(0, 1);

		const arma::mat _grain = this->ptr_to_model->get_model_ref().get_grain();
		const auto _neighs = this->ptr_to_model->get_model_ref().get_neighs();
		const arma::vec _disorder = this->ptr_to_model->get_model_ref().get_disorder();
		const arma::vec _interaction = this->ptr_to_model->get_model_ref().get_interaction();
		std::cout << " - - - - - - finished diagonalization in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();

		arma::vec Pq_GS(q_ipr_list.size(), arma::fill::zeros);
		arma::vec Sq_GS(q_ipr_list.size(), arma::fill::zeros);
		arma::vec Pq_EX(q_ipr_list.size(), arma::fill::zeros);
		arma::vec Sq_EX(q_ipr_list.size(), arma::fill::zeros);
		arma::vec S_GS(subsystem_sizes.size(), arma::fill::zeros);
		arma::vec S_EX = S_GS;
		arma::vec S_site_GS = S_GS;
		arma::vec S_site_EX = S_GS;

		outer_threads = this->thread_number;
		omp_set_num_threads(1);
		std::cout << outer_threads << "\t\t" << omp_get_num_threads() << std::endl;
		
		
		arma::Col<element_type> state_GS = arma::normalise(this->ptr_to_model->get_eigenState(0));
		arma::Col<element_type> state_excited = arma::normalise(this->ptr_to_model->get_eigenState(1));
	
	#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
		for(int LA_idx = 0; LA_idx < subsystem_sizes.size() - 1; LA_idx++)
		{	
			auto start_LA = std::chrono::system_clock::now();
			int LA = subsystem_sizes[LA_idx];
			S_GS(LA_idx) = entropy::schmidt_decomposition(state_GS, this->L - LA, this->L);	// bipartite entanglement at subsystem size LA
			S_EX(LA_idx) = entropy::schmidt_decomposition(state_excited, this->L - LA, this->L);	// bipartite entanglement at subsystem size LA
			
			arma::vec permuted_state = arma::real(permutation_op[LA_idx].multiply(state_GS));
			S_site_GS(LA_idx) = entropy::schmidt_decomposition(permuted_state, this->L - 1, this->L);	// single site entanglement at site LA
			
			permuted_state = arma::real(permutation_op[LA_idx].multiply(state_excited));
			S_site_EX(LA_idx) = entropy::schmidt_decomposition(permuted_state, this->L - 1, this->L);	// single site entanglement at site LA
			
			std::cout << " - - - - - - Finished Entropies for LA = " << LA << " in : " << tim_s(start_LA) << " s - - - - - - " << std::endl;
		}
		std::cout << " - - - - - - finished all entropies in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();
	
	#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
		for(int iq = 0; iq < q_ipr_list.size(); iq++)
		{
			if(q_ipr_list(iq) == 1)
			{
				double _pr_GS = 0, _pr_EX = 0;
			// #pragma omp parallel for reduction(+: _pr_)
				for (int n = 0; n < state_GS.size(); n++) {
					double value = std::abs(std::conj(state_GS(n)) * state_GS(n));
					_pr_GS += (std::abs(value) > 0) ? -value * std::log(value) : 0;

					value = std::abs(std::conj(state_excited(n)) * state_excited(n));
					_pr_EX += (std::abs(value) > 0) ? -value * std::log(value) : 0;
				}
				Pq_GS(iq) = arma::norm(state_GS);
				Sq_GS(iq) = _pr_GS;

				Pq_EX(iq) = arma::norm(state_excited);
				Sq_EX(iq) = _pr_EX;
			} else {
				double _pr_GS = 0, _pr_EX = 0;
				for (int n = 0; n < N; n++) {
					double value = std::abs(std::conj(state_GS(n)) * state_GS(n));
					_pr_GS += std::pow(value, q_ipr_list(iq));
					
					value = std::abs(std::conj(state_excited(n)) * state_excited(n));
					_pr_EX += std::pow(value, q_ipr_list(iq));
				}
				Pq_GS(iq) = _pr_GS;
				Sq_GS(iq) = -std::log(_pr_GS) / (1 - q_ipr_list(iq));

				Pq_EX(iq) = _pr_EX;
				Sq_EX(iq) = -std::log(_pr_EX) / (1 - q_ipr_list(iq));
			}
		}
		std::cout << " - - - - - - finished all participation_ratios in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();
		
		arma::vec Sx_GS(this->L, arma::fill::zeros);
		arma::cx_vec Sy_GS(this->L, arma::fill::zeros);
		arma::vec Sz_GS(this->L, arma::fill::zeros);

		arma::mat SxSx_GS(this->L, this->L, arma::fill::value(0.25));
		arma::mat SySy_GS(this->L, this->L, arma::fill::value(0.25));
		arma::mat SzSz_GS(this->L, this->L, arma::fill::value(0.25));
		
		arma::vec Sx_EX(this->L, arma::fill::zeros);
		arma::cx_vec Sy_EX(this->L, arma::fill::zeros);
		arma::vec Sz_EX(this->L, arma::fill::zeros);

		arma::mat SxSx_EX(this->L, this->L, arma::fill::value(0.25));
		arma::mat SySy_EX(this->L, this->L, arma::fill::value(0.25));
		arma::mat SzSz_EX(this->L, this->L, arma::fill::value(0.25));
		for(int i = 0; i < this->L; i++)
		{	
			//<! GROUND STATE
			arma::cx_vec new_state = Sx_list[i].multiply(state_GS);
			Sx_GS(i) = std::real( dot_prod(state_GS, new_state) );
			new_state = Sy_list[i].multiply(state_GS);
			Sy_GS(i) = dot_prod(state_GS, new_state);
			new_state = Sz_list[i].multiply(state_GS);
			Sz_GS(i) = std::real( dot_prod(state_GS, new_state) );
			
			//<! EXCITED STATE
			new_state = Sx_list[i].multiply(state_excited);
			Sx_EX(i) = std::real( dot_prod(state_excited, new_state) );
			new_state = Sy_list[i].multiply(state_excited);
			Sy_EX(i) = dot_prod(state_excited, new_state);
			new_state = Sz_list[i].multiply(state_excited);
			Sz_EX(i) = std::real( dot_prod(state_excited, new_state) );

			for(int j = i + 1; j < this->L; j++)
			{
				auto start_ij = std::chrono::system_clock::now();
				
				//<! GROUND STATE
				new_state = SxSx_list[i][j].multiply(state_GS);
				SxSx_GS(i, j) = std::real( dot_prod(state_GS, new_state) );	SxSx_GS(j, i) = SxSx_GS(i, j);
				new_state = SySy_list[i][j].multiply(state_GS);
				SySy_GS(i, j) = std::real( dot_prod(state_GS, new_state) );	SySy_GS(j, i) = SySy_GS(i, j);
				new_state = SzSz_list[i][j].multiply(state_GS);
				SzSz_GS(i, j) = std::real( dot_prod(state_GS, new_state) );	SzSz_GS(j, i) = SzSz_GS(i, j);
				
				//<! EXCITED STATE
				new_state = SxSx_list[i][j].multiply(state_excited);
				SxSx_EX(i, j) = std::real( dot_prod(state_excited, new_state) );	SxSx_EX(j, i) = SxSx_EX(i, j);
				new_state = SySy_list[i][j].multiply(state_excited);
				SySy_EX(i, j) = std::real( dot_prod(state_excited, new_state) );	SySy_EX(j, i) = SySy_EX(i, j);
				new_state = SzSz_list[i][j].multiply(state_excited);
				SzSz_EX(i, j) = std::real( dot_prod(state_excited, new_state) );	SzSz_EX(j, i) = SzSz_EX(i, j);

				std::cout << " - - - - - - Finished correlator for i=" << i << " and j=" << j << "\tin : " << tim_s(start_ij) << " s - - - - - - " << std::endl;
			}
		}
		std::cout << " - - - - - - finished correlation matrices in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
		{
			std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
			createDirs(dir_realis);
			E.save(arma::hdf5_name(dir_realis + info + ".hdf5", "energies"));

			_grain.save(arma::hdf5_name(dir_realis + info + ".hdf5", "MODEL/grain", arma::hdf5_opts::append));
			_neighs.save(arma::hdf5_name(dir_realis + info + ".hdf5", "MODEL/neighbours", arma::hdf5_opts::append));
			_disorder.save(arma::hdf5_name(dir_realis + info + ".hdf5", "MODEL/disorder", arma::hdf5_opts::append));
			_interaction.save(arma::hdf5_name(dir_realis + info + ".hdf5", "MODEL/interaction", arma::hdf5_opts::append));

			S_GS.save(arma::hdf5_name(dir_realis + info + ".hdf5", "ENTANGLEMENT/entropy GS", arma::hdf5_opts::append));
			S_site_GS.save(arma::hdf5_name(dir_realis + info + ".hdf5", "ENTANGLEMENT/single_site_entropy GS", arma::hdf5_opts::append));
			S_EX.save(arma::hdf5_name(dir_realis + info + ".hdf5", "ENTANGLEMENT/entropy excited", arma::hdf5_opts::append));
			S_site_EX.save(arma::hdf5_name(dir_realis + info + ".hdf5", "ENTANGLEMENT/single_site_entropy excited", arma::hdf5_opts::append));
			subsystem_sizes.save(arma::hdf5_name(dir_realis + info + ".hdf5", "ENTANGLEMENT/subsystem sizes", arma::hdf5_opts::append));

			Pq_GS.save(arma::hdf5_name(dir_realis + info + ".hdf5", "FRACTALITY/participation_ratio GS", arma::hdf5_opts::append));
			Sq_GS.save(arma::hdf5_name(dir_realis + info + ".hdf5", "FRACTALITY/information_entropy GS", arma::hdf5_opts::append));
			Pq_EX.save(arma::hdf5_name(dir_realis + info + ".hdf5", "FRACTALITY/participation_ratio excited", arma::hdf5_opts::append));
			Sq_EX.save(arma::hdf5_name(dir_realis + info + ".hdf5", "FRACTALITY/information_entropy excited", arma::hdf5_opts::append));
			q_ipr_list.save(arma::hdf5_name(dir_realis + info + ".hdf5", "FRACTALITY/qs", arma::hdf5_opts::append));

			Sx_GS.save(arma::hdf5_name(dir_realis + info + ".hdf5", "SPIN_EXP_VAL/Sx GS", arma::hdf5_opts::append));
			Sy_GS.save(arma::hdf5_name(dir_realis + info + ".hdf5", "SPIN_EXP_VAL/Sy GS", arma::hdf5_opts::append));
			Sz_GS.save(arma::hdf5_name(dir_realis + info + ".hdf5", "SPIN_EXP_VAL/Sz GS", arma::hdf5_opts::append));

			Sx_EX.save(arma::hdf5_name(dir_realis + info + ".hdf5", "SPIN_EXP_VAL/Sx excited", arma::hdf5_opts::append));
			Sy_EX.save(arma::hdf5_name(dir_realis + info + ".hdf5", "SPIN_EXP_VAL/Sy excited", arma::hdf5_opts::append));
			Sz_EX.save(arma::hdf5_name(dir_realis + info + ".hdf5", "SPIN_EXP_VAL/Sz excited", arma::hdf5_opts::append));

			SxSx_GS.save(arma::hdf5_name(dir_realis + info + ".hdf5", "CORRELATORS/SxSx GS", arma::hdf5_opts::append));
			SySy_GS.save(arma::hdf5_name(dir_realis + info + ".hdf5", "CORRELATORS/SySy GS", arma::hdf5_opts::append));
			SzSz_GS.save(arma::hdf5_name(dir_realis + info + ".hdf5", "CORRELATORS/SzSz GS", arma::hdf5_opts::append));

			SxSx_EX.save(arma::hdf5_name(dir_realis + info + ".hdf5", "CORRELATORS/SxSx excited", arma::hdf5_opts::append));
			SySy_EX.save(arma::hdf5_name(dir_realis + info + ".hdf5", "CORRELATORS/SySy excited", arma::hdf5_opts::append));
			SzSz_EX.save(arma::hdf5_name(dir_realis + info + ".hdf5", "CORRELATORS/SzSz excited", arma::hdf5_opts::append));
		}
		
		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start_re) << " s - - - - - - " << std::endl; // simulation end
		
	}
}

/// @brief Calculate AGPs from matrix elements of local operators
void ui::agp()
{
	std::string dir = this->saving_dir + "AGP" + kPSep;
	// if(this->op > 0) dir += "OtherObservables" + kPSep;
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info();
	std::string name = "";
	switch(this->op){
		case 1: name = "Sx_dot_Sx_last"; break;
		case 2: name = "Sz_dot_Sz_last"; break;
		case 3: name = "SzSz_last"; break;
		case 4: name = "SxSx_last"; break;
		case 5: name = "Sz_tot"; break;
		default: name = ""; break;
	}
	if(this->op > 0){
		info = name + info;
	}
	const size_t size = dim > 1e5? this->l_steps : dim;

	arma::vec betas = arma::regspace(0.0, 0.01, 10);
	// betas = arma::join_cols(arma::vec({0}), betas);
	arma::vec Z(betas.size(), arma::fill::zeros);
	arma::vec agp_temperature(betas.size(), arma::fill::zeros);
	arma::vec agp_temperature_regularized(betas.size(), arma::fill::zeros);

	arma::vec energy_density = arma::linspace(0, 1, 100);
	energy_density = arma::vec( {0.0, 0.038, 0.0676, 0.0921, 0.1132, 0.1319, 0.1488, 0.1641, 0.1783, 0.1916, 0.204, 0.2157, 0.2269, 0.2375, 0.2476, 0.2574, 0.2668, 0.2759, 0.2847, 0.2932, 0.3015, 0.3096, 0.3175, 0.3253, 0.3328, 0.3402, 0.3475, 0.3546, 0.3617, 0.3686, 0.3754, 0.3821, 0.3888, 0.3953, 0.4018, 0.4083, 0.4146, 0.4209, 0.4272, 0.4334, 0.4396, 0.4457, 0.4519, 0.4579, 0.464, 0.47, 0.476, 0.482, 0.488, 0.494, 0.5, 0.506, 0.512, 0.518, 0.524, 0.53, 0.536, 0.5421, 0.5481, 0.5543, 0.5604, 0.5666, 0.5728, 0.5791, 0.5854, 0.5917, 0.5982, 0.6047, 0.6112, 0.6179, 0.6246, 0.6314, 0.6383, 0.6454, 0.6525, 0.6598, 0.6672, 0.6747, 0.6825, 0.6904, 0.6985, 0.7068, 0.7153, 0.7241, 0.7332, 0.7426, 0.7524, 0.7625, 0.7731, 0.7843, 0.796, 0.8084, 0.8217, 0.8359, 0.8512, 0.8681, 0.8868, 0.9079, 0.9324, 0.962, 1.0} );
	// energy_density	=	arma::vec( {0.   , 0.184, 0.219, 0.241, 0.259, 0.273, 0.285, 0.296, 0.305, 0.314, 0.323, 0.33,
	// 								0.338, 0.344, 0.351, 0.357, 0.363, 0.369, 0.374, 0.38 , 0.385, 0.39 , 0.395, 0.4,
	// 								0.404, 0.409, 0.413, 0.418, 0.422, 0.426, 0.431, 0.435, 0.439, 0.443, 0.447, 0.451,
	// 								0.455, 0.459, 0.462, 0.466, 0.47 , 0.474, 0.477, 0.481, 0.485, 0.488, 0.492, 0.496,
	// 								0.499, 0.503, 0.507, 0.51 , 0.514, 0.518, 0.521, 0.525, 0.529, 0.532, 0.536, 0.54,
	// 								0.543, 0.547, 0.551, 0.555, 0.559, 0.563, 0.566, 0.57 , 0.574, 0.578, 0.582, 0.587,
	// 								0.591, 0.595, 0.599, 0.604, 0.608, 0.613, 0.618, 0.622, 0.627, 0.633, 0.638, 0.643,
	// 								0.649, 0.655, 0.661, 0.667, 0.674, 0.681, 0.688, 0.697, 0.705, 0.715, 0.725, 0.737,
	// 								0.751, 0.768, 0.79 , 0.823, 1});
	arma::vec count(energy_density.size()-1, arma::fill::zeros);
	arma::vec count_proj(energy_density.size()-1, arma::fill::zeros);
	arma::vec agp_energy(energy_density.size()-1, arma::fill::zeros);
	arma::vec agp_energy_proj(energy_density.size()-1, arma::fill::zeros);

	arma::vec energies(size, arma::fill::zeros);

	int Ll = this->L;
	int N = this->grain_size;
	double AGP = 0, TYP_SUSC = 0, SUSC = 0;
	int counter = 0;

	auto neighbor_generator = disorder<int>(this->seed);
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
		E.save(	  			arma::hdf5_name(dir_realis + info + ".hdf5", "energies"));
		betas.save(	  		arma::hdf5_name(dir_realis + info + ".hdf5", "betas",   arma::hdf5_opts::append));
		energy_density.save(arma::hdf5_name(dir_realis + info + ".hdf5", "energy_density",   arma::hdf5_opts::append));
		
		start = std::chrono::system_clock::now();
		// arma::Mat<element_type> mat_elem = V * Sz_ops[i] * V.t();
		auto _operator = QOps::genOp();
		switch(this->op){
			case 0:
			{
				auto kernel_def = [Ll, N](u64 state){ 
					auto [val1, tmp22] = operators::sigma_z<cpx>(state, Ll, Ll - 1 );
					return std::make_pair(state, val1);
					};
				_operator = QOps::genOp(this->L, std::move(kernel_def), 1.0);
			}
			case 1:
			{
					auto kernel = [Ll, N, &neighbor_generator](u64 state){ 
						int nei = neighbor_generator.uniform_dist<int>(0, N-1);
						auto [val1, state_out1] = operators::sigma_x<cpx>(state, Ll, nei );
						auto [val2, state_out2] = operators::sigma_x<cpx>(state_out1, Ll, Ll - 1 );
						return std::make_pair(state_out2, val1 * val2);
					};
					_operator = QOps::genOp(this->L, std::move(kernel), 1.0);
			}
				break;
			case 2:
			{
				auto kernel = [Ll, N, &neighbor_generator](u64 state){ 
						int nei = neighbor_generator.uniform_dist<int>(0, N-1);
						auto [val1, state_out1] = operators::sigma_z<cpx>(state, Ll, nei );
						auto [val2, state_out2] = operators::sigma_z<cpx>(state_out1, Ll, Ll - 1 );
						return std::make_pair(state_out2, val1 * val2);
				};
				_operator = QOps::genOp(this->L, std::move(kernel), 1.0);
			}
				break;
			case 3:
			{
				auto kernel = [Ll, N](u64 state){ 
					auto [val1, state_out1] = operators::sigma_z<cpx>(state, Ll, Ll - 2 );
					auto [val2, state_out2] = operators::sigma_z<cpx>(state_out1, Ll, Ll - 1 );
					return std::make_pair(state_out2, val1 * val2);
				};
				_operator = QOps::genOp(this->L, std::move(kernel), 1.0);
			}
				break;
			case 4:
			{
				auto kernel = [Ll, N](u64 state){ 
					auto [val1, state_out1] = operators::sigma_x<cpx>(state, Ll, Ll - 2 );
					auto [val2, state_out2] = operators::sigma_x<cpx>(state_out1, Ll, Ll - 1 );
					return std::make_pair(state_out2, val1 * val2);
				};
				_operator = QOps::genOp(this->L, std::move(kernel), 1.0);
			}
				break;
			default:
			{
				auto kernel_def = [Ll, N](u64 state){ 
					auto [val1, tmp22] = operators::sigma_z<cpx>(state, Ll, Ll - 1 );
					return std::make_pair(state, val1);
					};
				_operator = QOps::genOp(this->L, std::move(kernel_def), 1.0);
			}
		}
		arma::sp_mat oper(dim, dim);
		if(this->op == 5){
			for(int j = N; j < this->L; j++){
				auto kernel_def = [Ll, j](u64 state){ 
					auto [val1, tmp22] = operators::sigma_z<cpx>(state, Ll, j );
					return std::make_pair(state, val1);
					};
				_operator = QOps::genOp(this->L, std::move(kernel_def), 1.0);
				oper += arma::real(_operator.to_matrix(dim));
			}
			oper = oper / std::sqrt(this->L_loc);
		} else {
			oper = arma::real(_operator.to_matrix(dim));
		}
		
		arma::Mat<element_type> mat_elem = V.t() * oper * V;
		auto [_Z, _count, _count_proj,AGP_T, AGP_T_reg, AGP_E, AGP_E_proj] = adiabatics::gauge_potential_finite_T(mat_elem, E, betas, energy_density);
		auto [_agp, _typ_susc, _susc, tmp] = adiabatics::gauge_potential(mat_elem, E, this->L);

		std::cout << " - - - - - - finished Sz_L matrix elements in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		// #ifndef MY_MAC
		{
			_Z.save(	  	arma::hdf5_name(dir_realis + info + ".hdf5", "Z",   arma::hdf5_opts::append));
			betas.save(	  		arma::hdf5_name(dir_realis + info + ".hdf5", "betas",   arma::hdf5_opts::append));
			
			_count.save(arma::hdf5_name(dir_realis + info + ".hdf5", "count",   arma::hdf5_opts::append));
			_count_proj.save(arma::hdf5_name(dir_realis + info + ".hdf5", "count_proj",   arma::hdf5_opts::append));
			energy_density.save(arma::hdf5_name(dir_realis + info + ".hdf5", "energy_density",   arma::hdf5_opts::append));

			AGP_T.save(	  	arma::hdf5_name(dir_realis + info + ".hdf5", "agp_T",   arma::hdf5_opts::append));
			AGP_T_reg.save(	arma::hdf5_name(dir_realis + info + ".hdf5", "agp_T_reg",   arma::hdf5_opts::append));
			AGP_E.save(	  	arma::hdf5_name(dir_realis + info + ".hdf5", "agp_E",   arma::hdf5_opts::append));
			AGP_E_proj.save(	arma::hdf5_name(dir_realis + info + ".hdf5", "agp_E_proj",   arma::hdf5_opts::append));

			arma::vec({_agp}).save(	arma::hdf5_name(dir_realis + info + ".hdf5", "AGP",   arma::hdf5_opts::append));
			arma::vec({_susc}).save(	arma::hdf5_name(dir_realis + info + ".hdf5", "SUSC",   arma::hdf5_opts::append));
			arma::vec({_typ_susc}).save(	arma::hdf5_name(dir_realis + info + ".hdf5", "TYP_SUSC",   arma::hdf5_opts::append));
		}
		// #endif
		count += _count;
		count_proj += _count_proj;
		Z += _Z;
		agp_temperature += AGP_T;
		agp_temperature_regularized += AGP_T_reg;
		agp_energy += AGP_E;
		agp_energy_proj += AGP_E_proj;

		AGP += _agp;
		SUSC += _susc;
		TYP_SUSC += _typ_susc;
		counter++;
		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start_re) << " s - - - - - - " << std::endl; // simulation end
	}
	if(counter == 0) return;
	
	#ifdef MY_MAC
		AGP /= double(counter);
		SUSC /= double(counter);
		TYP_SUSC /= double(counter);

		Z /= double(counter);
		agp_temperature /= double(counter);
		agp_temperature_regularized /= double(counter);

		betas.save(	  		arma::hdf5_name(dir + info + ".hdf5", "betas"));
		Z.save(	  			arma::hdf5_name(dir + info + ".hdf5", "Z",   arma::hdf5_opts::append));
		count.save(	  		arma::hdf5_name(dir + info + ".hdf5", "count",   arma::hdf5_opts::append));
		count_proj.save(	arma::hdf5_name(dir + info + ".hdf5", "count_proj",   arma::hdf5_opts::append));
		energy_density.save(arma::hdf5_name(dir + info + ".hdf5", "energy_density",   arma::hdf5_opts::append));
		
		agp_temperature.save(			 arma::hdf5_name(dir + info + ".hdf5", "agp_T",   arma::hdf5_opts::append));
		agp_temperature_regularized.save(arma::hdf5_name(dir + info + ".hdf5", "agp_T_reg",   arma::hdf5_opts::append));
		
		agp_energy.save(	  		arma::hdf5_name(dir + info + ".hdf5", "agp_E",   arma::hdf5_opts::append));
		agp_energy_proj.save(arma::hdf5_name(dir + info + ".hdf5", "agp_E_proj",   arma::hdf5_opts::append));

		arma::vec({AGP}).save(	arma::hdf5_name(dir + info + ".hdf5", "AGP",   arma::hdf5_opts::append));
		arma::vec({SUSC}).save(	arma::hdf5_name(dir + info + ".hdf5", "SUSC",   arma::hdf5_opts::append));
		arma::vec({TYP_SUSC}).save(	arma::hdf5_name(dir + info + ".hdf5", "TYP_SUSC",   arma::hdf5_opts::append));
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
					auto [val1, tmp22] = operators::sigma_z<cpx>(state, Ll, Ll - 1 );
					return std::make_pair(state, val1);
					};
		auto _operator = QOps::genOp(this->L, std::move(kernel_def), 1.0);
		arma::sp_mat oper = arma::real(_operator.to_matrix(dim));
		
		arma::Mat<element_type> mat_elem = V.t() * oper * V;
		auto [_susc, _susc_r] = adiabatics::gauge_potential_save(mat_elem, E);

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

void ui::agp_mu()
{
	std::string dir = this->saving_dir + "AGP" + kPSep + "CUTOFF" + kPSep;
	// if(this->op > 0) dir += "OtherObservables" + kPSep;
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info();
	const size_t size = dim > 1e5? this->l_steps : dim;

	arma::vec energies(size, arma::fill::zeros);
	arma::vec power = arma::linspace(-2, 2, 9);
	std::cout << power << std::endl;
	arma::vec cutoff(power.size(), arma::fill::zeros);
	for(int ii = 0; ii < power.size(); ii++)
		cutoff(ii) = 1.0 / double(dim) / std::pow(this->L, power(ii));

	arma::vec susc(cutoff.size(), arma::fill::zeros);
	arma::vec susc_typ(cutoff.size(), arma::fill::zeros);

	int Ll = this->L;
	int N = this->grain_size;
	int counter = 0;

	auto kernel_def = [Ll, N](u64 state){ 
				auto [val1, tmp22] = operators::sigma_z<cpx>(state, Ll, Ll - 1 );
				return std::make_pair(state, val1);
				};
	auto _operator = QOps::genOp(this->L, std::move(kernel_def), 1.0);
	arma::sp_mat oper = arma::real(_operator.to_matrix(dim));

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
		// E.save(arma::hdf5_name(dir_realis + info + ".hdf5", "energies"));
		energies += E;

		start = std::chrono::system_clock::now();
		arma::Mat<element_type> mat_elem = V.t() * oper * V;
		std::cout << " - - - - - - finished Sz_L matrix elements in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		
		start = std::chrono::system_clock::now();
		auto [_susc, _susc_typ] = adiabatics::gauge_potential_mu(mat_elem, E, cutoff);
		std::cout << " - - - - - - finished AGP in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		// #ifndef MY_MAC
		{
			cutoff.save(arma::hdf5_name(dir_realis + info + ".hdf5", "cutoff"));
			_susc.save(	arma::hdf5_name(dir_realis + info + ".hdf5", "susc", arma::hdf5_opts::append));
			_susc_typ.save(	arma::hdf5_name(dir_realis + info + ".hdf5", "susc_typ", arma::hdf5_opts::append));
		}
		// #endif
		susc += _susc;
		susc_typ += _susc_typ;
		counter++;
		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start_re) << " s - - - - - - " << std::endl; // simulation end
	}
	if(counter == 0) return;
	
	#ifdef MY_MAC
		susc /= double(counter);
		susc_typ /= double(counter);
		energies /= double(counter);

		energies.save(arma::hdf5_name(dir + info + ".hdf5", "energies"));
		susc.save(    arma::hdf5_name(dir + info + ".hdf5", "susc", arma::hdf5_opts::append));
		susc_typ.save(arma::hdf5_name(dir + info + ".hdf5", "susc_typ", arma::hdf5_opts::append));
		cutoff.save(  arma::hdf5_name(dir + info + ".hdf5", "cutoff", arma::hdf5_opts::append));
	#endif
}

/// @brief Calculate matrix elements of local operators
void ui::matrix_elements()
{
	// std::string dir = this->saving_dir + "MatrixElements" + kPSep;
	std::string dir = this->saving_dir + "SpectralsSiteResolved" + kPSep;
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info();

	// arma::vec sites = arma::linspace(0, this->L-1, this->L);
	arma::vec sites = arma::linspace(0, this->L-1, this->L);
	// arma::Col<int> sites = arma::Col<int>({(int)this->L - 1});

	// arma::vec agp_norm_Sz(sites.size(), arma::fill::zeros);
	// arma::vec typ_susc_Sz(sites.size(), arma::fill::zeros);
	// arma::mat diag_mat_elem_Sz(dim, sites.size(), arma::fill::zeros);

	// arma::vec agp_norm_SzSz(sites.size(), arma::fill::zeros);
	// arma::vec typ_susc_SzSz(sites.size(), arma::fill::zeros);
	// arma::mat diag_mat_elem_SzSz(dim, sites.size(), arma::fill::zeros);

	// arma::vec agp_norm_kin(sites.size(), arma::fill::zeros);
	// arma::vec typ_susc_kin(sites.size(), arma::fill::zeros);
	// arma::mat diag_mat_elem_kin(dim, sites.size(), arma::fill::zeros);
	arma::vec energies(dim, arma::fill::zeros);

	int Ll = this->L;
	int N = this->grain_size;

	arma::vec omegax = arma::logspace(int(std::log10(0.1/dim)), int(std::log10( 5 + this->L )), 10 * this->L);
	const arma::vec energy_density = arma::regspace(0.05, 0.02, 0.95);
	const double chi = 0.341345;

	const double wH = std::sqrt(this->L) / (chi * dim);
	double tH = 1. / wH;
	double r1 = 0.0, r2 = 0.0;
	int time_end = (int)std::ceil(std::log10(50 * tH));
	time_end = (time_end / std::log10(tH) < 1.5) ? time_end + 2 : time_end;

	arma::vec times = arma::logspace(-2, time_end, 2000);


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
		
		arma::Mat<element_type> agp_norm_Sz_r(dim, sites.size(), arma::fill::zeros);
		arma::Mat<element_type> typ_susc_Sz_r(dim, sites.size(), arma::fill::zeros);
		arma::Mat<element_type> diag_mat_elem_Sz_r(dim, sites.size(), arma::fill::zeros);
		
		arma::sp_mat H = this->ptr_to_model->get_hamiltonian();
		double meanH = arma::trace(H) / double(dim);
		H = H - meanH *  arma::eye<arma::sp_mat>(dim, dim);
		arma::sp_mat H2 = H*H;
		double varH = arma::trace(H2) / double(dim);// - meanH * meanH;
		
		arma::vec gec(dim, arma::fill::zeros);
	#pragma omp parallel for
		for(u64 k = 0; k < dim; k++)
			gec(k) = 2 * H2(k,k) - H(k,k) * H(k,k); 
		gec = gec / varH;
		gec.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "gec",   arma::hdf5_opts::append));
		// arma::Mat<element_type> agp_norm_Sx_r(dim, sites.size(), arma::fill::zeros);
		// arma::Mat<element_type> typ_susc_Sx_r(dim, sites.size(), arma::fill::zeros);
		// arma::Mat<element_type> diag_mat_elem_Sx_r(dim, sites.size(), arma::fill::zeros);

		// arma::Mat<element_type> agp_norm_SzSz_r(dim, sites.size(), arma::fill::zeros);
		// arma::Mat<element_type> typ_susc_SzSz_r(dim, sites.size(), arma::fill::zeros);
		// arma::Mat<element_type> diag_mat_elem_SzSz_r(dim, sites.size(), arma::fill::zeros);

		// arma::Mat<element_type> agp_norm_kin_r(dim, sites.size(), arma::fill::zeros);
		// arma::Mat<element_type> typ_susc_kin_r(dim, sites.size(), arma::fill::zeros);
		// arma::Mat<element_type> diag_mat_elem_kin_r(dim, sites.size(), arma::fill::zeros);
		arma::vec Hdiagonal = arma::diagvec( this->ptr_to_model->get_dense_hamiltonian() );

		double E_av = arma::trace(E) / double(dim);
		auto i = min_element(begin(Hdiagonal), end(Hdiagonal), [=](double x, double y) {
			return abs(x - E_av) < abs(y - E_av);
		});
		const u64 idx_state = i - begin(Hdiagonal);
		double quench_E = Hdiagonal(idx_state);

		arma::Col<element_type> coeff = V.row(idx_state).t();
		for(int i = 0; i < sites.size(); i++)
		{
			int site = sites(i);
			// double _agp, _typ_susc, _susc;
			arma::vec _susc, _susc_r;
			start = std::chrono::system_clock::now();
			// arma::Mat<element_type> mat_elem = V * Sz_ops[i] * V.t();
			auto kernel_Sz = [Ll, site](u64 state){ 
				auto [val, tmp11] = operators::sigma_z<cpx>(state, Ll, site ); 
				return std::make_pair(state, val); 
				};
			auto _operator = QOps::genOp(this->L, std::move(kernel_Sz), 1.0);
			arma::sp_mat op_mat = arma::real(_operator.to_matrix(dim));
			double HSnorm = arma::trace(op_mat * op_mat) / double(dim);
			op_mat = op_mat / std::sqrt(HSnorm);

			arma::Mat<element_type> mat_elem = V.t() * op_mat * V;
			// arma::Mat<element_type> _submat_ = mat_elem.submat(idx_min, idx_min, idx_max -1, idx_max - 1);
			// _submat_.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "MAT_ELEM/Sz_i=" + std::to_string(site),   arma::hdf5_opts::append));
			// std::tie(_agp, _typ_susc, _susc, tmp) = adiabatics::gauge_potential(mat_elem, E, this->L);
			std::tie(_susc, _susc_r) = adiabatics::gauge_potential_save(mat_elem, E);
			agp_norm_Sz_r.col(i) = _susc;
			typ_susc_Sz_r.col(i) = _susc_r;
			diag_mat_elem_Sz_r.col(i) = arma::diagvec(mat_elem);
			
    		std::cout << " - - - - - - finished Sz matrix elements for site i = " << sites(i) << "in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
			
			arma::Col<element_type> time_evolution(times.size(), arma::fill::zeros);
			// arma::Col<element_type> Sz_state(dim, arma::fill::zeros);
			// for(int k = 0; k < dim; k++)
			{
				// Sz_state(k) = op_mat(k,k);
			#pragma omp parallel for
				for(long t_idx = 0; t_idx < times.size(); t_idx++){
					// quench(t_idx) = std::real( arma::cdot(psi.col(t_idx), op_mat * psi.col(t_idx)) );
					double time = times(t_idx);
					arma::cx_vec init_state(dim, arma::fill::zeros);
					for(long alfa = 0; alfa < dim; alfa++)
					{
						auto state = V.col(alfa);
						// psi.col(t_idx) += std::exp(-1i * time * E(alfa)) * state * state(idx);
						init_state += std::exp(-1i * time * E(alfa)) * state * coeff(alfa);
					}
					time_evolution(t_idx) = std::real( arma::cdot(init_state, op_mat * init_state) );
				}
			}
			arma::vec state_Sz = arma::vec({op_mat(idx_state, idx_state)});
			state_Sz.save(arma::hdf5_name(dir_realis + info + ".hdf5", "j=" + std::to_string(site) + "/state_Sz",   arma::hdf5_opts::append));
			time_evolution.save(arma::hdf5_name(dir_realis + info + ".hdf5", "j=" + std::to_string(site) + "/time_evolution",   arma::hdf5_opts::append));

			auto [autocorr, _LTA] = spectrals::autocorrelation_function(mat_elem, E, times);
			arma::vec LTA = arma::vec( {_LTA} );

			LTA.save(arma::hdf5_name(dir_realis + info + ".hdf5", "j=" + std::to_string(site) + "/LTA",   arma::hdf5_opts::append));
			autocorr.save(arma::hdf5_name(dir_realis + info + ".hdf5", "j=" + std::to_string(site) + "/autocorrelation",   arma::hdf5_opts::append));
			std::cout << " - - - - - - finished Sz time evolution for site i = " << sites(i) << "in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
				
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
							if(idx < omegax.size() && idx >= 0){
								const double _b_ = std::abs(mat_elem(n, m));
								_spectral_fun_eps(idx, ii) += 2 * _b_ * _b_;
								_spectral_fun_typ_eps(idx, ii) += 2 * std::log(_b_ * _b_);
								_element_count_eps(idx, ii) += 2;
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
					if(idx < omegax.size() && idx >= 0){
						const double _b_ = std::abs(mat_elem(n, m));
						_spectral_fun(idx) += 2 * _b_ * _b_;
						_spectral_fun_typ(idx) += 2 * std::log(_b_ * _b_);
						_element_count(idx) += 2;
					}
				}	
			}
			_spectral_fun.save(arma::hdf5_name(dir_realis + info + ".hdf5", "j=" + std::to_string(site) + "/spectral_fun",   arma::hdf5_opts::append));
			_spectral_fun_typ.save(arma::hdf5_name(dir_realis + info + ".hdf5", "j=" + std::to_string(site) + "/spectral_fun_typ",   arma::hdf5_opts::append));
			_element_count.save(arma::hdf5_name(dir_realis + info + ".hdf5", "j=" + std::to_string(site) + "/element_count",   arma::hdf5_opts::append));
			std::cout << " - - - - - - finished spectral function for site i = " << sites(i) << "in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end

			// start = std::chrono::system_clock::now();
			// // arma::Mat<element_type> mat_elem = V * Sz_ops[i] * V.t();
			// auto kernel_Sx = [Ll, site](u64 state){ 
			// 	auto [val, num] = operators::sigma_x<cpx>(state, Ll, site ); 
			// 	return std::make_pair(num, val); 
			// 	};
			// _operator = QOps::genOp(this->L, std::move(kernel_Sx), 1.0);
			// op = arma::real(_operator.to_matrix(dim));
			// mat_elem = V.t() * op * V;
			// _submat_ = mat_elem.submat(idx_min, idx_min, idx_max -1, idx_max - 1);
			// _submat_.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "MAT_ELEM/Sx_i=" + std::to_string(site),   arma::hdf5_opts::append));
			// // std::tie(_agp, _typ_susc, _susc, tmp) = adiabatics::gauge_potential(mat_elem, E, this->L);
			// // agp_norm_Sx_r(i) = _agp;
			// // typ_susc_Sx_r(i) = _typ_susc;
			// std::tie(_susc, _susc_r) = adiabatics::gauge_potential_save(mat_elem, E);
			// agp_norm_Sx_r.col(i) = _susc;
			// typ_susc_Sx_r.col(i) = _susc_r;
			// diag_mat_elem_Sx_r.col(i) = arma::diagvec(mat_elem);
			
    		// std::cout << " - - - - - - finished Sx matrix elements for site i = " << sites(i) << "in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
			// {
			// 	start = std::chrono::system_clock::now();
			// 	auto kernel_SzSz = [Ll, N, site, &neighbor_generator](u64 state){ 
			// 		int nei = neighbor_generator.uniform_dist<int>(0, N-1);
			// 		auto [val1, tmp22] = operators::sigma_z<cpx>(state, Ll, site );
			// 		auto [val2, tmp33] = operators::sigma_z<cpx>(state, Ll, nei );
			// 		return std::make_pair(state, val1 * val2);
			// 		};
			// 	_operator = QOps::genOp(this->L, std::move(kernel_SzSz), 1.0);
			// 	op = arma::real(_operator.to_matrix(dim));
			// 	mat_elem = V.t() * op * V;
			// 	_submat_ = mat_elem.submat(idx_min, idx_min, idx_max -1, idx_max - 1);
			// 	_submat_.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "MAT_ELEM/SzSz_i=" + std::to_string(site),   arma::hdf5_opts::append));
			// 	// std::tie(_agp, _typ_susc, _susc, tmp) = adiabatics::gauge_potential(mat_elem, E, this->L);
			// 	// agp_norm_SzSz_r(i) = _agp;
			// 	// typ_susc_SzSz_r(i) = _typ_susc;
			// 	std::tie(_susc, _susc_r) = adiabatics::gauge_potential_save(mat_elem, E);
			// 	agp_norm_SzSz_r.col(i) = _susc;
			// 	typ_susc_SzSz_r.col(i) = _susc_r;
			// 	diag_mat_elem_SzSz_r.col(i) = arma::diagvec(mat_elem); 
				
				// std::cout << " - - - - - - finished SzSz matrix elements for site i = " << sites(i) << "in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
			// 	start = std::chrono::system_clock::now();
			// 	auto kernel_kin = [Ll, N, site, &neighbor_generator](u64 state){ 
			// 		int nei = site==0? site+1 : neighbor_generator.uniform_dist<int>(0, N-1);
			// 		// auto [spin1, tmp11] = operators::sigma_z<cpx>(state, Ll, site );
			// 		// auto [spin2, tmp22] = operators::sigma_z<cpx>(state, Ll, nei );
			// 		// if(std::real(spin1 * spin2) < 0)
			// 		{
			// 			auto [val1, num] = operators::sigma_x<cpx>(state, Ll, site );
			// 			auto [val2, num2] = operators::sigma_x<cpx>(num, Ll, nei );
			// 			return std::make_pair(num2, val1 * val2); 
			// 		} 
			// 		// else 
			// 		// 	return std::make_pair(state, cpx(0.0));
			// 		};
			// 	_operator = QOps::genOp(this->L, std::move(kernel_kin), 1.0);
			// 	op = arma::real(_operator.to_matrix(dim));
			// 	mat_elem = V.t() * op * V;
			// 	_submat_ = mat_elem.submat(idx_min, idx_min, idx_max -1, idx_max - 1);
			// 	_submat_.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "MAT_ELEM/kin_i=" + std::to_string(site),   arma::hdf5_opts::append));
			// 	// std::tie(_agp, _typ_susc, _susc, tmp) = adiabatics::gauge_potential(mat_elem, E, this->L);
			// 	// agp_norm_kin_r(i) = _agp;
			// 	// typ_susc_kin_r(i) = _typ_susc;
			// 	std::tie(_susc, _susc_r) = adiabatics::gauge_potential_save(mat_elem, E);
			// 	agp_norm_kin_r.col(i) = _susc;
			// 	typ_susc_kin_r.col(i) = _susc_r;
			// 	diag_mat_elem_kin_r.col(i) = arma::diagvec(mat_elem); 
			// 	std::cout << " - - - - - - finished kinetic matrix elements for site i = " << sites(i) << "in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
			// }
		}
		// #ifndef MY_MAC
		{
			agp_norm_Sz_r.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "susc",   arma::hdf5_opts::append));
			// agp_norm_Sx_r.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "SUSC/Sx",   arma::hdf5_opts::append));
			// agp_norm_SzSz_r.save( arma::hdf5_name(dir_realis + info + ".hdf5", "SUSC/SzSz", arma::hdf5_opts::append));
			// agp_norm_kin_r.save(  arma::hdf5_name(dir_realis + info + ".hdf5", "SUSC/SxSx",  arma::hdf5_opts::append));

			typ_susc_Sz_r.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "susc_r",   arma::hdf5_opts::append));
			// typ_susc_Sx_r.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "SUSC_R/Sx",   arma::hdf5_opts::append));
			// typ_susc_SzSz_r.save( arma::hdf5_name(dir_realis + info + ".hdf5", "SUSC_R/SzSz", arma::hdf5_opts::append));
			// typ_susc_kin_r.save(  arma::hdf5_name(dir_realis + info + ".hdf5", "SUSC_R/SxSx",  arma::hdf5_opts::append));

			diag_mat_elem_Sz_r.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "diag_mat",   arma::hdf5_opts::append));
			// diag_mat_elem_Sx_r.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "DIAG_MAT/Sx",   arma::hdf5_opts::append));
			// diag_mat_elem_SzSz_r.save( arma::hdf5_name(dir_realis + info + ".hdf5", "DIAG_MAT/SzSz", arma::hdf5_opts::append));
			// diag_mat_elem_kin_r.save(  arma::hdf5_name(dir_realis + info + ".hdf5", "DIAG_MAT/SxSx",  arma::hdf5_opts::append));
		}
		// #endif
		
		// agp_norm_Sz += agp_norm_Sz_r;
		// typ_susc_Sz += arma::log(typ_susc_Sz_r);
		// diag_mat_elem_Sz += diag_mat_elem_Sz_r;

		// agp_norm_SzSz += agp_norm_SzSz_r;
		// typ_susc_SzSz += arma::log(typ_susc_SzSz_r);
		// diag_mat_elem_SzSz += diag_mat_elem_SzSz_r;

		// agp_norm_kin += agp_norm_kin_r;
		// typ_susc_kin += arma::log(typ_susc_kin_r);
		// diag_mat_elem_kin += diag_mat_elem_kin_r;

		// energies += E;
		// counter++;
		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start_re) << " s - - - - - - " << std::endl; // simulation end
	}
	if(counter == 0) return;
	
	// #ifdef MY_MAC
	// 	agp_norm_Sz /= double(counter);
	// 	typ_susc_Sz = arma::exp(typ_susc_Sz / double(counter));
	// 	diag_mat_elem_Sz /= double(counter);

	// 	agp_norm_SzSz /= double(counter);
	// 	typ_susc_SzSz = arma::exp(typ_susc_SzSz / double(counter));
	// 	diag_mat_elem_SzSz /= double(counter);

	// 	agp_norm_kin /= double(counter);
	// 	typ_susc_kin = arma::exp(typ_susc_kin / double(counter));
	// 	diag_mat_elem_kin /= double(counter);

	// 	energies /= double(counter);
	// 	sites.save(arma::hdf5_name(dir + info + ".hdf5", "sites"));
	// 	// agp_norm.save(arma::hdf5_name(dir + info + ".hdf5", "agp norm", arma::hdf5_opts::append));
	// 	// typ_susc.save(arma::hdf5_name(dir + info + ".hdf5", "typical susceptibility", arma::hdf5_opts::append));
	// 	// susc.save(arma::hdf5_name(dir + info + ".hdf5", "susceptibility", arma::hdf5_opts::append));
	// 	energies.save(		arma::hdf5_name(dir + info + ".hdf5", "energies",   arma::hdf5_opts::append));
	// 	agp_norm_Sz.save(	arma::hdf5_name(dir + info + ".hdf5", "AGP/Sz",   arma::hdf5_opts::append));
	// 	agp_norm_SzSz.save( arma::hdf5_name(dir + info + ".hdf5", "AGP/SzSz", arma::hdf5_opts::append));
	// 	agp_norm_kin.save(  arma::hdf5_name(dir + info + ".hdf5", "AGP/kin",  arma::hdf5_opts::append));

	// 	typ_susc_Sz.save(	arma::hdf5_name(dir + info + ".hdf5", "TYP_SUSC/Sz",   arma::hdf5_opts::append));
	// 	typ_susc_SzSz.save( arma::hdf5_name(dir + info + ".hdf5", "TYP_SUSC/SzSz", arma::hdf5_opts::append));
	// 	typ_susc_kin.save(  arma::hdf5_name(dir + info + ".hdf5", "TYP_SUSC/kin",  arma::hdf5_opts::append));

	// 	diag_mat_elem_Sz.save(   arma::hdf5_name(dir + info + ".hdf5", "DIAG_MAT/Sz",   arma::hdf5_opts::append));
	// 	diag_mat_elem_SzSz.save( arma::hdf5_name(dir + info + ".hdf5", "DIAG_MAT/SzSz", arma::hdf5_opts::append));
	// 	diag_mat_elem_kin.save(  arma::hdf5_name(dir + info + ".hdf5", "DIAG_MAT/kin",  arma::hdf5_opts::append));
	// #endif
}

/// @brief Calculate matrix elements of local operators
void ui::correlators()
{
	std::string dir = this->saving_dir + "Correlators" + kPSep;
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info();

	const size_t size = dim > 1e5? this->l_steps : dim;

	// arma::vec sites = arma::linspace(0, this->L-1, this->L);
	const int Lhalf = this->L / 2;
	// std::vector<std::pair<int,int>> site_pairs = std::vector<std::pair<int,int>>(
	// 		{std::make_pair(this->grain_size, this->grain_size + 1), std::make_pair(this->grain_size, this->grain_size + 2),
	// 		std::make_pair(this->grain_size, this->L - 1), std::make_pair(this->grain_size + 1, this->L - 1),
	// 		std::make_pair(Lhalf, Lhalf + 1), std::make_pair(Lhalf, this->L - 1)}
	// 		);
	std::vector<std::pair<int,int>> site_pairs = std::vector<std::pair<int,int>>(
			{std::make_pair(this->L - 2, this->L - 1)}
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

	arma::vec energies(size, arma::fill::zeros);

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
		double E_av = arma::trace(E) / double(dim);

		auto i = min_element(begin(E), end(E), [=](double x, double y) {
			return abs(x - E_av) < abs(y - E_av);
		});
		const long Eav_idx = i - begin(E);

		std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
		createDirs(dir_realis);
		E.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "energies"));
		

		arma::vec LTA_r(site_pairs.size() + 1, arma::fill::zeros);
		arma::vec agp_norm_r(site_pairs.size() + 1, arma::fill::zeros);
		arma::vec typ_susc_r(site_pairs.size() + 1, arma::fill::zeros);
		arma::Mat<element_type> diag_mat_elem_r(size, site_pairs.size(), arma::fill::zeros);

		const double window_width = 0.001 * this->L;
		spectrals::preset_omega set_omega(E, window_width, E(Eav_idx));
		arma::vec omegas(set_omega.num_of_omegas, arma::fill::zeros);
		arma::Mat<element_type> spectral_funs(set_omega.num_of_omegas, site_pairs.size(), arma::fill::zeros);

	// 	arma::mat quench_AFM(times.size(), site_pairs.size(), arma::fill::zeros);
	// 	arma::mat quench_random(times.size(), site_pairs.size(), arma::fill::zeros);
	// 	arma::mat autocorr(times.size(), site_pairs.size(), arma::fill::zeros);

	// 	arma::cx_mat psi_AFM(dim, times.size(), arma::fill::zeros);
	// 	arma::cx_mat psi_random(dim, times.size(), arma::fill::zeros);
	// 	u64 idx = (dim - 1) / 3;

	// 	start = std::chrono::system_clock::now();
	// 	arma::cx_vec random_state = this->random_product_state();
	// 	// arma::mat R(2, 2);
	// 	// arma::vec spiral_state = up;
	// 	// for (int j = 1; j < this->L; j++)
	// 	// {
	// 	// 	auto the = pi / this->L * double(j);
	// 	// 	R(0, 0) = std::cos(the); R(1, 1) = std::cos(the);
	// 	// 	R(0, 1) = std::sin(the); R(1, 0) = -std::sin(the);
	// 	// 	spiral_state = arma::kron(spiral_state, R * up);
	// 	// }
	// 	// spiral_state = arma::normalise(spiral_state);
	// 	// arma::vec coeff_spiral(dim);
	// 	arma::cx_vec coeff_random(dim);
	// #pragma omp parallel for
	// 	for(long alfa = 0; alfa < dim; alfa++)
	// 	{
	// 		auto state = V.col(alfa);
	// 		// coeff_spiral(alfa) = dot_prod(state, spiral_state);
	// 		coeff_random(alfa) = dot_prod(state, random_state);
	// 	}
	// 	std::cout << " - - - - - - finished preparing initial states FM, AFM, spiral, random product in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();
	// #pragma omp parallel for
	// 	for(long t_idx = 0; t_idx < times.size(); t_idx++)
	// 	{
	// 		double time = times(t_idx);
	// 		for(long alfa = 0; alfa < dim; alfa++)
	// 		{
	// 			auto state = V.col(alfa);
	// 			// psi_FM.col(t_idx) += std::exp(-1i * time * E(alfa)) * state * state(0);
	// 			psi_AFM.col(t_idx) += std::exp(-1i * time * E(alfa)) * state * state(idx);
	// 			// psi_spiral.col(t_idx) += std::exp(-1i * time * E(alfa)) * state * coeff_spiral(alfa);
	// 			psi_random.col(t_idx) += std::exp(-1i * time * E(alfa)) * state * coeff_random(alfa);
	// 		}
	// 	}
	// 	arma::vec quench_E(2);
	// 	auto H = this->ptr_to_model->get_hamiltonian();
	// 	quench_E(0) = H(idx, idx);
	// 	quench_E(1) = std::real(arma::cdot(random_state, H * random_state));
	// 	// quench_E(2) = arma::cdot(spiral_state, H * spiral_state);

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
				auto [val1, tmp22] = operators::sigma_z<cpx>(state, Ll, site_1 );
				auto [val2, tmp33] = operators::sigma_z<cpx>(state, Ll, site_2 );
				return std::make_pair(state, val1 * val2);
				};
			auto _operator = QOps::genOp(this->L, std::move(kernel), 1.0);
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
		// 	start = std::chrono::system_clock::now();
		// #pragma omp parallel for
		// 	for(long t_idx = 0; t_idx < times.size(); t_idx++)
		// 	{
		// 		// quench_FM(t_idx, i) = std::real(arma::cdot(psi_FM.col(t_idx), op * psi_FM.col(t_idx)));
		// 		quench_AFM(t_idx, i) = std::real(arma::cdot(psi_AFM.col(t_idx), op * psi_AFM.col(t_idx)));
		// 		// quench_spiral(t_idx, i) = std::real(arma::cdot(psi_spiral.col(t_idx), op * psi_spiral.col(t_idx)));
		// 		quench_random(t_idx, i) = std::real(arma::cdot(psi_random.col(t_idx), op * psi_random.col(t_idx)));
		// 	}
    	// 	std::cout << " - - - - - - finished time evolution for sites: i=" << site_1 << ", j=" << site_2 << " in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		// 	start = std::chrono::system_clock::now();
		// 	auto [_autocorr, LTA] = spectrals::autocorrelation_function(mat_elem, E, times);
		// 	autocorr.col(i) = _autocorr;
		// 	LTA_r(i) = LTA;
    		// std::cout << " - - - - - - finished auto correlator time evolution for sites: i=" << site_1 << ", j=" << site_2 << " in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		}
		start = std::chrono::system_clock::now();
		// arma::Mat<element_type> mat_elem = V * Sz_ops[i] * V.t();
		auto kernel = [Ll, N](u64 state){ 
			auto [val1, tmp22] = operators::sigma_z<cpx>(state, Ll, Ll - 1 );
			return std::make_pair(state, val1);
			};
		auto _operator = QOps::genOp(this->L, std::move(kernel), 1.0);
		arma::sp_mat op = arma::real(_operator.to_matrix(dim));
		arma::Mat<element_type> mat_elem = V.t() * op * V;
		auto [_agp, _typ_susc, _susc, tmp] = adiabatics::gauge_potential(mat_elem, E, this->L);
		agp_norm_r(site_pairs.size()) = _agp;
		typ_susc_r(site_pairs.size()) = _typ_susc;
		arma::vec diag_mat_elem_Sz_r = arma::diagvec(mat_elem);
		auto [omegas_i, matter] = set_omega.get_matrix_elements(mat_elem);

		std::cout << " - - - - - - finished Sz_L matrix elements in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
	// 	start = std::chrono::system_clock::now();
	// 	// arma::vec quench_FM_Sz(times.size(), arma::fill::zeros);
	// 	arma::vec quench_AFM_Sz(times.size(), arma::fill::zeros);
	// 	// arma::vec quench_spiral_Sz(times.size(), arma::fill::zeros);
	// 	arma::vec quench_random_Sz(times.size(), arma::fill::zeros);
	// #pragma omp parallel for
	// 	for(long t_idx = 0; t_idx < times.size(); t_idx++)
	// 	{
	// 		// quench_FM_Sz(t_idx) = std::real(arma::cdot(psi_FM.col(t_idx), op * psi_FM.col(t_idx)));
	// 		quench_AFM_Sz(t_idx) = std::real(arma::cdot(psi_AFM.col(t_idx), op * psi_AFM.col(t_idx)));
	// 		// quench_spiral_Sz(t_idx) = std::real(arma::cdot(psi_spiral.col(t_idx), op * psi_spiral.col(t_idx)));
	// 		quench_random_Sz(t_idx) = std::real(arma::cdot(psi_random.col(t_idx), op * psi_random.col(t_idx)));
	// 	}
	// 	std::cout << " - - - - - - finished time evolution for Sz_L in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
	// 	start = std::chrono::system_clock::now();
	// 	auto [autocorr_Sz, LTA_Sz] = spectrals::autocorrelation_function(mat_elem, E, times);
	// 	LTA_r(site_pairs.size()) = LTA_Sz;
	// 	std::cout << " - - - - - - finished auto correlator time evolution for Sz_L in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		// #ifndef MY_MAC
		{
			agp_norm_r.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "agp",   arma::hdf5_opts::append));
			typ_susc_r.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "typ_susc",   arma::hdf5_opts::append));
			diag_mat_elem_r.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "diag_mat",   arma::hdf5_opts::append));
			omegas.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "omegas",   arma::hdf5_opts::append));
			spectral_funs.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "spectral_funs",   arma::hdf5_opts::append));

			matter.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "spectral_Sz_L",   arma::hdf5_opts::append));
			diag_mat_elem_Sz_r.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "diag_mat_Sz_L",   arma::hdf5_opts::append));
			// quench_FM_Sz.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench_FM_Sz_L",   arma::hdf5_opts::append));
			// quench_AFM_Sz.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench_AFM_Sz_L",   arma::hdf5_opts::append));
			// // quench_spiral_Sz.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench_spiral_Sz_L",   arma::hdf5_opts::append));
			// quench_random_Sz.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench_random_Sz_L",   arma::hdf5_opts::append));
			// autocorr_Sz.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "autocorr_Sz_L",   arma::hdf5_opts::append));

			// times.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "times",   arma::hdf5_opts::append));
			// // quench_FM.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench_FM",   arma::hdf5_opts::append));
			// quench_AFM.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench_AFM",   arma::hdf5_opts::append));
			// // quench_spiral.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench_spiral",   arma::hdf5_opts::append));
			// quench_random.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench_random",   arma::hdf5_opts::append));
			// autocorr.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "autocorr",   arma::hdf5_opts::append));
			// LTA_r.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "LTA",   arma::hdf5_opts::append));

			// quench_E.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench_energy",   arma::hdf5_opts::append));
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
	arma::vec omegax = arma::logspace(int(std::log10(0.1/dim)), int(std::log10( 5 + this->L )), 20 * this->L);
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

		start = std::chrono::system_clock::now();
		// arma::Mat<element_type> mat_elem = V * Sz_ops[i] * V.t();
		auto kernel = [Ll, N](u64 state){ 
			auto [val1, tmp22] = operators::sigma_z<cpx>(state, Ll, Ll - 1 );
			return std::make_pair(state, val1);
			};
		auto _operator = QOps::genOp(this->L, std::move(kernel), 1.0);
		arma::sp_mat opmat = arma::real(_operator.to_matrix(dim));
		arma::Mat<element_type> mat_elem = V.t() * opmat * V;
		std::cout << " - - - - - - finished matrix elements in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		
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
			omegax.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "omegas",   arma::hdf5_opts::append));
			energy_density.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "energy_density",   arma::hdf5_opts::append));
			_spectral_fun.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "spectral_fun",   arma::hdf5_opts::append));
			_spectral_fun_typ.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "log(_spectral_fun_typ)",   arma::hdf5_opts::append));
			_element_count.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "element_count",   arma::hdf5_opts::append));
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
	
	#ifdef MY_MAC
		energies /= double(counter);
		spectral_fun = spectral_fun / element_count;
		spectral_fun_typ = arma::exp(spectral_fun_typ / element_count);

		energies.save(		arma::hdf5_name(dir + info + ".hdf5", "energies"));
		energy_density.save(   arma::hdf5_name(dir + info + ".hdf5", "energy_density",   arma::hdf5_opts::append));
		spectral_fun.save(	arma::hdf5_name(dir + info + ".hdf5", "spectral_fun",   arma::hdf5_opts::append));
		spectral_fun_typ.save(	arma::hdf5_name(dir + info + ".hdf5", "spectral_fun_typ",   arma::hdf5_opts::append));
		element_count.save(	arma::hdf5_name(dir + info + ".hdf5", "element_count",   arma::hdf5_opts::append));
		omegax.save(   		arma::hdf5_name(dir + info + ".hdf5", "omegax",   arma::hdf5_opts::append));
		arma::vec({(double)counter}).save(	arma::hdf5_name(dir + info + ".hdf5", "realisations",   arma::hdf5_opts::append));
	#endif
}

/// @brief Calculate matrix elements of local operators
void ui::quench()
{
	std::string dir = this->saving_dir + "Quench" + kPSep;
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info();

	const size_t size = dim > 1e5? this->l_steps : dim;

	const int Lhalf = this->L / 2;

	const double chi = 0.341345;
	const double wH = std::sqrt(this->L) / (chi * dim);
	double tH = 1. / wH;
	double r1 = 0.0, r2 = 0.0;
	int time_end = (int)std::ceil(std::log10(50 * tH));
	time_end = (time_end / std::log10(tH) < 20 ) ? time_end + 2 : time_end;

	arma::vec times = arma::logspace(-2, time_end, this->num_of_points);

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
		E.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "energies"));
		
		arma::vec Hdiagonal = arma::diagvec( this->ptr_to_model->get_dense_hamiltonian() );

		double E_av = arma::trace(E) / double(dim);
		auto i = min_element(begin(Hdiagonal), end(Hdiagonal), [=](double x, double y) {
			return abs(x - E_av) < abs(y - E_av);
		});
		const u64 idx = i - begin(Hdiagonal);
		double quench_E = Hdiagonal(idx);

		arma::vec coeff = V.row(idx).t();
		coeff.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "coefficients"));

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
		auto kernel = [Ll, N](u64 state){ 
			auto [val1, tmp22] = operators::sigma_z<cpx>(state, Ll, Ll - 1 );
			return std::make_pair(state, val1);
			};
		auto _operator = QOps::genOp(this->L, std::move(kernel), 1.0);
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

void ui::quench_fourier()
{
	std::string dir = this->saving_dir + "Quench" + kPSep + "Fourier" + kPSep;
	
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info();

	const size_t size = dim > 1e5? this->l_steps : dim;

	const int Lhalf = this->L / 2;

	double r1 = 0.0, r2 = 0.0;

	double bandwidth = this->L;
	double sigma = 0.5 * std::sqrt(this->L);
	double dt = constants<double>::two_pi / bandwidth;
	double tH = constants<double>::two_pi * dim / sigma;
	double tmin = tH - this->num_of_points / 2 * dt;
	if( tmin < 0 ) tmin = tH / 100;
	arma::vec times = tmin + arma::linspace(0, this->num_of_points / 2 * dt, this->num_of_points + 1);

	int Ll = this->L;
	int N = this->grain_size;

	int counter = 0;
	arma::vec omegax = arma::logspace(int(std::log10(0.1/dim)), int(std::log10( 5 + this->L )), 20 * this->L);
	arma::vec energy_density = arma::regspace(0.05, 0.02, 0.95);

	const double dw_log = std::log10(omegax[1]) - std::log10(omegax[0]);
	const double w0_log = std::log10(omegax[0]);
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
		auto kernel = [Ll, N](u64 state){ 
			auto [val1, tmp22] = operators::sigma_z<cpx>(state, Ll, Ll - 1 );
			return std::make_pair(state, val1);
			};
		auto _operator = QOps::genOp(this->L, std::move(kernel), 1.0);
		arma::sp_mat op = arma::real(_operator.to_matrix(dim));
		arma::Mat<element_type> mat_elem = V.t() * op * V;
		arma::vec diag_mat_elem = arma::diagvec(mat_elem);

		std::cout << " - - - - - - finished Sz_L matrix elements in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();
		arma::Mat<element_type> _integrated_spectral_fun(omegax.size()-1, energy_density.size(), arma::fill::zeros);
		arma::Mat<element_type> _spectral_fun(omegax.size()-1, energy_density.size(), arma::fill::zeros);
		arma::Col<element_type> _spectral_fun_all(omegax.size()-1, arma::fill::zeros);
		arma::Mat<element_type> _element_count(omegax.size()-1, energy_density.size(), arma::fill::zeros);
		arma::Col<element_type> _element_count_all(omegax.size()-1, arma::fill::zeros);
		
		const double bandwidth = E(E.size() - 1) - E(0);
		for(int n = 0; n < E.size() - 1; n++){
			for(int m = n+1; m < E.size() - 1; m++){
				double wnm = E(m) - E(n);
				const auto idx = int( (std::log10(wnm) - w0_log) / dw_log);
				if(idx < omegax.size()-1 && idx >= 0){
					_spectral_fun_all(idx) += 2 * std::abs(mat_elem(n, m) * mat_elem(m, n));
					_element_count_all(idx) += 2;
				}
			}	
		}
		std::cout << " - - - - - - finished Sz_L matrix elements at all energy density in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();
		
	#pragma omp parallel for
		for(int ii = 0; ii < energy_density.size(); ii++)
		{
			const double eps = energy_density(ii);
			const double energyx = eps * bandwidth + E(0);
			for(int n = 0; n < E.size(); n++)
			{
				for(int m = n+1; m < E.size(); m++){
					if (std::abs((E(n) + E(m)) / 2. - energyx) < window_width / 2.){
						double wnm = std::abs(E(m) - E(n));
						const auto idx = int( (std::log10(wnm) - w0_log) / dw_log);
						if(idx < omegax.size()-1 && idx >= 0){
							_spectral_fun(idx, ii) += 2 * std::abs(mat_elem(n, m) * mat_elem(m, n));
							_element_count(idx, ii) += 2;
						}
					}
				}	
			}
		}
		std::cout << " - - - - - - finished Sz_L matrix elements at finite energy density in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();
	#pragma omp parallel for
		for(long t_idx = 0; t_idx < times.size(); t_idx++)
			quench(t_idx) = std::real( arma::cdot(psi.col(t_idx), op * psi.col(t_idx)) );
		
		std::cout << " - - - - - - finished time evolution for Sz_L in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();

		arma::Mat<element_type> K_EOA(omegax.size()-1, energy_density.size(), arma::fill::zeros);
		arma::Mat<element_type> Kelement_count(omegax.size()-1, energy_density.size(), arma::fill::zeros);

		arma::Col<element_type> K_EOA_all(omegax.size()-1, arma::fill::zeros);
		arma::Col<element_type> Kelement_count_all(omegax.size()-1, arma::fill::zeros);
		
		mat_elem = coeff * coeff.t();
		for(int n = 0; n < E.size(); n++){
			for(int m = n+1; m < E.size(); m++){
				double wnm = E(m) - E(n);
				const auto idx = int( (std::log10(wnm) - w0_log) / dw_log);
				if(idx < omegax.size()-1 && idx >= 0){
					K_EOA_all(idx) += 2 * std::abs(mat_elem(n, m) * mat_elem(m, n));
					Kelement_count_all(idx) += 2;
				}
			}	
		}
		std::cout << " - - - - - - finished K_EAO at all energy density for realis = " << realis << " in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();
		
	#pragma omp parallel for
		for(int ii = 0; ii < energy_density.size(); ii++)
		{
			const double eps = energy_density(ii);
			const double energyx = eps * bandwidth + E(0);
			for(int n = 0; n < E.size() - 1; n++)
			{
				for(int m = n+1; m < E.size() - 1; m++){
					if (std::abs((E(n) + E(m)) / 2. - energyx) < window_width / 2.){
						double wnm = std::abs(E(m) - E(n));
						const auto idx = int( (std::log10(wnm) - w0_log) / dw_log);
						if(idx < omegax.size()-1 && idx >= 0){
							K_EOA(idx, ii) += 2 * std::abs(mat_elem(n, m) * mat_elem(m, n));
							Kelement_count(idx, ii) += 2;
						}
					}
				}	
			}
		}
		std::cout << " - - - - - - finished K_EAO at finite energy density for realis = " << realis << " in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();
		// start = std::chrono::system_clock::now();
		// auto [autocorr_Sz, LTA_Sz] = spectrals::autocorrelation_function(mat_elem, E, times);
		// std::cout << " - - - - - - finished auto correlator time evolution for Sz_L in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		// #ifndef MY_MAC
		{
			diag_mat_elem.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "diag_mat",   arma::hdf5_opts::append));
			times.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "times",   arma::hdf5_opts::append));
			quench.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench",   arma::hdf5_opts::append));
			arma::vec( {quench_E} ).save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench_energy",   arma::hdf5_opts::append));
			
			K_EOA.save(	arma::hdf5_name(dir_realis + info + ".hdf5", "K_EOA",   arma::hdf5_opts::append));
			K_EOA_all.save(	arma::hdf5_name(dir_realis + info + ".hdf5", "K_EOA_all",   arma::hdf5_opts::append));
			Kelement_count.save(	arma::hdf5_name(dir_realis + info + ".hdf5", "Kelement_count",   arma::hdf5_opts::append));
			Kelement_count_all.save(	arma::hdf5_name(dir_realis + info + ".hdf5", "Kelement_count_all",   arma::hdf5_opts::append));
			
			_spectral_fun.save(	arma::hdf5_name(dir_realis + info + ".hdf5", "spectral_fun",   arma::hdf5_opts::append));
			_spectral_fun_all.save(	arma::hdf5_name(dir_realis + info + ".hdf5", "spectral_fun_all",   arma::hdf5_opts::append));
			_element_count.save(	arma::hdf5_name(dir_realis + info + ".hdf5", "element_count",   arma::hdf5_opts::append));
			_element_count_all.save(	arma::hdf5_name(dir_realis + info + ".hdf5", "element_count_all",   arma::hdf5_opts::append));
		}
		// #endif
		
		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start_re) << " s - - - - - - " << std::endl; // simulation end
	}
}

void ui::multifractality(){
	clk::time_point start = std::chrono::system_clock::now();

	std::string subdir = "ParticipationRatio" + kPSep;
	std::string dir = this->saving_dir + "NewBasis" + kPSep + subdir;
	createDirs(dir);

	std::string info = this->set_info();
	std::string filename = info;
	size_t dim = this->ptr_to_model->get_hilbert_size();

	int counter = 0;

	const int size = dim;	
	// arma::vec q_ipr_list = arma::linspace(2.0 / double(this->num_of_points), 2.0, this->num_of_points);
	arma::vec q_ipr_list = {0.5, 1.0, 1.5, 2, 3.0};
	double energy_window = 0.01;
	arma::vec energy_density = arma::vec({0.0, 0.0831, 0.1265, 0.1572, 0.1814, 0.2017, 0.2194, 0.235, 0.2493, 0.2623, 0.2744, 0.2857, 0.2964, 0.3065, 0.3162, 0.3254, 0.3343, 0.3429, 0.3512, 0.3592, 0.367, 0.3747, 0.3821, 0.3894, 0.3965, 0.4036, 0.4105, 0.4172, 0.4239, 0.4306, 0.4371, 0.4436, 0.45, 0.4563, 0.4627, 0.4689, 0.4752, 0.4814, 0.4876, 0.4938, 0.5, 0.5062, 0.5124, 0.5186, 0.5248, 0.5311, 0.5373, 0.5437, 0.55, 0.5564, 0.5629, 0.5694, 0.5761, 0.5828, 0.5895, 0.5964, 0.6035, 0.6106, 0.6179, 0.6253, 0.633, 0.6408, 0.6488, 0.6571, 0.6657, 0.6746, 0.6838, 0.6935, 0.7036, 0.7143, 0.7256, 0.7377, 0.7507, 0.765, 0.7806, 0.7983, 0.8186, 0.8428, 0.8735, 0.9169, 1.0	});
	arma::vec energy_density2 = arma::sort(0.5 - arma::logspace(-2, 0, 51));

	disorder<double> disorder_generator = disorder<double>(this->seed);
	disorder<int> neigh_generator = disorder<int>(this->seed);
	GOE grain_generator(this->seed);
	for(int realis = 0; realis < this->realisations; realis++)
	{
		if(realis > 0)
			this->ptr_to_model->generate_hamiltonian();
		
    	clk::time_point start_loop = std::chrono::system_clock::now();
		// this->ptr_to_model->diagonalization();
		const size_t dim_loc = ULLPOW( (this->L_loc) );
		const size_t dim_erg = ULLPOW( (this->grain_size) );

		arma::mat H = arma::mat(dim, dim, arma::fill::zeros);
		arma::mat H0 = arma::mat(dim, dim, arma::fill::zeros);
		arma::mat Sz_L = arma::mat(dim, dim, arma::fill::zeros);
		auto _disorder = disorder_generator.uniform(this->L_loc, this->h - this->w, this->h + this->w);
		std::cout << "AAAAA: " << _disorder.t() << std::endl;
		
		/* Create random neighbours for coupling hamiltonian */
		auto random_neigh = neigh_generator.uniform(this->L_loc, 0, this->grain_size - 1);

		/* Create GOE Matrix */
		arma::mat H_grain = this->gamma * grain_generator.generate_matrix(dim_erg);
		// H_grain = H_grain - arma::trace(H_grain);
		// H_grain /= std::sqrt(ULLPOW(this->grain_size) + 1);
		H_grain /= std::sqrt( arma::trace(H_grain * H_grain) / double(dim_erg) );

		/* Create random couplings */
		auto _long_range_couplings = arma::vec(this->L_loc, arma::fill::zeros);
		if(this->alfa > 0){
			
			if( std::abs(this->alfa - 1) < 1e-10){
				_long_range_couplings = arma::vec(this->L_loc, arma::fill::ones);
			} else {
				double u_j = 1 + disorder_generator.uniform_dist<double>(-this->zeta, this->zeta);
				_long_range_couplings(0) = 1.0;
				for (int j = 1; j < this->L_loc; j++){
					int pos = j;
					double u_j = pos + disorder_generator.uniform_dist<double>(-this->zeta, this->zeta);
					_long_range_couplings(j) = std::pow(this->alfa, u_j);
				}
			}
		}
		_extra_debug(
			std::cout << "disorder: \t\t" << _disorder.t() << std::endl;   
			std::cout << "couplings: \t\t" << _long_range_couplings.t() << std::endl;
			std::cout << "random_neigh: \t\t" << random_neigh.t() << std::endl;
			std::cout << "Grain matrix: \t\t" << H_grain << std::endl;
		)

		/* Generate coupling and spin hamiltonian */
		clk::time_point start = std::chrono::system_clock::now();
		for (u64 k = 0; k < dim; k++) {
			u64 base_state = k;
			for (int j = 0; j < this->L - this->grain_size; j++)  // sum over spin d.o.f
			{
				const int pos_in_array = this->L - 1 - this->grain_size - j;                // array index of localised spin
				/* disorder on localised spins */
				auto [val, Sz_k] = operators::sigma_z<double>(base_state, this->L, j);
				H(k, k) += _disorder(pos_in_array) * (val);
				H0(k, k) += _disorder(pos_in_array) * (val);
				
				if(j == 0){
					Sz_L(k, k) = val;
				}
			
				/* coupling of localised spins to GOE grain */
				int nei = random_neigh(pos_in_array);
				auto [val1, Sx_k] = operators::sigma_x<double>(base_state, this->L, j);
				auto [val2, SxSx_k] = operators::sigma_x<double>(Sx_k, this->L, this->L - this->grain_size + nei);
				double mat_element = this->J * _long_range_couplings(pos_in_array) * (val1 * val2);
				H(SxSx_k, k) += mat_element;
				if(j > 0)
					H0(SxSx_k, k) += mat_element;
			}
		}
		std::cout << " - - - - - - finished Hamiltonian in : " << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		// H = H + arma::kron<arma::mat>(arma::mat(H_grain), arma::mat(arma::eye<arma::mat>(dim_loc, dim_loc)));
		// H0 = H0 + arma::kron<arma::mat>(arma::mat(H_grain), arma::mat(arma::eye<arma::mat>(dim_loc, dim_loc)));
		H = H + arma::kron<arma::mat>(arma::mat(arma::eye<arma::mat>(dim_loc, dim_loc)), arma::mat(H_grain));
		H0 = H0 + arma::kron<arma::mat>(arma::mat(arma::eye<arma::mat>(dim_loc, dim_loc)), arma::mat(H_grain));

		arma::vec E;
		arma::mat V;
		arma::eig_sym(E, V, H);
		

		std::cout << " - - - - - - finished diagonalization in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();
		
		arma::vec E0;
		arma::mat V0;
		arma::eig_sym(E0, V0, H0);
		double dE0 = E0(E0.size()-1) - E0(0);


		// u64 E_av_idx = spectrals::get_mean_energy_index(E);
		double E_av = 0.5 * dE0 + E0(0);
			
		auto i = min_element(begin(E), end(E), [=](double x, double y) {
				return abs(x - E_av) < abs(y - E_av);
				});
		u64 E_av_idx = i - E.begin();

		u64 num_of_states = std::min( u64(this->l_steps), u64(0.02*dim) );
		u64	Emin = E_av_idx - num_of_states / 2;
		u64	Emax = E_av_idx + num_of_states / 2;

		std::cout << " - - - - - - finished diagonalization of L-1 sized matrix in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end

		H.save(   arma::hdf5_name("HamiltonianQSM.hdf5", "H"));
		H0.save(   arma::hdf5_name("HamiltonianQSM.hdf5", "H0", arma::hdf5_opts::append));
		E.save(   arma::hdf5_name("HamiltonianQSM.hdf5", "E", arma::hdf5_opts::append));
		E0.save(   arma::hdf5_name("HamiltonianQSM.hdf5", "E0", arma::hdf5_opts::append));
		V.save(   arma::hdf5_name("HamiltonianQSM.hdf5", "V", arma::hdf5_opts::append));
		V0.save(   arma::hdf5_name("HamiltonianQSM.hdf5", "V0", arma::hdf5_opts::append));
		arma::mat H_in_H0 = V0.t() * H * V0;
		arma::vec Sz_L_of_H0 = arma::diagvec( V0.t() * Sz_L * V0 );
		H_in_H0.save(   arma::hdf5_name("HamiltonianQSM.hdf5", "H_in_H0", arma::hdf5_opts::append));
		Sz_L_of_H0.save(   arma::hdf5_name("HamiltonianQSM.hdf5", "Sz_L_of_H0", arma::hdf5_opts::append));
		start = std::chrono::system_clock::now();

		arma::mat part_ratio(num_of_states, q_ipr_list.size(), arma::fill::zeros);
		arma::mat info_ent(num_of_states, q_ipr_list.size(), arma::fill::zeros);
		
		arma::vec part_ratio_d2(size, arma::fill::zeros);
		arma::vec info_ent_d2(size, arma::fill::zeros);
		arma::vec part_ratio_d2_comp(size, arma::fill::zeros);
		arma::vec info_ent_d2_comp(size, arma::fill::zeros);
		arma::mat ldos(num_of_states, energy_density.size()-1, arma::fill::zeros);
		arma::mat ldos2(num_of_states, energy_density2.size()-1, arma::fill::zeros);

		outer_threads = this->thread_number;
		omp_set_num_threads(1);

		for(int iq = 0; iq < q_ipr_list.size(); iq++)
		{
		#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
			for(int n = 0; n < num_of_states; n++)
			{
				arma::Col<element_type> eigenstate = arma::normalise(V.col(n + Emin));
				if(q_ipr_list(iq) == 1)
				{
					double _pr_ = 0;
					for (int k = 0; k < eigenstate.size(); k++) {
						arma::vec state_k = arma::normalise(V0.col(k));
						auto c_k = dot_prod( state_k, eigenstate);
						double value = std::abs(std::conj(c_k) * c_k);
						_pr_ += (std::abs(value) > 0) ? -value * std::log(value) : 0;
					}
					part_ratio(n, iq) = arma::norm(eigenstate);
					info_ent(n, iq) = _pr_;
				}
				else{
					double _pr_ = statistics::participation_ratio(eigenstate, V0, q_ipr_list(iq));
					part_ratio(n, iq) = _pr_;
					info_ent(n, iq) = std::log(_pr_) / (1 - q_ipr_list(iq));
				}
			}
		}
		std::cout << " - - - - - - finished IPR in spectrum center in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();

	#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
		for(int n = 0; n < dim; n++)
		{
			arma::Col<element_type> eigenstate = arma::normalise(V.col(n));
			double _pr_ = statistics::participation_ratio(eigenstate, V0, 2);
			part_ratio_d2(n) = _pr_;
			info_ent_d2(n) = -std::log(_pr_);

			_pr_ = statistics::participation_ratio(eigenstate, 2);
			part_ratio_d2_comp(n) = _pr_;
			info_ent_d2_comp(n) = -std::log(_pr_);

			//!------- LDOS CALCULATION
			if(n >= Emin && n < Emax){
				// const auto idx = int( (std::log10(E0(n)) - E0(0)) / energy_window);
				arma::vec overlaps = V0.t() * eigenstate;
				for(int e = 0; e < energy_density.size()-1; e++)
				{
					double E_minus = energy_density(e) * dE0 + E0(0);
					double E_plus = energy_density(e+1) * dE0 + E0(0);
					arma::uvec indices = arma::find(E0 >= E_minus && E0 < E_plus);
					ldos(n-Emin, e) = arma::accu( arma::square(overlaps.rows(indices)) );
				}
				for(int e = 0; e < energy_density2.size()-1; e++)
				{
					double E_minus = energy_density2(e) * dE0 + E0(0);
					double E_plus = energy_density2(e+1) * dE0 + E0(0);
					arma::uvec indices = arma::find(E0 >= E_minus && E0 < E_plus);
					ldos2(n-Emin, e) = arma::accu( arma::square(overlaps.rows(indices)) );
				}
			}
		}
		std::cout << " - - - - - - finished IPR all for q=2 in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end

		omp_set_num_threads(this->thread_number);

		std::string dir_realis = dir + "realisation=" + std::to_string(realis + this->jobid) + kPSep;
		createDirs(dir_realis);
		q_ipr_list.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "qs"));

		ldos.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "LDOS", arma::hdf5_opts::append));
		energy_density.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "energy_density", arma::hdf5_opts::append));
		ldos2.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "LDOS2", arma::hdf5_opts::append));
		energy_density2.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "energy_density2", arma::hdf5_opts::append));

		E.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "energies", arma::hdf5_opts::append));
		E0.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "unperturbed energies", arma::hdf5_opts::append));
		part_ratio.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "pr", arma::hdf5_opts::append));
		info_ent.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "info", arma::hdf5_opts::append));
		part_ratio_d2.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "pr_d2", arma::hdf5_opts::append));
		info_ent_d2.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "info_d2", arma::hdf5_opts::append));
		part_ratio_d2_comp.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "pr_d2_comp", arma::hdf5_opts::append));
		info_ent_d2_comp.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "info_d2_comp", arma::hdf5_opts::append));
		
		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start_loop) << " s - - - - - - " << std::endl; // simulation end
	};
    std::cout << " - - - - - - FINISHED IPR CALCULATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
}


void ui::geometric_tensor(){
	std::string dir = this->saving_dir + "GeometricTensor" + kPSep;
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info();

	const size_t size = dim > 1e5? this->l_steps : dim;

	arma::vec energies(size, arma::fill::zeros);

	int Ll = this->L;
	int counter = 0;
	
	arma::vec omegax = arma::logspace(int(std::log10(0.1/dim)), int(std::log10( 5 + this->L )), 20 * this->L);
	const arma::vec energy_density = arma::regspace(0.05, 0.02, 0.95);

	arma::Mat<element_type> spectral_fun(omegax.size()-1, energy_density.size(), arma::fill::zeros);
	arma::Mat<element_type> spectral_fun_typ(omegax.size()-1, energy_density.size(), arma::fill::zeros);
	arma::Mat<element_type> element_count(omegax.size()-1, energy_density.size(), arma::fill::zeros);
	
	double window_width = 0.04;

	int N = this->grain_size;
	auto neighbor_generator = disorder<int>(this->seed);
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

		
		long int E_min = dim < 0? 0 : Eav_idx - long(dim / 4);
		long int E_max = dim > 1e5? dim : Eav_idx + long(dim / 4);

		double cutoff = std::sqrt(this->L) / double(dim);

		std::vector<arma::Mat<element_type>> mat_elements;
		start = std::chrono::system_clock::now();
		// arma::Mat<element_type> mat_elem = V * Sz_ops[i] * V.t();
		auto kernel = [Ll](u64 state) -> std::pair<u64, double>
			{ 
			auto [val1, tmp22] = operators::sigma_z<double>(state, Ll, Ll - 1 );
			return std::make_pair(state, val1);
			};
		auto _operator = QOps::generic_operator<double>(this->L, std::move(kernel), 1.0);
		arma::sp_mat opmat = _operator.to_matrix(dim);
		// arma::mat opmat = arma::diagmat( arma::vec(dim, arma::fill::randn) );
		double _operator_HSnorm = arma::trace(opmat * opmat) / double(dim);
		opmat /= std::sqrt(_operator_HSnorm);
		mat_elements.push_back( V.t() * opmat * V );

		auto kernel_SxSx = [Ll, N, &neighbor_generator](u64 state) -> std::pair<u64, double>
		{ 
			int nei = neighbor_generator.uniform_dist<int>(0, N-1);
			auto [val1, Sx] = operators::sigma_x<double>(state, Ll, Ll - 1 );
			auto [val2, SxSx] = operators::sigma_x<double>(Sx, Ll, nei );
			return std::make_pair(SxSx, val1 * val2);
		};
		_operator = QOps::generic_operator<double>(this->L, std::move(kernel_SxSx), 1.0);
		opmat = _operator.to_matrix(dim);
		// ENSEMBLE random_matrix;
		// opmat = random_matrix.generate_matrix(dim);
		_operator_HSnorm = arma::trace(opmat * opmat) / double(dim);
		opmat /= std::sqrt(_operator_HSnorm);
		mat_elements.push_back( V.t() * opmat * V );
		
		std::cout << " - - - - - - finished matrix elements in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();

		// auto [_Z, _count, _count_proj,AGP_T, AGP_T_reg, AGP_E, AGP_E_proj] = adiabatics::gauge_potential_finite_T(mat_elem, E, betas, energy_density);
		auto [_susc, _susc_r] = adiabatics::gauge_potential_save(mat_elements[0], E, this->L, cutoff);
		auto [_susc2, _susc_r2] = adiabatics::gauge_potential_save(mat_elements[1], E, this->L, cutoff);
		
		#if _MAT_ENSEMBLE_ == 0
			arma::mat geometric_tensor = arma::real(adiabatics::geometric_tensor(mat_elements, E, this->L, cutoff));
			arma::vec chi; arma::mat chi_states;
			arma::eig_sym(chi, chi_states, geometric_tensor);
		#else
			auto geometric_tensor = adiabatics::geometric_tensor(mat_elements, E, this->L, cutoff);
			arma::vec chi; arma::cx_mat chi_states;
			arma::eig_sym(chi, chi_states, geometric_tensor);
		#endif

		std::cout << " - - - - - - finished AGP in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end

		std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
		createDirs(dir_realis);
		E.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "energies"));
		omegax.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "omegas",   arma::hdf5_opts::append));
		energy_density.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "energy_density",   arma::hdf5_opts::append));

		_susc.save(	 arma::hdf5_name(dir_realis + info + ".hdf5", "Sz/susc",     arma::hdf5_opts::append));
		_susc_r.save(arma::hdf5_name(dir_realis + info + ".hdf5", "Sz/susc_reg", arma::hdf5_opts::append));
		_susc2.save(  arma::hdf5_name(dir_realis + info + ".hdf5", "SxSx/susc",     arma::hdf5_opts::append));
		_susc_r2.save(arma::hdf5_name(dir_realis + info + ".hdf5", "SxSx/susc_reg", arma::hdf5_opts::append));
		geometric_tensor.save(arma::hdf5_name(dir_realis + info + ".hdf5", "geometric_tensor", arma::hdf5_opts::append));
		chi.save(arma::hdf5_name(dir_realis + info + ".hdf5", "susceptibilities", arma::hdf5_opts::append));
		chi_states.save(arma::hdf5_name(dir_realis + info + ".hdf5", "chi states", arma::hdf5_opts::append));
		double phi_min = std::acos( arma::cdot(chi_states.col(0), arma::vec({0,1})) );
		arma::vec({phi_min}).save(arma::hdf5_name(dir_realis + info + ".hdf5", "phi_min", arma::hdf5_opts::append));
		double phi_max = std::acos( arma::cdot(chi_states.col(1), arma::vec({0,1})) );
		arma::vec({phi_max}).save(arma::hdf5_name(dir_realis + info + ".hdf5", "phi_max", arma::hdf5_opts::append));
		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start_re) << " s - - - - - - " << std::endl; // simulation end
	}
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
        if(std::abs(this->alfa - 1.0) > 1e-10) name += ",zeta=" + to_string_prec(this->zeta);
        
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




		// disorder<double> disorder_generator = disorder<double>(this->seed);
		// disorder<int> neigh_generator = disorder<int>(this->seed);
		// GOE grain_generator(this->seed);
		// const size_t dim_loc = ULLPOW( (this->L_loc) );
		// const size_t dim_erg = ULLPOW( (this->grain_size) );

		// arma::sp_mat H = arma::sp_mat(dim, dim);
		// auto _disorder = disorder_generator.uniform(this->L_loc, this->h - this->w, this->h + this->w);
		// std::cout << "AAAAA: " << _disorder.t() << std::endl;
		
		// /* Create random neighbours for coupling hamiltonian */
		// auto random_neigh = neigh_generator.uniform(this->L_loc, 0, this->grain_size - 1);

		// /* Create GOE Matrix */
		// arma::mat H_grain = this->gamma * grain_generator.generate_matrix(dim_erg);
		// // H_grain = H_grain - arma::trace(H_grain);
		// H_grain /= std::sqrt(ULLPOW(this->grain_size) + 1);
		// // H_grain /= std::sqrt( arma::trace(H_grain * H_grain) / double(dim_erg) );

		// /* Create random couplings */
		// auto _long_range_couplings = arma::vec(this->L_loc, arma::fill::zeros);
		// if(this->alfa > 0){
			
		// 	if( std::abs(this->alfa - 1) < 1e-10){
		// 		_long_range_couplings = arma::vec(this->L_loc, arma::fill::ones);
		// 	} else {
		// 		// double u_j = 1 + disorder_generator.uniform_dist<double>(-this->zeta, this->zeta);
		// 		_long_range_couplings(0) = 1.0;
		// 		for (int j = 1; j < this->L_loc; j++){
		// 			int pos = j;
		// 			double u_j = pos + disorder_generator.uniform_dist<double>(-this->zeta, this->zeta);
		// 			_long_range_couplings(j) = std::pow(this->alfa, u_j);
		// 			printSeparated(std::cout, "\t", 20, true, pos, u_j);
		// 		}
		// 	}
		// }
		// _extra_debug(
		// 	std::cout << "disorder: \t\t" << _disorder.t() << std::endl;   
		// 	std::cout << "couplings: \t\t" << _long_range_couplings.t() << std::endl;
		// 	std::cout << "random_neigh: \t\t" << random_neigh.t() << std::endl;
		// 	std::cout << "Grain matrix: \t\t" << H_grain << std::endl;
		// )

		// /* Generate coupling and spin hamiltonian */
		// clk::time_point start = std::chrono::system_clock::now();
		// for (u64 k = 0; k < dim; k++) {
		// 	u64 base_state = k;
		// 	for (int j = 0; j < this->L - this->grain_size; j++)  // sum over spin d.o.f
		// 	{
		// 		const int pos_in_array = this->L - 1 - this->grain_size - j;                // array index of localised spin
		// 		/* disorder on localised spins */
		// 		auto [val, Sz_k] = operators::sigma_z<double>(base_state, this->L, j);
		// 		H(k, k) += _disorder(pos_in_array) * (val);
			
		// 		/* coupling of localised spins to GOE grain */
		// 		int nei = random_neigh(pos_in_array);
		// 		auto [val1, Sx_k] = operators::sigma_x<double>(base_state, this->L, j);
		// 		auto [val2, SxSx_k] = operators::sigma_x<double>(Sx_k, this->L, this->L - this->grain_size + nei);
		// 		double mat_element = this->J * _long_range_couplings(pos_in_array) * (val1 * val2);
		// 		H(SxSx_k, k) += mat_element;
		// 	}
		// }
		// std::cout << " - - - - - - finished Hamiltonian in : " << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		// // H = H + arma::kron<arma::mat>(arma::mat(H_grain), arma::mat(arma::eye<arma::mat>(dim_loc, dim_loc)));
		// // H0 = H0 + arma::kron<arma::mat>(arma::mat(H_grain), arma::mat(arma::eye<arma::mat>(dim_loc, dim_loc)));
		
		// H = H + arma::kron<arma::sp_mat>(arma::sp_mat(arma::eye<arma::sp_mat>(dim_loc, dim_loc)), arma::sp_mat(H_grain));
