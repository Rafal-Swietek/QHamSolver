#include "includes/QSunUI.hpp"

int outer_threads = 1;
int num_of_threads = 1;

bool normalize_grain = 1;

namespace QSunUI{

void ui::make_sim(){
    printAllOptions();
	
	clk::time_point start = std::chrono::system_clock::now();

	this->ptr_to_model = this->create_new_model_pointer();
	
	

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
	// auto polfed = polfed::POLFED<ui::element_type>(Hamil, this->l_steps, this->l_bundle, -1, this->tol, 0.2, this->seed, true);
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
								agp(); continue;

	const int Ll = this->L;

	auto disorder_generator = disorder<double>(this->seed);
	this->ptr_to_model.reset(new QHS::QHamSolver<QuantumSun>(this->L_loc, this->J, this->alfa, this->gamma, 0, 0, 
																	this->seed, this->grain_size, this->zeta, this->initiate_avalanche, normalize_grain)); 
	
	u64 dim = this->ptr_to_model->get_hilbert_size();

	auto kernel = [Ll](u64 state){ 
			auto [val1, tmp22] = operators::sigma_z(state, Ll, Ll - 1 );
			return std::make_pair(state, val1);
			};
	auto _operator = QOps::generic_operator<>(this->L, std::move(kernel), 1.0);
	arma::sp_mat Sz = arma::real(_operator.to_matrix(dim));
	
	arma::mat H = this->ptr_to_model->get_dense_hamiltonian();
	arma::vec disorder_base = disorder_generator.uniform(this->L_loc, this->h - this->w, this->h + this->w);
	for (u64 k = 0; k < dim; k++) {
		for (int j = this->grain_size; j < this->L; j++)  // sum over spin d.o.f
		{
			const int pos_in_array = j - this->grain_size;                // array index of localised spin
			auto [val, Sz_k] = operators::sigma_z(k, this->L, j);
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
				auto [val, Sz_k] = operators::sigma_z(base_state, this->L, ell);
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
		auto _operator = QOps::generic_operator<>();
		switch(this->op){
			case 0:
			{
				auto kernel_def = [Ll, N](u64 state){ 
					auto [val1, tmp22] = operators::sigma_z(state, Ll, Ll - 1 );
					return std::make_pair(state, val1);
					};
				_operator = QOps::generic_operator<>(this->L, std::move(kernel_def), 1.0);
			}
			case 1:
			{
					auto kernel = [Ll, N, &neighbor_generator](u64 state){ 
						int nei = neighbor_generator.uniform_dist<int>(0, N-1);
						auto [val1, state_out1] = operators::sigma_x(state, Ll, nei );
						auto [val2, state_out2] = operators::sigma_x(state_out1, Ll, Ll - 1 );
						return std::make_pair(state_out2, val1 * val2);
					};
					_operator = QOps::generic_operator<>(this->L, std::move(kernel), 1.0);
			}
				break;
			case 2:
			{
				auto kernel = [Ll, N, &neighbor_generator](u64 state){ 
						int nei = neighbor_generator.uniform_dist<int>(0, N-1);
						auto [val1, state_out1] = operators::sigma_z(state, Ll, nei );
						auto [val2, state_out2] = operators::sigma_z(state_out1, Ll, Ll - 1 );
						return std::make_pair(state_out2, val1 * val2);
				};
				_operator = QOps::generic_operator<>(this->L, std::move(kernel), 1.0);
			}
				break;
			case 3:
			{
				auto kernel = [Ll, N](u64 state){ 
					auto [val1, state_out1] = operators::sigma_z(state, Ll, Ll - 2 );
					auto [val2, state_out2] = operators::sigma_z(state_out1, Ll, Ll - 1 );
					return std::make_pair(state_out2, val1 * val2);
				};
				_operator = QOps::generic_operator<>(this->L, std::move(kernel), 1.0);
			}
				break;
			case 4:
			{
				auto kernel = [Ll, N](u64 state){ 
					auto [val1, state_out1] = operators::sigma_x(state, Ll, Ll - 2 );
					auto [val2, state_out2] = operators::sigma_x(state_out1, Ll, Ll - 1 );
					return std::make_pair(state_out2, val1 * val2);
				};
				_operator = QOps::generic_operator<>(this->L, std::move(kernel), 1.0);
			}
				break;
			default:
			{
				auto kernel_def = [Ll, N](u64 state){ 
					auto [val1, tmp22] = operators::sigma_z(state, Ll, Ll - 1 );
					return std::make_pair(state, val1);
					};
				_operator = QOps::generic_operator<>(this->L, std::move(kernel_def), 1.0);
			}
		}
		arma::sp_mat oper(dim, dim);
		if(this->op == 5){
			for(int j = N; j < this->L; j++){
				auto kernel_def = [Ll, j](u64 state){ 
					auto [val1, tmp22] = operators::sigma_z(state, Ll, j );
					return std::make_pair(state, val1);
					};
				_operator = QOps::generic_operator<>(this->L, std::move(kernel_def), 1.0);
				oper += arma::real(_operator.to_matrix(dim));
			}
			oper = oper / std::sqrt(this->L_loc);
		} else {
			oper = arma::real(_operator.to_matrix(dim));
		}
		
		arma::Mat<element_type> mat_elem = V.t() * oper * V;
		mat_elem.save(	  		arma::hdf5_name(dir_realis + info + ".hdf5", "matelem",   arma::hdf5_opts::append));
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
					auto [val1, tmp22] = operators::sigma_z(state, Ll, Ll - 1 );
					return std::make_pair(state, val1);
					};
		auto _operator = QOps::generic_operator<>(this->L, std::move(kernel_def), 1.0);
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
		susc.save(    arma::hdf5_name(dir + info + ".hdf5", "susc"));
		susc_r.save(  arma::hdf5_name(dir + info + ".hdf5", "susc_reg"));
	#endif
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