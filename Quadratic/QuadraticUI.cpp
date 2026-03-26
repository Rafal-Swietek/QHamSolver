#include "includes/QuadraticUI.hpp"

int outer_threads = 1;
int num_of_threads = 1;


// #include "../include/hilbert_space/symmetries.hpp"
// #include "../include/hilbert_space/constrained.hpp"

namespace QuadraticUI{

void ui::make_sim(){
    printAllOptions();
	clk::time_point start = std::chrono::system_clock::now();
	this->ptr_to_model = this->create_new_model_pointer();
	size_t dim = this->ptr_to_model->get_hilbert_size();
	
	// arma::vec gammas = arma::linspace(0.0, 2.5, 26);
	// arma::vec H_trace(gammas.size(), arma::fill::zeros);
	// arma::vec H_trace2(gammas.size(), arma::fill::zeros);
	// arma::vec av(gammas.size(), arma::fill::zeros);
	// arma::vec var(gammas.size(), arma::fill::zeros);
	// arma::vec av2(gammas.size(), arma::fill::zeros);
	// arma::vec var2(gammas.size(), arma::fill::zeros);
	// for(int r = 0; r < this->realisations; r++)
	// {
	// 	start = std::chrono::system_clock::now();
	// 	for(int iig = 0; iig < gammas.size(); iig++)
	// 	{
	// 		this->seed = std::random_device{}();
	// 		this->g = gammas(iig);
	// 		this->reset_model_pointer();
	// 		arma::sp_mat H = this->ptr_to_model->get_hamiltonian();
	// 		arma::sp_mat H2 = H*H;
	// 		double meanH = arma::trace(H) / double(dim);
	// 		double varH = arma::trace(H2) / double(dim) - meanH * meanH;
	// 		H_trace(iig) += meanH;
	// 		H_trace2(iig) += varH;
			
	// 		arma::vec gec(dim, arma::fill::zeros);
	// 	#pragma omp parallel for
	// 		for(u64 k = 0; k < dim; k++)
	// 			gec(k) = 2 * H2(k,k) - H(k,k) * H(k,k);   
	// 		av2(iig) += arma::mean(gec);
	// 		var2(iig) += arma::var(gec);

	// 		H = H - meanH * arma::eye<arma::sp_mat>(dim, dim);
	// 		H2 = H*H;
	// 		gec = arma::vec(dim, arma::fill::zeros);
	// 	#pragma omp parallel for
	// 		for(u64 k = 0; k < dim; k++)
	// 			gec(k) = 2 * H2(k,k) - H(k,k) * H(k,k); 
	// 		gec = gec / varH;
	// 		av(iig) += arma::mean(gec);
	// 		var(iig) += arma::var(gec);
	// 	}
	// 	std::cout << " - - - - - - finished realization r=" << r << "\t in :" << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
	// }
	// av /= double(this->realisations);
	// av2 /= double(this->realisations);
	// var /= double(this->realisations);
	// var2 /= double(this->realisations);
	// H_trace /= double(this->realisations);
	// H_trace2 /= double(this->realisations);

	// av2 = av2 / H_trace2;
	// var2 = var2 / arma::square(H_trace2);
	
	// std::string dir = "GEC_data/" + kPSep;
	// createDirs(dir);
	// std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid) + kPSep;
	// createDirs(dir_realis);
	// gammas.save(	  arma::hdf5_name(dir_realis + "GEC_L=" + std::to_string(this->L) + ".hdf5", "gammas"));
	// av.save(	  arma::hdf5_name(dir_realis + "GEC_L=" + std::to_string(this->L) + ".hdf5", "av", arma::hdf5_opts::append));
	// var.save(	  arma::hdf5_name(dir_realis + "GEC_L=" + std::to_string(this->L) + ".hdf5", "var", arma::hdf5_opts::append));
	// av2.save(	  arma::hdf5_name(dir_realis + "GEC_L=" + std::to_string(this->L) + ".hdf5", "av2", arma::hdf5_opts::append));
	// var2.save(	  arma::hdf5_name(dir_realis + "GEC_L=" + std::to_string(this->L) + ".hdf5", "var2", arma::hdf5_opts::append));
	// H_trace.save(	  arma::hdf5_name(dir_realis + "GEC_L=" + std::to_string(this->L) + ".hdf5", "trace_H", arma::hdf5_opts::append));
	// H_trace2.save(	  arma::hdf5_name(dir_realis + "GEC_L=" + std::to_string(this->L) + ".hdf5", "trace_H2", arma::hdf5_opts::append));
	// return;
	// arma::vec gammas = arma::linspace(0.0, 2.5, 26);
	// arma::vec max(gammas.size());
	// arma::vec av(gammas.size());
	// arma::vec var(gammas.size());
	// arma::vec kurtosis(gammas.size());
	// arma::vec binder(gammas.size());
	// for(int r = 0; r < this->realisations; r++){
	// 	for(int iig = 0; iig < gammas.size(); iig++){
	// 		this->g = gammas(iig);
	// 		this->reset_model_pointer();
	// 		const auto& H = this->ptr_to_model->get_hamiltonian();
	// 		arma::sp_mat H2 = H*H;

	// 		arma::vec gec(dim, arma::fill::zeros);
	// 		for(u64 k = 0; k < dim; k++)
	// 			gec(k) = 2 * H2(k,k) - H(k,k) * H(k,k);   
	// 		gec = dim * gec / arma::trace(H2);
	// 		av(iig) += arma::mean(gec);
	// 		var(iig) += arma::mean( arma::square(gec) );
	// 		max(iig) += arma::max(gec);
	// 		kurtosis(iig) += arma::mean(arma::square(arma::square(gec - arma::mean(gec)))) / (var(iig) * var(iig));
	// 		binder(iig) += 1 - kurtosis(iig) / 3.;
	// 	}
	// 	av /= double(this->realisations);
	// 	var = var / double(this->realisations) - arma::square(av);
	// 	max /= double(this->realisations);
	// 	binder /= double(this->realisations);
	// 	kurtosis /= double(this->realisations);
	// }
	// gammas.save(	  arma::hdf5_name("GEC_L=" + std::to_string(this->L) + ".hdf5", "gammas"));
	// av.save(	  arma::hdf5_name("GEC_L=" + std::to_string(this->L) + ".hdf5", "av", arma::hdf5_opts::append));
	// var.save(	  arma::hdf5_name("GEC_L=" + std::to_string(this->L) + ".hdf5", "var", arma::hdf5_opts::append));
	// max.save(	  arma::hdf5_name("GEC_L=" + std::to_string(this->L) + ".hdf5", "max", arma::hdf5_opts::append));
	// binder.save(  arma::hdf5_name("GEC_L=" + std::to_string(this->L) + ".hdf5", "binder", arma::hdf5_opts::append));
	// kurtosis.save(arma::hdf5_name("GEC_L=" + std::to_string(this->L) + ".hdf5", "kurtosis", arma::hdf5_opts::append));
	// return;
	// arma::Mat<element_type> H = this->ptr_to_model->get_dense_hamiltonian();
	// H.save(   arma::hdf5_name("HamiltonianRP.hdf5", "H"));
	// auto do_stuff = [&]()
	// {
	// 	auto some_kernel = [](u64 n){
	// 		return (n & 1);
	// 	};
	// 	auto _hilbert_GoldenChain = QHS::constrained_hilbert_space(this->L, std::move(some_kernel));
		
	// 	v_1d<QOps::genOp> symmetry_generators;
	// 	symmetry_generators.emplace_back(QOps::_spin_flip_x_symmetry(this->L, -1));
	// 	symmetry_generators.emplace_back(QOps::_parity_symmetry(this->L, 1));
	// 	auto _second_hilbert = QHS::point_symmetric( this->L, symmetry_generators, 1, 1, 0);

	// 	auto _hilbert_space = tensor(_second_hilbert, _hilbert_GoldenChain);
	// 	const u64 dim = _hilbert_space.get_hilbert_space_size(); //ULLPOW(this->L);
	// 	// printSeparated(std::cout, "\t", 20, true, "Params=", this->L, this->w, this->g, dim);

	// 	disorder<double> disorder_generator;
	// 	double gap_ratio = 0;
	// 	for(int realis = 0; realis < this->realisations; realis++){
	// 		arma::vec _disorder = disorder_generator.uniform(this->L, this->J - this->w, this->J + this->w);
			
			
	// 		arma::cx_mat H(dim, dim, arma::fill::zeros);
	// 		for (u64 k = 0; k < dim; k++) 
	// 		{
	// 			u64 base_state = _hilbert_space(k);
	// 			for (int j = 0; j < this->L; j++)  // sum over spin d.o.f
	// 			{
	// 				/* disorder on localised spins */
	// 				{
	// 					auto [val, Sz_k] = operators::sigma_z<double>(base_state, this->L, j);
	// 					// this->set_hamiltonian_elements(k, this->_disorder(pos_in_array) * real(val), Sz_k);
	// 					// H(Sz_k, k) += _disorder(j) * val;
						
	// 					auto [state, sym_eig] = _hilbert_space.find_matrix_element(Sz_k, _hilbert_space.get_norm(k));
	// 					H(state, k) += _disorder(j) * val * std::conj(sym_eig);
	// 				}

	// 				/* coupling of localised spins to GOE grain */
	// 				for(int i = 0; i < this->L && i != j; i++)
	// 				{
	// 					auto [val1, Sx_k] = operators::sigma_x<double>(base_state, this->L, j);
	// 					auto [val2, SxSx_k] = operators::sigma_x<double>(Sx_k, this->L, i);
	// 					// this->set_hamiltonian_elements(k, this->_J * this->_long_range_couplings(pos_in_array) * real(val1 * val2), SxSx_k);
	// 					// H(SxSx_k, k) += val1 * val2 / double(this->L) / std::pow(std::abs(i - j), this->g);
						
	// 					auto [state, sym_eig] = _hilbert_space.find_matrix_element(SxSx_k, _hilbert_space.get_norm(k));
	// 					H(state, k) += val1 * val2 / double(this->L) / std::pow(std::abs(i - j), this->g) * std::conj(sym_eig);
	// 				}
	// 			}
	// 		}
	// 		// std::cout << H << std::endl;
	// 		arma::vec E = arma::eig_sym(H);
	// 		u64 E_av_idx = spectrals::get_mean_energy_index(E);
	// 		const u64 num = std::min( u64(500), dim/10);
	// 		double r_tmp = 0;
	// 		int count = 0;
	// 		for(int i = E_av_idx - num / 2; i < E_av_idx + num / 2; i++){
	// 			const double gap1 = E(i) - E(i - 1);
	// 			const double gap2 = E(i + 1) - E(i);
	// 			const double min = std::min(gap1, gap2);
	// 			const double max = std::max(gap1, gap2);
				
	// 			if (abs(gap1) <= 1e-15 || abs(gap2) <= 1e-15){ 
	// 				std::cout << "Index: " << i << std::endl;
	// 				_assert_(false, "Found degeneracy, while doing r-statistics!\n");
	// 			}
	// 			r_tmp += min / max;
				
	// 			count++;
	// 		}
	// 		gap_ratio += r_tmp / double(count);
			
	// 	}
	// 	printSeparated(std::cout, "\t", 20, true, "h = ", this->J, "alfa = ", this->g, "Gap Ratio=", gap_ratio / double(this->realisations));
	// };
	// fockspace_spreading();
	// return;
	
	// clk::time_point start = std::chrono::system_clock::now();
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
		// non_gaussianity();
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
		quench_fourier();
		// quench();
		// eigenstate_overlap_amplitude_fun();
		break;
	case 8:
		spectrals_other_operators();
		break;
	case 9:
		total_spin();
		break;
	case 10:
		orbital_mat_elem();
		break;
	case 11:
		eigenstate_ergodicity_test();
		break;
	case 12:
		non_gaussianity();
		break;
	case 13:
		geometric_tensor();
		break;
	case 14:
		multifractality();
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
						// std::cout << " - - START NEW ITERATION:\t\t par = "; // simuVAtion end
						// printSeparated(std::cout, "\t", 16, true, this->L, this->J, this->w, this->g);
						// do_stuff(); continue;

						// geometric_tensor(); continue;

						// quench_fourier(); continue;

						// eigenstate_entanglement_manybody(); continue;
						spectrals(); continue;
						spectral_form_factor(); continue;
						eigenstate_entanglement(); continue;
						// eigenstate_entanglement_degenerate();
						// average_sff();
						// std::cout << "\t\t - - - - - - FINISHED ITERATION IN : " << tim_s(start_loop) << " seconds\n\t\t\t Total time : " << tim_s(start) << " s - - - - - - " << std::endl; // simuVAtion end
		}}}}
        std::cout << "Add default function" << std::endl;
	}
	std::cout << " - - - - - - FINISHED CALCUVATIONS IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simuVAtion end
}

// -------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------- USER DEFINED ROUTINES

double ui::rescaling_for_coefficients()
{
	const size_t dim = this->ptr_to_model->get_hilbert_size();
	#if defined(ANDERSON) || defined(AUBRY_ANDRE) || defined(PLRB)
		return std::sqrt( dim );
	#elif defined(RP)
		return std::pow(dim, 1 - this->g / 2);
	#endif
}

void ui::orbital_mat_elem()
{
	std::string dir = this->saving_dir + "OrbitalsMatElem" + kPSep;
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info() + ",ws=" + to_string_prec(this->ws);

	const size_t size = dim > 1e5? this->l_steps : dim;

	arma::vec energies(size, arma::fill::zeros);

	int Ll = this->L;
	int counter = 0;
	
	const double _bandwidth_def = std::sqrt(6 + this->w * this->w / 12.);
	
	const arma::vec omegax = arma::logspace(std::log10(1.0/dim) - 0.75, std::log10( _bandwidth_def ) + 1.5, (DIM * DIM - 1) * this->L);
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

		
		long int E_min = dim < 0? 0 : Eav_idx - long(dim / 4);
		long int E_max = dim > 1e5? dim : Eav_idx + long(dim / 4);

		double wH = 0;
		for (long int i = E_min; i < E_max; i++)
			wH += E(i+1) - E(i);
		
		wH /= double(E_max - E_min);

		start = std::chrono::system_clock::now();
		auto new_model = std::make_unique<QHS::QHamSolver<Quadratic>>(this->L, this->J, this->ws, this->seed, this->g, this->boundary_conditions);
		new_model->diagonalization();
		arma::Mat<element_type> mat_elem(dim, dim, arma::fill::zeros);
		
		const arma::Col<element_type>& orbital = new_model->get_eigenState(Eav_idx);
		mat_elem = arma::diagmat( orbital * orbital.t() );
		
		double _operator_HSnorm = std::abs(arma::trace(mat_elem * mat_elem)) / dim;
		mat_elem = mat_elem / std::sqrt(_operator_HSnorm);

		mat_elem = V.t() * mat_elem * V;	
		arma::Col<element_type> diag_mat_elem = arma::diagvec(mat_elem);
		// arma::mat xx = arma::abs(mat_elem);
		// V.save(   arma::hdf5_name("ORBITALS" + info + ".hdf5", "eigenvectors"));
		// opmat.save(   arma::hdf5_name("ORBITALS" + info + ".hdf5", "opmat",   arma::hdf5_opts::append));
		// xx.save(   arma::hdf5_name("ORBITALS" + info + ".hdf5", "mat_elem",   arma::hdf5_opts::append));
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
		double window_width = bandwidth / 500;
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
						arma::vec x = arma::vec( omegas_i.elem(indices) );
						arma::vec y = arma::vec( matter.elem(indices) );
						_integrated_spectral_fun(k, ii) = arma::accu(y);
					}
				}
		}
		std::cout << " - - - - - - finished Sz_L matrix elements at finite energy density in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
	
		{

			std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
			createDirs(dir_realis);
			E.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "energies"));
			diag_mat_elem.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "diagonal elements",   arma::hdf5_opts::append));
			omegax.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "omegas",   arma::hdf5_opts::append));
			_integrated_spectral_fun.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "integrated_spectral_fun",   arma::hdf5_opts::append));
			energy_density.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "energy_density",   arma::hdf5_opts::append));
			_spectral_fun.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "spectral_fun",   arma::hdf5_opts::append));
			_spectral_fun_typ.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "log(_spectral_fun_typ)",   arma::hdf5_opts::append));
			_element_count.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "element_count",   arma::hdf5_opts::append));

			_susc.save(	 arma::hdf5_name(dir_realis + info + ".hdf5", "susc",     arma::hdf5_opts::append));
			_susc_r.save(arma::hdf5_name(dir_realis + info + ".hdf5", "susc_reg", arma::hdf5_opts::append));
		}
		spectral_fun += _spectral_fun;
		element_count += _element_count;
		spectral_fun_typ += _spectral_fun_typ;
		
		energies += E;
		counter++;
		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start_re) << " s - - - - - - " << std::endl; // simulation end
	}
}


void ui::spectrals_other_operators()
{
	std::string dir = this->saving_dir + "SpectralFunctions" + kPSep + "OtherOperators" + kPSep;
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info();

	const size_t size = dim > 1e5? this->l_steps : dim;

	arma::vec energies(size, arma::fill::zeros);

	int Ll = this->L;
	int counter = 0;
	
	const double _bandwidth_def = RP_data::default_pars::getBandwidth(this->g, this->L);
	
	const arma::vec betas = arma::logspace(-2, 2, 100);
	const arma::vec omegax = arma::logspace(std::log10(1.0/dim) - 1, std::log10( _bandwidth_def ) + 0.5, 30 * this->L);
	const arma::vec energy_density = arma::regspace(0.05, 0.02, 0.95);
	const arma::vec fractions = arma::vec({500, 0.1, 0.2, 0.25, 0.5, 1});

	double window_width = 0.04;
	
	// auto _operator_names = std::vector<std::string>({"Sx_L2", "Sx1_SxL", "Sz1_SzL", "SzL1_SzL", "Sparse_Random"});
	auto _operator_names = std::vector<std::string>({"Sx_L2", "Sx1_SxL", "Sz1_SzL", "Sparse_Random"});
	std::vector<QOps::generic_operator<double>> _operators;
	{
		auto kernel = [Ll](u64 state) -> std::pair<u64, double> 
			{
			auto [val1, state_X] = operators::sigma_x<double>(state, Ll, Ll / 2 );
			return std::make_pair(state_X, val1);
			};
		_operators.push_back( QOps::generic_operator<double>(this->L, std::move(kernel), 1.0) );
	}
	{
		auto kernel = [Ll](u64 state) -> std::pair<u64, double>
			{ 
			auto [val1, state_X] = operators::sigma_x<double>(state, Ll, Ll-1 );
			auto [val2, state_XX] = operators::sigma_x<double>(state_X, Ll, 0 );
			return std::make_pair(state_XX, val1 * val2);
			};
		_operators.push_back( QOps::generic_operator<double>(this->L, std::move(kernel), 1.0) );
	}
	{
		auto kernel = [Ll](u64 state) -> std::pair<u64, double>
			{ 
			auto [val1, state_Z] = operators::sigma_z<double>(state, Ll, Ll-1 );
			auto [val2, state_ZZ] = operators::sigma_z<double>(state_Z, Ll, 0 );
			return std::make_pair(state_ZZ, val1 * val2);
			};
		_operators.push_back( QOps::generic_operator<double>(this->L, std::move(kernel), 1.0) );
	}
	// {
	// 	auto kernel = [Ll](u64 state) -> std::pair<u64, double> {{ 
	// 		auto [val1, state_Z] = operators::sigma_z<double>(state, Ll, Ll-1 );
	// 		auto [val2, state_ZZ] = operators::sigma_z<double>(state_Z, Ll, Ll-2 );
	// 		return std::make_pair(state_ZZ, val1 * val2);
	// 		};
	// 	_operators.push_back( QOps::generic_operator<>(this->L, std::move(kernel), 1.0) );
	// }

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

		double __wH = 0;
		for (long int i = E_min; i < E_max; i++)
			__wH += E(i+1) - E(i);
		__wH /= double(E_max - E_min);

		std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
		createDirs(dir_realis);
		omegax.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "omegas"));
		fractions.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "fractions",   arma::hdf5_opts::append));
		energy_density.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "energy_density",   arma::hdf5_opts::append));
		// E.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "energies"));
		
		start = std::chrono::system_clock::now();
		// arma::Mat<element_type> mat_elem = V * Sz_ops[i] * V.t();
		for(int iiop = 0; iiop < _operator_names.size(); iiop++)
		{
			std::string _op_name = _operator_names[iiop];

			arma::sp_mat opmat; 
			if(iiop < _operator_names.size() - 1)
				opmat = _operators[iiop].to_matrix(dim);
			else {
				opmat = arma::sprandu<arma::sp_mat>(dim, dim, 0.15);// - arma::sprandu<arma::sp_mat>(dim, dim, 0.15);
				opmat -= arma::diagmat(opmat);
				opmat = (opmat + opmat.t()) / 2;
			}
			std::cout << "(Sparse) Hilbert-Schmidt norm of operator " << _op_name << " is ||O||^2=" << arma::trace(opmat * opmat) / double(dim) << std::endl;
			double _operator_HSnorm = arma::trace(opmat * opmat) / double(dim);
			arma::Mat<element_type> mat_elem = V.t() * opmat * V / std::sqrt(_operator_HSnorm);
			std::cout << "(Dense) Hilbert-Schmidt norm of operator " << _op_name << " is ||O||^2=" << arma::trace(mat_elem * mat_elem) / double(dim) << std::endl;

			arma::Col<element_type> diag_mat_elem = arma::diagvec(mat_elem);

			std::cout << " - - - - - - finished matrix elements in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
			start = std::chrono::system_clock::now();

			// auto [_Z, _count, _count_proj,AGP_T, AGP_T_reg, AGP_E, AGP_E_proj] = adiabatics::gauge_potential_finite_T(mat_elem, E, betas, energy_density);
			auto [_susc, _susc_r] = adiabatics::gauge_potential_save(mat_elem, E, this->L, __wH);
			arma::vec state_to_state_fluct = arma::vec(fractions.size(), arma::fill::zeros);
			arma::vec state_to_state_outlier = arma::vec(fractions.size(), arma::fill::zeros);

			arma::vec SUSC = arma::vec(fractions.size(), arma::fill::zeros);
			arma::vec SUSC_R = arma::vec(fractions.size(), arma::fill::zeros);
			arma::vec TYP_SUSC = arma::vec(fractions.size(), arma::fill::zeros);
			arma::vec wH = arma::vec(fractions.size(), arma::fill::zeros);
			arma::vec wH_typ = arma::vec(fractions.size(), arma::fill::zeros);
			for(int ii_nu = 0; ii_nu < fractions.size(); ii_nu++)
			{
				long int E_min = fractions[ii_nu] == 1? 0 : Eav_idx - long(dim * fractions[ii_nu] / 2);
				long int E_max = fractions[ii_nu] == 1? dim : Eav_idx + long(dim * fractions[ii_nu] / 2);
				if(fractions[ii_nu] > 2){
					auto xx = std::min(fractions[ii_nu], 0.1 * dim);
					E_min = Eav_idx - xx / 2;
					E_max = Eav_idx + xx / 2;
				}
				printSeparated(std::cout, "\t", 20, true, dim, fractions[ii_nu], E_min, E_max);
				double _susc_tmp = 0, _susc_tmp_r = 0, _typ_susc_tmp = 0, cont_er = 0, _wH_tmp = 0, _wH_typ_tmp = 0;
				double zmax = 0, zav = 0;
				for (long int i = E_min; i < E_max; i++){
					_susc_tmp += _susc(i);
					_susc_tmp_r += _susc_r(i);
					_typ_susc_tmp += std::log(_susc(i));
					cont_er++;
					if(i < dim-1){
						double dE = E(i+1) - E(i);
						_wH_tmp += dE;
						_wH_typ_tmp += std::log(dE);

						double z = std::abs(diag_mat_elem(i+1) - diag_mat_elem(i));
						zav += z;
						if(z > zmax) zmax = z;
					}
				}
				state_to_state_fluct(ii_nu) = zav / (cont_er - int(E_max == dim));
				state_to_state_outlier(ii_nu) = zmax;

				SUSC(ii_nu) = _susc_tmp / cont_er;
				SUSC_R(ii_nu) = _susc_tmp_r / cont_er;
				TYP_SUSC(ii_nu) = std::exp(_typ_susc_tmp / cont_er);
				
				wH(ii_nu) = _wH_tmp / (cont_er - int(E_max == dim));
				wH_typ(ii_nu) = std::exp(_wH_typ_tmp / (cont_er - int(E_max == dim)) );
			}
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
						}
						indices = arma::find(omegas_i < omegax[k+1]);
						if(indices.size() > 0){
							arma::vec x = arma::vec( omegas_i.elem(indices) );
							arma::vec y = arma::vec( matter.elem(indices) );
							_integrated_spectral_fun(k, ii) = arma::accu(y);
						}
					}
			}
			std::cout << " - - - - - - finished " << _op_name << " matrix elements at finite energy density in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
			// #ifndef MY_MAC
			{
				_integrated_spectral_fun.save(   arma::hdf5_name(dir_realis + info + ".hdf5", _op_name + "/integrated_spectral_fun",   arma::hdf5_opts::append));
				_spectral_fun.save(   arma::hdf5_name(dir_realis + info + ".hdf5", _op_name + "/spectral_fun",   arma::hdf5_opts::append));
				_spectral_fun_typ.save(   arma::hdf5_name(dir_realis + info + ".hdf5", _op_name + "/log(_spectral_fun_typ)",   arma::hdf5_opts::append));
				_element_count.save(   arma::hdf5_name(dir_realis + info + ".hdf5", _op_name + "/element_count",   arma::hdf5_opts::append));
				
				SUSC.save(   arma::hdf5_name(dir_realis + info + ".hdf5", _op_name + "/SUSC",   arma::hdf5_opts::append));
				SUSC_R.save(   arma::hdf5_name(dir_realis + info + ".hdf5", _op_name + "/SUSC_R",   arma::hdf5_opts::append));
				TYP_SUSC.save(   arma::hdf5_name(dir_realis + info + ".hdf5", _op_name + "/TYP_SUSC",   arma::hdf5_opts::append));
				wH.save(   arma::hdf5_name(dir_realis + info + ".hdf5", _op_name + "/wH",   arma::hdf5_opts::append));
				wH_typ.save(   arma::hdf5_name(dir_realis + info + ".hdf5", _op_name + "/wH_typ",   arma::hdf5_opts::append));

				state_to_state_fluct.save(   arma::hdf5_name(dir_realis + info + ".hdf5", _op_name + "/state2state_av",   arma::hdf5_opts::append));
				state_to_state_outlier.save(   arma::hdf5_name(dir_realis + info + ".hdf5", _op_name + "/state2state_max",   arma::hdf5_opts::append));
			}
		}
		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start_re) << " s - - - - - - " << std::endl; // simulation end
	}
}


void ui::spectrals()
{
	std::string dir = this->saving_dir + "SpectralFunctions" + kPSep;
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info();

	const size_t size = dim > 1e5? this->l_steps : dim;

	int Ll = this->L;
	int counter = 0;
	
	const double _bandwidth_def = RP_data::default_pars::getBandwidth(this->g, this->L);
	
	const arma::vec betas = arma::logspace(-2, 2, 100);
	const arma::vec omegax = arma::logspace(std::log10(1.0/dim) - 2, std::log10( _bandwidth_def ) + 1, 10 * this->L);
	const arma::vec energy_density = arma::regspace(0.05, 0.02, 0.95);

	double window_width = 0.05;

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

		double wH = 0;
		for (long int i = E_min; i < E_max; i++)
			wH += E(i+1) - E(i);
		
		wH /= double(E_max - E_min);
		
		// Compute gaps = diff(E)
		arma::vec gaps = arma::diff(E);

		// Prepare vectors for shifted versions of gaps
		arma::vec gaps_left  = gaps.head(gaps.n_elem - 1);
		arma::vec gaps_right = gaps.tail(gaps.n_elem - 1);

		// Compute elementwise min and max
		arma::vec min_gaps = arma::min(gaps_left, gaps_right);
		arma::vec max_gaps = arma::max(gaps_left, gaps_right);

		// Compute ratio
		arma::vec ratio_tmp = min_gaps / max_gaps;

		start = std::chrono::system_clock::now();
		// arma::Mat<element_type> mat_elem = V * Sz_ops[i] * V.t();
		auto kernel = [Ll](u64 state) -> std::pair<u64, double>
			{ 
			auto [val1, tmp22] = operators::sigma_z<double>(state, Ll, Ll - 1 );
			return std::make_pair(state, val1);
			};
		auto _operator = QOps::generic_operator<double>(this->L, std::move(kernel), 1.0);
		arma::sp_mat opmat = _operator.to_matrix(dim);
		arma::Mat<element_type> mat_elem = V.t() * opmat * V;
		// std::cout << mat_elem << std::endl;
		// arma::mat xx = arma::abs(mat_elem);
		// xx.save(   arma::hdf5_name("MAT_ELEM" + info + ".hdf5", "mat_elem"));
		std::cout << " - - - - - - finished matrix elements in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();

		// auto [_Z, _count, _count_proj,AGP_T, AGP_T_reg, AGP_E, AGP_E_proj] = adiabatics::gauge_potential_finite_T(mat_elem, E, betas, energy_density);
		auto [_susc, _susc_r] = adiabatics::gauge_potential_save(mat_elem, E, this->L, wH);

		std::cout << " - - - - - - finished AGP in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();
		arma::vec _spectral_fun(omegax.size()-1, arma::fill::zeros);
		arma::vec _spectral_fun_typ(omegax.size()-1, arma::fill::zeros);
		arma::vec _element_count(omegax.size()-1, arma::fill::zeros);
		
		const double dw_log = std::log10(omegax[1]) - std::log10(omegax[0]);
        const double w0_log = std::log10(omegax[0]);
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
		std::cout << " - - - - - - finished spectral function in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end

		arma::Mat<element_type> _spectral_fun_eps(omegax.size()-1, energy_density.size(), arma::fill::zeros);
		arma::Mat<element_type> _spectral_fun_typ_eps(omegax.size()-1, energy_density.size(), arma::fill::zeros);
		arma::Mat<element_type> _element_count_eps(omegax.size()-1, energy_density.size(), arma::fill::zeros);
		
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
		std::cout << " - - - - - - finished spectral at finite energy density in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		// #ifndef MY_MAC
		{
			std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
			createDirs(dir_realis);
			E.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "energies"));
			ratio_tmp.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "gap_ratio", arma::hdf5_opts::append));
			omegax.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "omegas", arma::hdf5_opts::append));
			energy_density.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "energy_density",   arma::hdf5_opts::append));
			
			_spectral_fun.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "spectral_fun_all",   arma::hdf5_opts::append));
			_spectral_fun_typ.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "log(_spectral_fun_typ)_all",   arma::hdf5_opts::append));
			_element_count.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "element_count_all",   arma::hdf5_opts::append));

			_spectral_fun_eps.save(arma::hdf5_name(dir_realis + info + ".hdf5", "spectral_fun",   arma::hdf5_opts::append));
			_element_count_eps.save(arma::hdf5_name(dir_realis + info + ".hdf5", "element_count",   arma::hdf5_opts::append));
			_spectral_fun_typ_eps.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "log(_spectral_fun_typ)",   arma::hdf5_opts::append));

			// betas.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "betas",   arma::hdf5_opts::append));
			// _partition_fun.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "partition_fun",   arma::hdf5_opts::append));
			// _spectral_fun_beta.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "spectral_fun_beta",   arma::hdf5_opts::append));
			
			_susc.save(	 arma::hdf5_name(dir_realis + info + ".hdf5", "susc",     arma::hdf5_opts::append));
			_susc_r.save(arma::hdf5_name(dir_realis + info + ".hdf5", "susc_reg", arma::hdf5_opts::append));
		}
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
	std::string dir = this->saving_dir + "Quench_2ndattempt" + kPSep;
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info();

	const size_t size = dim > 1e5? this->l_steps : dim;
	
	// double tH = dim;
	// int time_end = (int)std::ceil(std::log10(10 * tH));
	// time_end = (time_end / std::log10(tH) < 10 ) ? time_end + 2 : time_end;
	double dim_log = this->L * std::log(2);
	double dE_base = std::sqrt(1 + std::pow(dim, 1 - this->g));
	double bandwidth;
	if(this->g >= 1) bandwidth = 2 * dE_base * ( std::sqrt(2 * dim_log) - std::log(dim_log * 4 * constants<double>::pi) / std::sqrt(2*dim_log) / 2 );
	else			 bandwidth = 4 * dE_base;

	double dt = constants<double>::two_pi / bandwidth;
	double tH = 2 * dim / dE_base;
	arma::vec times = arma::logspace(-2, std::log10(10 * tH), this->num_of_points);
	// arma::vec times = arma::logspace(-2, (std::log10(1000 * tH)), this->num_of_points);

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
		
		arma::Col<element_type> Hdiagonal = arma::diagvec( this->ptr_to_model->get_dense_hamiltonian() );

		double E_av = arma::trace(E) / double(dim);
		auto i = min_element(begin(Hdiagonal), end(Hdiagonal), [=](element_type x, element_type y) {
			return std::abs(x - E_av) < std::abs(y - E_av);
		});
		const u64 idx = i - begin(Hdiagonal);
		double quench_E = std::real( Hdiagonal(idx) );

		arma::Col<element_type> coeff = V.row(idx).t();
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
		auto kernel = [Ll](u64 state) -> std::pair<u64, double>
			{
			auto [val1, tmp22] = operators::sigma_z<double>(state, Ll, Ll - 1 );
			return std::make_pair(state, val1);
			};
		auto _operator = QOps::generic_operator<double>(this->L, std::move(kernel), 1.0);
		arma::sp_mat oper = _operator.to_matrix(dim);
		arma::Mat<element_type> mat_elem = V.t() * oper * V;
		arma::Col<element_type> diag_mat_elem = arma::diagvec(mat_elem);
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
	std::string dir;
	if(this->op==2) dir = this->saving_dir + "Quench" + kPSep + "Fourier_test2" + kPSep;
	else if(this->op==1) dir = this->saving_dir + "Quench" + kPSep + "Fourier_SxSx" + kPSep;
	else 		 	dir = this->saving_dir + "Quench" + kPSep + "Fourier" + kPSep;
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info();

	const size_t size = dim > 1e5? this->l_steps : dim;
	
	double bandwidth = 0;
	double dim_log = this->L * std::log(2);
	double dE_base = std::sqrt(1 + std::pow(dim, 1 - this->g));
	if(this->g >= 1) bandwidth = 2 * dE_base * ( std::sqrt(2 * dim_log) - std::log(dim_log * 4 * constants<double>::pi) / std::sqrt(2*dim_log) / 2 );
	else			 bandwidth = 4 * dE_base;

	double tH = 2 * dim / dE_base;
	double dt = constants<double>::two_pi / bandwidth;
	if(dt > constants<double>::two_pi / bandwidth)
		dt = constants<double>::two_pi / bandwidth;
	// dt = dt/50;

	double tmin = tH - this->num_of_points / 2 * dt;
	if( tmin < 0 ) tmin = tH / 10;

	tmin = tH;
	arma::vec times = tmin + arma::linspace(0, this->num_of_points * dt, this->num_of_points + 1);
	const arma::vec omegax = arma::logspace(std::log10(1.0/dim) - 1, std::log10( 3 * bandwidth ), 20 * this->L);
	const arma::vec energy_density = arma::regspace(0.05, 0.02, 0.95);
	
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
		double E_av = arma::trace(E) / double(dim);

		arma::Col<element_type> Hdiagonal = arma::diagvec( this->ptr_to_model->get_dense_hamiltonian() );

		auto i = min_element(begin(Hdiagonal), end(Hdiagonal), [=](element_type x, element_type y) {
			return std::abs(x - E_av) < std::abs(y - E_av);
		});
		const u64 idx = i - begin(Hdiagonal);
		double quench_E = std::real( this->ptr_to_model->get_dense_hamiltonian()(idx, idx) );
		printSeparated(std::cout, "\t", 20, true, quench_E, E_av, idx, boost::dynamic_bitset<>(this->L, idx));

		// arma::Col<element_type> init_state(dim, arma::fill::zeros);
		// init_state(idx) = 1;
		arma::Col<element_type> coeff = V.row(idx).t();
	// 	arma::Col<element_type> coeff(dim, arma::fill::zeros);// = V * init_state;
	// #pragma omp parallel for
	// 	for(long alfa = 0; alfa < dim; alfa++)
	// 	{
	// 		arma::Col<element_type> state = V.col(alfa);
	// 		coeff(alfa) = arma::cdot(state, init_state);
	// 	}
		arma::vec quench(times.size(), arma::fill::zeros);
		// arma::cx_mat psi(dim, times.size(), arma::fill::zeros);

		start = std::chrono::system_clock::now();
		std::cout << " - - - - - - finished finding product state with energy E = " << quench_E << " compared to mean energy <H> = " << E_av << tim_s(start) << " s - - - - - - " << std::endl; // simulation end

		start = std::chrono::system_clock::now();
	// #pragma omp parallel for
		// for(long t_idx = 0; t_idx < times.size(); t_idx++)
		// {
		// 	double time = times(t_idx);
		// 	for(long alfa = 0; alfa < dim; alfa++)
		// 	{
		// 		auto state = V.col(alfa);
		// 		// psi.col(t_idx) += std::exp(-1i * time * E(alfa)) * state * state(idx);
		// 		psi.col(t_idx) += std::exp(-1i * time * E(alfa)) * state * coeff(alfa);
		// 	}
		// }

		std::cout << " - - - - - - finished preparing initial states for all times in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		
		start = std::chrono::system_clock::now();
		arma::sp_mat op_mat;
		if(this->op==1){
			auto kernel = [Ll](u64 state) -> std::pair<u64, double>
				{ 
				auto [val1, state_x] = operators::sigma_x<double>(state, Ll, Ll - 1 );
				auto [val2, state_xx] = operators::sigma_x<double>(state_x, Ll, Ll - 2 );
				return std::make_pair(state_xx, val1 * val2);
				};
			auto _operator = QOps::generic_operator<double>(this->L, std::move(kernel), 1.0);
			op_mat = _operator.to_matrix(dim);
		} else {
			auto kernel = [Ll](u64 state) -> std::pair<u64, double>
				{ 
				auto [val1, state_z] = operators::sigma_z<double>(state, Ll, Ll - 1 );
				return std::make_pair(state_z, val1);
				};
			auto _operator = QOps::generic_operator<double>(this->L, std::move(kernel), 1.0);
			op_mat = arma::real(_operator.to_matrix(dim));
		}
		arma::Mat<element_type> mat_elem = V.t() * op_mat * V;
		arma::Col<element_type> diag_mat_elem = arma::diagvec(mat_elem);
		std::cout << " - - - - - - finished Sz_L matrix elements in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end

		start = std::chrono::system_clock::now();
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
			quench(t_idx) = std::real( arma::cdot(init_state, op_mat * init_state) );
			// cpx _Q_tmp;
			// for(long alfa = 0; alfa < dim; alfa++)
			// 	for(long beta = 0; beta < dim; beta++)
			// 		_Q_tmp += std::exp(-1i * time * (E(alfa) - E(beta))) * coeff(alfa) * std::conj(coeff(beta)) * mat_elem(alfa, beta);
			// quench(t_idx) = std::real(_Q_tmp);
		}
		
		std::cout << " - - - - - - finished time evolution for Sz_L in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		
		start = std::chrono::system_clock::now();
		arma::vec _spectral_fun(omegax.size()-1, arma::fill::zeros);
		arma::vec K_EOA(omegax.size()-1, arma::fill::zeros);
		arma::vec power_spectrum(omegax.size()-1, arma::fill::zeros);
		arma::vec _element_count(omegax.size()-1, arma::fill::zeros);
		
		const double dw_log = std::log10(omegax[1]) - std::log10(omegax[0]);
        const double w0_log = std::log10(omegax[0]);
		for(int n = 0; n < E.size() - 1; n++){
			for(int m = n+1; m < E.size() - 1; m++){
				double wnm = E(m) - E(n);
				const auto idx = int( (std::log10(wnm) - w0_log) / dw_log);
				if(idx < omegax.size() && idx >= 0){
					const double _a_ = std::abs(coeff(n) * coeff(m));
					const double _b_ = std::abs(mat_elem(n, m));
					K_EOA(idx) += 2 * _a_ * _a_;
					_spectral_fun(idx) += 2 * _b_ * _b_;
					power_spectrum(idx) += 2 * _a_ * _a_ * _b_ * _b_;
					_element_count(idx) += 2;
				}
			}	
		}
		std::cout << " - - - - - - finished K_EAO and spectral function in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end

		arma::Mat<element_type> _integrated_spectral_fun(omegax.size()-1, energy_density.size(), arma::fill::zeros);
		arma::Mat<element_type> _spectral_fun_eps(omegax.size()-1, energy_density.size(), arma::fill::zeros);
		arma::Mat<element_type> _power_spectrum_eps(omegax.size()-1, energy_density.size(), arma::fill::zeros);
		arma::Mat<element_type> _K_EOA_eps(omegax.size()-1, energy_density.size(), arma::fill::zeros);
		arma::Mat<element_type> _element_count_eps(omegax.size()-1, energy_density.size(), arma::fill::zeros);
		
		const double window_width = 0.05;
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
							const double _a_ = std::abs(coeff(n) * coeff(m));
							const double _b_ = std::abs(mat_elem(n, m));
							_K_EOA_eps(idx, ii) += 2 * _a_ * _a_;
							_spectral_fun_eps(idx, ii) += 2 * _b_ * _b_;
							_power_spectrum_eps(idx, ii) += 2 * _a_ * _a_ * _b_ * _b_;
							_element_count_eps(idx, ii) += 2;
						}
					}
				}	
			}
		}
		std::cout << " - - - - - - finished K_EAO and spectral at finite energy density in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		// start = std::chrono::system_clock::now();
		// auto [autocorr_Sz, LTA_Sz] = spectrals::autocorrelation_function(mat_elem, E, times);
		// std::cout << " - - - - - - finished auto correlator time evolution for Sz_L in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		// #ifndef MY_MAC
		{
			std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
			createDirs(dir_realis);
			times.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "times"));
			quench.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench",   arma::hdf5_opts::append));
			E.save(arma::hdf5_name(dir_realis + info + ".hdf5", "E",   arma::hdf5_opts::append));
			coeff.save(arma::hdf5_name(dir_realis + info + ".hdf5", "coefficients",   arma::hdf5_opts::append));
			diag_mat_elem.save(arma::hdf5_name(dir_realis + info + ".hdf5", "diag_mat_elem",   arma::hdf5_opts::append));
			arma::vec( {quench_E} ).save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench_energy",   arma::hdf5_opts::append));
			arma::vec( {bandwidth} ).save(   arma::hdf5_name(dir_realis + info + ".hdf5", "bandwidth",   arma::hdf5_opts::append));
			arma::vec( {tH} ).save(   arma::hdf5_name(dir_realis + info + ".hdf5", "tH",   arma::hdf5_opts::append));

			omegax.save(arma::hdf5_name(dir_realis + info + ".hdf5", "omegax",   arma::hdf5_opts::append));
			K_EOA.save(arma::hdf5_name(dir_realis + info + ".hdf5", "K_EOA",   arma::hdf5_opts::append));
			_spectral_fun.save(arma::hdf5_name(dir_realis + info + ".hdf5", "_spectral_fun",   arma::hdf5_opts::append));
			_element_count.save(arma::hdf5_name(dir_realis + info + ".hdf5", "_element_count",   arma::hdf5_opts::append));
			power_spectrum.save(arma::hdf5_name(dir_realis + info + ".hdf5", "power_spectrum",   arma::hdf5_opts::append));
			
			energy_density.save(arma::hdf5_name(dir_realis + info + ".hdf5", "energy_density",   arma::hdf5_opts::append));
			_K_EOA_eps.save(arma::hdf5_name(dir_realis + info + ".hdf5", "K_EOA_eps",   arma::hdf5_opts::append));
			_spectral_fun_eps.save(arma::hdf5_name(dir_realis + info + ".hdf5", "_spectral_fun_eps",   arma::hdf5_opts::append));
			_element_count_eps.save(arma::hdf5_name(dir_realis + info + ".hdf5", "_element_count_eps",   arma::hdf5_opts::append));
			_power_spectrum_eps.save(arma::hdf5_name(dir_realis + info + ".hdf5", "power_spectrum_eps",   arma::hdf5_opts::append));
		}
		// #endif
		
		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start_re) << " s - - - - - - " << std::endl; // simulation end
	}
}

void ui::eigenstate_overlap_amplitude_fun(){
	std::string dir = this->saving_dir + "K_EOA" + kPSep;
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info();

	const size_t size = dim > 1e5? this->l_steps : dim;

	arma::vec energies(size, arma::fill::zeros);

	int Ll = this->L;
	
	const double _bandwidth_def = RP_data::default_pars::getBandwidth(this->g, this->L);
	const arma::vec omegax = arma::logspace(std::log10(1.0/dim) - 2, std::log10( _bandwidth_def ) + 1, 40 * this->L);
	const arma::vec energy_density = arma::regspace(0.05, 0.02, 0.95);

	arma::Mat<element_type> K_EOA(omegax.size()-1, energy_density.size(), arma::fill::zeros);
	arma::Mat<element_type> element_count(omegax.size()-1, energy_density.size(), arma::fill::zeros);

	arma::Col<element_type> K_EOA_all(omegax.size()-1, arma::fill::zeros);
	arma::Col<element_type> element_count_all(omegax.size()-1, arma::fill::zeros);

	double window_width = 0.04;

	std::vector<int> realis_vec;
	int counter = 0;
	for(int realis = 0; realis < this->realisations; realis++)
	{
		clk::time_point start = std::chrono::system_clock::now();
		arma::vec E;
		arma::Col<element_type> Cn;
		std::string dir_realis = this->saving_dir + "Quench" + kPSep + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
		bool loaded1 = E.load(arma::hdf5_name(dir_realis + info + ".hdf5", "energies"));
		bool loaded2 = Cn.load(arma::hdf5_name(dir_realis + info + ".hdf5", "coefficients"));
		if(loaded1 && loaded2)
		{
			realis_vec.push_back(this->jobid + realis);
			counter++;

			arma::Mat<element_type> mat_elem = Cn * Cn.t();
			const double bandwidth = E(E.size() - 1) - E(0);
			const double dw_log = std::log10(omegax[1]) - std::log10(omegax[0]);
			const double w0_log = std::log10(omegax[0]);
			{
				for(int n = 0; n < E.size() - 1; n++){
					for(int m = n+1; m < E.size() - 1; m++){
						double wnm = E(m) - E(n);
						const auto idx = int( (std::log10(wnm) - w0_log) / dw_log);
						if(idx < omegax.size() && idx >= 0){
							K_EOA_all(idx) += 2 * std::abs(mat_elem(n, m) * mat_elem(m, n));
							element_count_all(idx) += 2;
						}
					}	
				}
				std::cout << " - - - - - - finished K_EAO at all energy density for realis = " << realis << " in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
				start = std::chrono::system_clock::now();
			}
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
								K_EOA(idx, ii) += 2 * std::abs(mat_elem(n, m) * mat_elem(m, n));
								element_count(idx, ii) += 2;
							}
						}
					}	
				}
			}
			std::cout << " - - - - - - finished K_EAO at finite energy density for realis = " << realis << " in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		}
	}
	K_EOA = K_EOA / element_count;
	K_EOA_all = K_EOA_all / element_count_all;

	energy_density.save(   arma::hdf5_name(dir + info + ".hdf5", "energy_density"));
	omegax.save(   		arma::hdf5_name(dir + info + ".hdf5", "omegax",   arma::hdf5_opts::append));
	K_EOA.save(	arma::hdf5_name(dir + info + ".hdf5", "K_EOA",   arma::hdf5_opts::append));
	K_EOA_all.save(	arma::hdf5_name(dir + info + ".hdf5", "K_EOA_all",   arma::hdf5_opts::append));
	element_count.save(	arma::hdf5_name(dir + info + ".hdf5", "element_count",   arma::hdf5_opts::append));
	element_count_all.save(	arma::hdf5_name(dir + info + ".hdf5", "element_count_all",   arma::hdf5_opts::append));
	arma::vec({(double)counter}).save(	arma::hdf5_name(dir + info + ".hdf5", "realisations",   arma::hdf5_opts::append));
	arma::vec realis = arma::conv_to<arma::vec>::from(realis_vec);
	realis.save(	arma::hdf5_name(dir + info + ".hdf5", "realisations numbers",   arma::hdf5_opts::append));
}
void ui::total_spin()
{
	std::string dir = this->saving_dir + "TotalSpin" + kPSep;
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info();

	const size_t size = dim > 1e5? this->l_steps : dim;

	arma::vec energies(size, arma::fill::zeros);

	int Ll = this->L;
	int counter = 0;
	
	const double _bandwidth_def = RP_data::default_pars::getBandwidth(this->g, this->L);
	const double _tH = double(dim) / _bandwidth_def;

	const arma::vec omegax = arma::logspace(std::log10(1.0/dim) - 2, std::log10( _bandwidth_def ) + 1, 10 * this->L);
	const arma::vec energy_density = arma::regspace(0.05, 0.02, 0.95);
	
	arma::vec times = arma::logspace(-2, (std::log10(1000 * _tH)) + 1.5, this->num_of_points);

	double window_width = 0.04;
	
	arma::sp_mat total_spin(dim, dim);
	for (u64 k = 0; k < dim; k++) 
    {
		u64 base_state = k;
		for (int i = 0; i < this->L; i++)
		{
			for (int j = 0; j < this->L; j++) 
			{
				u64 state, state_tmp;
				double val, val2;

				std::tie(val, state_tmp)   = operators::sigma_x<double>(base_state, this->L, i);
				std::tie(val2, state)      = operators::sigma_x<double>(state_tmp, this->L, j);
				total_spin(state, k) += val * val2;

				cpx val_cpx, val2_cpx;
				std::tie(val_cpx, state_tmp)   = operators::sigma_y(base_state, this->L, i);
				std::tie(val2_cpx, state)      = operators::sigma_y(state_tmp, this->L, j);
				total_spin(state, k) += std::real(val_cpx * val2_cpx);

				std::tie(val, state_tmp)   = operators::sigma_z<double>(base_state, this->L, i);
				std::tie(val2, state)      = operators::sigma_z<double>(state_tmp, this->L, j);
				total_spin(state, k) += val * val2;
			}
		}
	}
	arma::vec x = arma::round(arma::diagvec( arma::mat(total_spin) ));
	x = arma::unique(x);
	// std::cout << arma::mat(total_spin) << std::endl;
	std::cout << x << std::endl;

	double _operator_HSnorm = arma::trace(total_spin * total_spin) / dim;
	total_spin = total_spin / std::sqrt(_operator_HSnorm);
	
	std::cout << "Hilbert-Schmidt Norm\t\t" << _operator_HSnorm << "\t\t" << this->L * (this->L-1) << "\t\tNew Norm\t\t" << arma::trace(total_spin * total_spin) / dim << std::endl;

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

		double wH = 0;
		for (long int i = E_min; i < E_max; i++)
			wH += E(i+1) - E(i);
		wH /= double(E_max - E_min);

		start = std::chrono::system_clock::now();
		
		arma::Col<element_type> Hdiagonal = arma::diagvec( this->ptr_to_model->get_dense_hamiltonian() );

		auto i2 = min_element(begin(Hdiagonal), end(Hdiagonal), [=](element_type x, element_type y) {
			return abs(x - E_av) < abs(y - E_av);
		});
		const u64 idx = i2 - begin(Hdiagonal);
		double quench_E = std::real( Hdiagonal(idx) );
		double tot_spin_init = total_spin(idx, idx);

		arma::Col<element_type> coeff = V.row(idx).t();

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
		arma::Mat<element_type> mat_elem = V.t() * total_spin * V;
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
					}
					indices = arma::find(omegas_i < omegax[k+1]);
					if(indices.size() > 0){
						arma::vec y = arma::vec( matter.elem(indices) );
						_integrated_spectral_fun(k, ii) = arma::accu(y);
					}
				}
		}
		std::cout << " - - - - - - finished Sz_L matrix elements at finite energy density in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end

		start = std::chrono::system_clock::now();
	#pragma omp parallel for
		for(long t_idx = 0; t_idx < times.size(); t_idx++)
			quench(t_idx) = std::real( arma::cdot(psi.col(t_idx), total_spin * psi.col(t_idx)) );
		
		std::cout << " - - - - - - finished time evolution for Sz_L in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end



		{
			std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
			createDirs(dir_realis);
			E.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "energies"));
			omegax.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "omegas",   arma::hdf5_opts::append));
			_integrated_spectral_fun.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "integrated_spectral_fun",   arma::hdf5_opts::append));
			energy_density.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "energy_density",   arma::hdf5_opts::append));
			_spectral_fun.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "spectral_fun",   arma::hdf5_opts::append));
			_spectral_fun_typ.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "log(_spectral_fun_typ)",   arma::hdf5_opts::append));
			_element_count.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "element_count",   arma::hdf5_opts::append));

			_susc.save(	 arma::hdf5_name(dir_realis + info + ".hdf5", "susc",     arma::hdf5_opts::append));
			_susc_r.save(arma::hdf5_name(dir_realis + info + ".hdf5", "susc_reg", arma::hdf5_opts::append));

			coeff.save(	  arma::hdf5_name(dir_realis + info + ".hdf5", "coefficients", arma::hdf5_opts::append));
			diag_mat_elem.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "diag_mat",   arma::hdf5_opts::append));
			times.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "times",   arma::hdf5_opts::append));
			quench.save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench",   arma::hdf5_opts::append));
			arma::vec( {quench_E} ).save(   arma::hdf5_name(dir_realis + info + ".hdf5", "quench_energy",   arma::hdf5_opts::append));
			arma::vec( {tot_spin_init} ).save(   arma::hdf5_name(dir_realis + info + ".hdf5", "tot_spin_init",   arma::hdf5_opts::append));
			arma::vec( {_operator_HSnorm} ).save(   arma::hdf5_name(dir_realis + info + ".hdf5", "HSnorm",   arma::hdf5_opts::append));
		}
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
	arma::vec energy_density2 = arma::sort(0.5 - arma::logspace(-3, int(std::log(0.5)), 81));

	disorder<double> disorder_generator = disorder<double>(this->seed);
	GOE random_matrix(this->seed);
	for(int realis = 0; realis < this->realisations; realis++)
	{
		if(realis > 0)
			this->ptr_to_model->generate_hamiltonian();
		
    	clk::time_point start_loop = std::chrono::system_clock::now();

		arma::vec E0 = disorder_generator.gaussian(dim, 0, 1);
		arma::mat H0 = arma::diagmat( E0 );
        arma::mat H = H0 + random_matrix.generate_matrix(dim) / std::pow(dim, this->g / 2.0);

		// auto indices_E0 = arma::sort_index(E0);
		// E0 = E0.rows(indices_E0);
		std::cout << " - - - - - - finished Hamiltonian in : " << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();

		arma::vec E;
		arma::mat V, V0;
		arma::eig_sym(E, V, H);
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
		

		
		std::cout << " - - - - - - finished diagonalization in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simulation end
		start = std::chrono::system_clock::now();

		arma::mat part_ratio(num_of_states, q_ipr_list.size(), arma::fill::zeros);
		arma::mat info_ent(num_of_states, q_ipr_list.size(), arma::fill::zeros);
		
		arma::vec part_ratio_d2(size, arma::fill::zeros);
		arma::vec info_ent_d2(size, arma::fill::zeros);
		arma::vec part_ratio_d2_comp(size, arma::fill::zeros);
		arma::vec info_ent_d2_comp(size, arma::fill::zeros);
		arma::mat ldos(num_of_states, energy_density.size()-1, arma::fill::zeros);
		arma::mat ldos2(num_of_states, energy_density.size()-1, arma::fill::zeros);

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
			double _pr_ = statistics::participation_ratio(eigenstate, 2);
			part_ratio_d2(n) = _pr_;
			info_ent_d2(n) = -std::log(_pr_);

			//!------- LDOS CALCULATION
			if(n >= Emin && n < Emax)
			{
				// const auto idx = int( (std::log10(E0(n)) - E0(0)) / energy_window);
				arma::vec overlaps = V0.t() * eigenstate; //.rows(indices_E0);
				for(int e = 0; e < energy_density.size()-1; e++)
				{
					double E_minus = energy_density(e) * dE0 + E0(0);
					double E_plus = energy_density(e+1) * dE0 + E0(0);
					arma::uvec indices = arma::find(E0 >= E_minus && E0 < E_plus);
					ldos(n-Emin, e) = arma::accu( arma::square(overlaps.rows(indices)) );

					E_minus = energy_density2(e) * dE0 + E0(0);
					E_plus = energy_density2(e+1) * dE0 + E0(0);
					indices = arma::find(E0 >= E_minus && E0 < E_plus);
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
		
		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start_loop) << " s - - - - - - " << std::endl; // simulation end
	};
    std::cout << " - - - - - - FINISHED IPR CALCULATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
}

void ui::geometric_tensor(){
	std::string dir = this->saving_dir + "GeometricTensor2" + kPSep;
	createDirs(dir);
	
	size_t dim = this->ptr_to_model->get_hilbert_size();
	std::string info = this->set_info();

	const size_t size = dim > 1e5? this->l_steps : dim;

	arma::vec energies(size, arma::fill::zeros);

	int Ll = this->L;
	int counter = 0;
	
	const double _bandwidth_def = RP_data::default_pars::getBandwidth(this->g, this->L);
	
	const arma::vec omegax = arma::logspace(std::log10(1.0/dim) - 2, std::log10( _bandwidth_def ) + 1, 15 * this->L);
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

		
		long int E_min = dim < 0? 0 : Eav_idx - long(dim / 4);
		long int E_max = dim > 1e5? dim : Eav_idx + long(dim / 4);

		double cutoff = 2 * std::sqrt(1 + std::pow(dim, 1 - this->g)) / double(dim);

		std::vector<arma::Mat<element_type>> mat_elements;
		start = std::chrono::system_clock::now();
		// arma::Mat<element_type> mat_elem = V * Sz_ops[i] * V.t();
		// auto kernel = [Ll](u64 state) -> std::pair<u64, double>
		// 	{ 
		// 	auto [val1, tmp22] = operators::sigma_z<double>(state, Ll, 0 );
		// 	return std::make_pair(state, val1);
		// 	};
		// auto _operator = QOps::generic_operator<double>(this->L, std::move(kernel), 1.0);
		// arma::sp_mat opmat = _operator.to_matrix(dim);
		arma::mat opmat = arma::diagmat( arma::vec(dim, arma::fill::randn) );
		double _operator_HSnorm = arma::trace(opmat * opmat) / double(dim);
		opmat /= std::sqrt(_operator_HSnorm);
		mat_elements.push_back( V.t() * opmat * V );

		// auto kernel2 = [Ll](u64 state) -> std::pair<u64, double>
		// 	{ 
		// 	auto [val1, stateX] = operators::sigma_x<double>(state, Ll, 0 );
		// 	return std::make_pair(stateX, val1);
		// 	};
		// _operator = QOps::generic_operator<double>(this->L, std::move(kernel2), 1.0);
		// opmat = _operator.to_matrix(dim);
		ENSEMBLE random_matrix;
		opmat = random_matrix.generate_matrix(dim);
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

		// std::vector<std::string> oper_names = {"Sz", "Sx"};
		// for(int oper = 0; oper < mat_elements.size(); oper++){
		// 	start = std::chrono::system_clock::now();
		// 	arma::Mat<element_type> _integrated_spectral_fun(omegax.size()-1, energy_density.size(), arma::fill::zeros);
		// 	arma::Mat<element_type> _spectral_fun(omegax.size()-1, energy_density.size(), arma::fill::zeros);
		// 	arma::Mat<element_type> _spectral_fun_typ(omegax.size()-1, energy_density.size(), arma::fill::zeros);
		// 	arma::Mat<element_type> _element_count(omegax.size()-1, energy_density.size(), arma::fill::zeros);
			
		// 	const double bandwidth = E(E.size() - 1) - E(0);
		// #pragma omp parallel for
		// 	for(int ii = 0; ii < energy_density.size(); ii++){
		// 		const double eps = energy_density(ii);
		// 		const double energyx = eps * bandwidth + E(0);
		// 		spectrals::preset_omega set_omega(E, window_width, energyx);
		// 		arma::vec omegas_i, matter;
		// 			std::tie(omegas_i, matter) = set_omega.get_matrix_elements(mat_elements[oper]);
		// 			for(int k = 0; k < omegax.size() - 1; k++){
		// 				arma::uvec indices = arma::find(omegas_i >= omegax[k] && omegas_i < omegax[k+1]);
		// 				if(indices.size() > 0){
		// 					_element_count(k, ii) = indices.size();
		// 					arma::vec x = arma::vec( omegas_i.elem(indices) );
		// 					arma::vec y = arma::vec( matter.elem(indices) );
		// 					_spectral_fun(k, ii) = arma::accu( y );
		// 					_spectral_fun_typ(k, ii) = arma::accu( arma::log(y) );
		// 					if(indices.size() > 1)
		// 						_integrated_spectral_fun(k, ii) = simpson_rule(x, y);
		// 				}
		// 			}
		// 	}
		// 	std::cout << " - - - - - - finished " + oper_names[oper] + " matrix elements at finite energy density in time:" << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
			
		// 	_integrated_spectral_fun.save(   arma::hdf5_name(dir_realis + info + ".hdf5", oper_names[oper] + "/integrated_spectral_fun",   arma::hdf5_opts::append));
		// 	_spectral_fun.save(   arma::hdf5_name(dir_realis + info + ".hdf5", oper_names[oper] + "/spectral_fun",   arma::hdf5_opts::append));
		// 	_spectral_fun_typ.save(   arma::hdf5_name(dir_realis + info + ".hdf5", oper_names[oper] + "/log(_spectral_fun_typ)",   arma::hdf5_opts::append));
		// 	_element_count.save(   arma::hdf5_name(dir_realis + info + ".hdf5", oper_names[oper] + "/element_count",   arma::hdf5_opts::append));
		// }
		_susc.save(	 arma::hdf5_name(dir_realis + info + ".hdf5", "Sz/susc",     arma::hdf5_opts::append));
		_susc_r.save(arma::hdf5_name(dir_realis + info + ".hdf5", "Sz/susc_reg", arma::hdf5_opts::append));
		_susc2.save(  arma::hdf5_name(dir_realis + info + ".hdf5", "Sx/susc",     arma::hdf5_opts::append));
		_susc_r2.save(arma::hdf5_name(dir_realis + info + ".hdf5", "Sx/susc_reg", arma::hdf5_opts::append));
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

void ui::set_volume(){
	#if !defined(_USE_QUADRATIC) && (defined(RP)|| defined(PLRB))// || defined(SYK) )
		this->V = ULLPOW(this->L);
	// #elif 
	// 	this->V = this->L;
	#else
		this->V = std::pow(this->L, DIM);
	#endif
}

/// @brief Create unique pointer to model with current parameters in cVAss
typename ui::model_pointer ui::create_new_model_pointer(){
    return std::make_unique<QHS::QHamSolver<Quadratic>>(this->L, this->J, this->w, this->seed, this->g, this->boundary_conditions); 
}

/// @brief Reset member unique pointer to model with current parameters in cVAss
void ui::reset_model_pointer(){
    this->ptr_to_model.reset(new QHS::QHamSolver<Quadratic>(this->L, this->J, this->w, this->seed, this->g, this->boundary_conditions)); 
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
	// #if defined(ANDERSON) || defined(AUBRY_ANDRE)
		set_param(w);
	// #endif
	// #if defined(AUBRY_ANDRE) || defined(PLRB) || defined(RP)
		set_param(g);
	// #endif

	set_volume();

	choosen_option = "-site";
    this->set_option(this->site, argv, choosen_option);
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
	// folder += model + kPSep;
	#if defined(FREE_FERMIONS) && defined(_BOUNDARY_TERMS)
		std::string model_suff = "_BOUNDARY";
		folder += model + model_suff + kPSep;
	#else
		folder += model + kPSep;
	#endif
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
	#else
		#ifdef _UNIFORM_DIAG
			folder += "UNIFORM" + kPSep;
		#endif
		switch(_mat_ensemble){
			case 0: folder += "GOE" + kPSep; break;
			case 1: folder += "GUE" + kPSep; break;
			case 2: folder += "CUE" + kPSep; break;
			default:
				folder += "GOE" + kPSep; 
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
// 					auto [val, temporary] = operators::sigma_z<double>(state, Ll, Lhalf );
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
// 						auto [val, temporary] = operators::sigma_z<double>(state, Ll, site );
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
// 		// 		auto [val1, tmp22] = operators::sigma_z<double>(state, Ll, site_1 );
// 		// 		auto [val2, tmp33] = operators::sigma_z<double>(state, Ll, site_2 );
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
// 		// 	auto [val1, tmp22] = operators::sigma_z<double>(state, Ll, Ll - 1 );
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