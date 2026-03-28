#include "includes/fermions_UI.hpp"

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

namespace Fermions_UI{

void ui::make_sim(){
    printAllOptions();
    
    this->ptr_to_model = this->create_new_model_pointer();

    // compare_energies();
    // return;
	
    
    
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
		purity();
		break;
	default:
		#define generate_scaling_array(name) arma::linspace(this->name, this->name + this->name##s * (this->name##n - 1), this->name##n)
        #define for_loop(param, var) for (auto& param : generate_scaling_array(var))

        for_loop(system_size, L){ 
            for_loop(t1x, t1){           
                for_loop(t2x, t2){ 
                    for_loop(V1x, V1){   
                        for_loop(V2x, V2)
        {
            this->L = system_size;
            this->t1 = t1x;
            this->t2 = t2x;
            this->V1 = V1x;
            this->V2 = V2x;
            
            this->site = this->L / 2.;
            const auto start_loop = std::chrono::system_clock::now();
            std::cout << " - - START NEW ITERATION:\t\t par = "; // simulation end
            printSeparated(std::cout, "\t", 16, true, this->L, this->t1, this->t2, this->V1, this->V2);
            
            auto kernel = [&](int k, int p, int zx)
                                    {
                                        this->syms.k_sym = k;
                                        this->syms.p_sym = p;
                                        this->syms.zx_sym = zx;
                                        
                                        this->reset_model_pointer();
                                        this->diagonal_matrix_elements();
                                        // this->diagonalize();
                                        // this->eigenstate_entanglement();
                                        // this->eigenstate_entanglement_degenerate();

                                    };
            // loopSymmetrySectors(kernel); continue;
            this->reset_model_pointer();
            auto Hamil = this->ptr_to_model->get_hamiltonian();
            this->l_steps = 0.05 * Hamil.n_cols;
            if(this->l_steps > 200) this->l_steps = 200;
            auto polfed = polfed::POLFED<ui::element_type>(Hamil, this->l_steps, this->l_bundle, -1, this->tol, 0.2, this->seed, true);
            continue;
            this->eigenstate_entanglement_degenerate(); 
            continue;

            diagonal_matrix_elements();
            std::cout << "\t\t - - - - - - FINISHED ITERATION IN : " << tim_s(start_loop) << " seconds\n\t\t\t Total time : " << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
        }}}}
            }
        std::cout << "Add default function" << std::endl;
	}
	std::cout << " - - - - - - FINISHED CALCULATIONS IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
}



// -------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------- MODEL DEPENDENT FUNCTIONS

// ------------------------------------------------ OVERRIDEN METHODS
arma::Col<ui::element_type> ui::cast_state(const arma::Col<ui::element_type>& state)
{
    auto U1sector = U1Hilbert(this->L, this->syms.N);
    arma::Col<ui::element_type> full_state(ULLPOW(this->L), arma::fill::zeros);
    for(int i = 0; i < U1sector.get_hilbert_space_size(); i++)
        full_state(U1sector(i)) = state(i);
    return full_state;
}

void ui::eigenstate_entanglement()
{
    clk::time_point start = std::chrono::system_clock::now();
	
	std::string dir = this->saving_dir + "Entropy" + kPSep + "Eigenstate2" + kPSep;
	createDirs(dir);
	
	std::string info = this->set_info();
	std::string filename = info;// + "_subsize=" + std::to_string(LA);
    
	size_t dim = this->ptr_to_model->get_hilbert_size();
	
    arma::vec emtpy_vec(1);
    if(dim == 0){
        emtpy_vec.save(arma::hdf5_name(dir + filename + ".hdf5", "nope"));
        return;
    }
    const size_t dim_cut = 7e0;
    if(dim < 5000)
        this->l_steps = u64(dim / 10.0);
    if(dim > dim_cut){
        double error = this->ptr_to_model->diag_sparse(this->l_steps, this->l_bundle, this->tol, this->seed);
        _assert_(error < 1e-10, "POLFED FAILED: Maximal Error = ");
	}else{
        this->ptr_to_model->diagonalization();
    }
    const int size = this->boundary_conditions == 2? (dim > dim_cut? this->l_steps : dim) : min(20, int(0.02 * dim));

    std::cout << " - - - - - - FINISHED DIAGONALIZATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
    
    start = std::chrono::system_clock::now();
    const arma::vec E = this->ptr_to_model->get_eigenvalues();
    double E_av = arma::mean(E);

    auto i = min_element(begin(E), end(E), [=](double x, double y) {
        return abs(x - E_av) < abs(y - E_av);
    });
    const long Eav_idx = i - begin(E);
    const long Emin = this->boundary_conditions == 2? 0 : Eav_idx - size / 2;
    printSeparated(std::cout, "\t", 20, true, arma::trace(this->ptr_to_model->get_hamiltonian()) / double(dim), E_av, Eav_idx, Emin, dim);

    const auto _hilbert = this->ptr_to_model->get_model_ref().get_hilbert_space();
    const auto U = _hilbert.symmetry_rotation();
    
    std::cout << " - - - - - - FINISHED CREATING SYMMETRY TRANSFORMATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
    start = std::chrono::system_clock::now();

    auto subsystem_sizes = arma::conv_to<arma::Col<int>>::from(arma::linspace(0, this->L - 1, this->L));
    // auto subsystem_sizes = arma::Col<int>( { int(this->L) / 2} );
    std::cout << subsystem_sizes.t() << std::endl;

    arma::mat S(size, subsystem_sizes.size(), arma::fill::zeros);
    arma::mat Scorr(size, subsystem_sizes.size(), arma::fill::zeros);
    arma::mat Scorr_site(size, subsystem_sizes.size(), arma::fill::zeros);
    arma::mat Purity(size, subsystem_sizes.size()+1, arma::fill::zeros);
    arma::mat Trace4(size, subsystem_sizes.size()+1, arma::fill::zeros);
    arma::mat Trace6(size, subsystem_sizes.size()+1, arma::fill::zeros);
    arma::vec NonGauss(size, arma::fill::zeros);

    // outer_threads = this->thread_number;
    // omp_set_num_threads(1);

// #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
    for(int n = 0; n < size; n++){
        clk::time_point start_n = std::chrono::system_clock::now();
        auto eigenstate = this->ptr_to_model->get_eigenState(Emin + n);
        arma::Col<element_type> state = U * eigenstate;
        
        // arma::cx_mat J_m_MB2(this->L, this->L, arma::fill::zeros);
        // for (u64 k = 0; k < dim; k++) {
        //     u64 base_state = _hilbert(k);

        //     for(int i = 0; i < this->L; i++)
        //     {
        //         auto [_spin, _] = operators::sigma_z<double>(base_state, this->L, i);
        //         if( _spin > 0){
        //             J_m_MB2(i, i) += std::conj(eigenstate(k)) * eigenstate(k);
        //         }
        //         for(int j = 0; j < this->L; j++)
        //         {
        //             if( j == i) continue;
        //             auto [val1, cm] = operators::fermions::spinless::anihilate<double>(base_state, this->L, j);
        //             auto [val2, cpcm] = operators::fermions::spinless::create<double>(cm, this->L, i);
        //             if(std::abs(val1 * val2) > 0)
        //             {
        //                 auto [_state_idx, sym_eig] = _hilbert.find_matrix_element(cpcm, _hilbert.get_norm(k));
        //                 auto _val_ = std::conj(eigenstate(_state_idx)) * eigenstate(k) * val1 * val2 * sym_eig;
        //                 J_m_MB2(i, j) += _val_;
        //                 // J_m_MB2(j, i) += std::conj(_val_);
        //             }
        //         }		
        //     }	
        // }

        arma::cx_mat J_m_MB(this->L, this->L, arma::fill::zeros);
        for(u64 base_state = 0; base_state < ULLPOW(this->L); base_state++)
        {
            for(int i = 0; i < this->L; i++)
            {
                auto [_spin, _] = operators::sigma_z<double>(base_state, this->L, i);
                if( _spin > 0){
                    J_m_MB(i, i) += std::conj(state(base_state)) * state(base_state);
                }
                for(int j = i+1; j < this->L; j++)
                {
                    auto [val1, cm] = operators::fermions::spinless::anihilate<double>(base_state, this->L, j);
                    auto [val2, cpcm] = operators::fermions::spinless::create<double>(cm, this->L, i);
                    if(std::abs(val1 * val2) > 0)
                    {
                        auto _val_ = std::conj(state(cpcm)) * state(base_state) * val1 * val2;
                        J_m_MB(i, j) += _val_;
                        J_m_MB(j, i) += std::conj(_val_);
                    }
                }		
            }	
        }
        // std::cout << arma::abs(J_m_MB) << std::endl;
        // std::cout << arma::abs(J_m_MB2) << std::endl << std::endl;

        J_m_MB = 2.0 * J_m_MB - arma::eye(this->L, this->L);

        auto lambdas = arma::eig_sym(J_m_MB);
        NonGauss(n) = QHS::single_particle::entanglement::vonNeumann(lambdas);

        Purity(n, subsystem_sizes.size()) = std::real( arma::trace(J_m_MB * J_m_MB) );
        Trace4(n, subsystem_sizes.size()) = std::real( arma::trace(J_m_MB * J_m_MB * J_m_MB * J_m_MB) );
        Trace6(n, subsystem_sizes.size()) = std::real( arma::trace(J_m_MB * J_m_MB * J_m_MB * J_m_MB * J_m_MB * J_m_MB) );

        for(int iiLA = 0; iiLA < subsystem_sizes.size(); iiLA++){
            int LA = subsystem_sizes[iiLA];
            S(n, iiLA) = entropy::schmidt_decomposition(state, LA, this->L);

            arma::uvec row_idx = arma::regspace<arma::uvec>(0, LA-1);
            arma::uvec col_idx = arma::regspace<arma::uvec>(0, LA-1);
            arma::cx_mat J_m_VA = J_m_MB.submat(row_idx, col_idx);
            auto lambdas = arma::eig_sym(J_m_VA);
            Scorr(n, iiLA) = QHS::single_particle::entanglement::vonNeumann(lambdas);
            
            double lambda = std::real( J_m_MB(LA, LA) );
            Scorr_site(n, iiLA) = QHS::single_particle::entanglement::vonNeumann_helper(lambda);

            Purity(n, iiLA) = std::real( arma::trace(J_m_VA * J_m_VA) );
            Trace4(n, iiLA) = std::real( arma::trace(J_m_VA * J_m_VA * J_m_VA * J_m_VA) );
            Trace6(n, iiLA) = std::real( arma::trace(J_m_VA * J_m_VA * J_m_VA * J_m_VA * J_m_VA * J_m_VA) );
        }
        if(this->boundary_conditions != 2)
            std::cout << " - - - - - - Finished state n = " << n << " in: " << tim_s(start_n) << " seconds - - - - - - " << std::endl; // simulation end
    }
    std::cout << " - - - - - - FINISHED ENTROPY CALCULATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
    
    // omp_set_num_threads(this->thread_number);
    // outer_threads = 1;
    
    E.save(arma::hdf5_name(dir + filename + ".hdf5", "energies"));
	S.save(arma::hdf5_name(dir + filename + ".hdf5", "entropy", arma::hdf5_opts::append));
    Scorr.save(arma::hdf5_name(dir + filename + ".hdf5", "entropy_corr_mat", arma::hdf5_opts::append));
    Scorr_site.save(arma::hdf5_name(dir + filename + ".hdf5", "entropy_single_site_corr_mat", arma::hdf5_opts::append));
    NonGauss.save(arma::hdf5_name(dir + filename + ".hdf5", "Non-Gaussianity", arma::hdf5_opts::append));
    Purity.save(arma::hdf5_name(dir + filename + ".hdf5", "Purity", arma::hdf5_opts::append));
    Trace4.save(arma::hdf5_name(dir + filename + ".hdf5", "Trace4", arma::hdf5_opts::append));
    Trace6.save(arma::hdf5_name(dir + filename + ".hdf5", "Trace6", arma::hdf5_opts::append));
    arma::uvec({dim}).save(arma::hdf5_name(dir + filename + ".hdf5", "D", arma::hdf5_opts::append));
}

void ui::purity()
{
    clk::time_point start = std::chrono::system_clock::now();
	
	std::string dir = this->saving_dir + "Purity" + kPSep;
	createDirs(dir);
	
	std::string info = this->set_info();
	std::string filename = info;// + "_subsize=" + std::to_string(LA);
    
	size_t dim = this->ptr_to_model->get_hilbert_size();
	
    arma::vec emtpy_vec(1);
    if(dim == 0){
        emtpy_vec.save(arma::hdf5_name(dir + filename + ".hdf5", "nope"));
        return;
    }
    const size_t dim_cut = 7e4;

    if(dim > dim_cut){
        double error = this->ptr_to_model->diag_sparse(this->l_steps, this->l_bundle, this->tol, this->seed);
        _assert_(error < 1e-10, "POLFED FAILED: Maximal Error = ");
	}else{
        this->ptr_to_model->diagonalization();
    }
    // const int size = min(500, int(0.1 * dim));
    const int size = this->boundary_conditions == 2? (dim > dim_cut? this->l_steps : dim) : min(20, int(0.02 * dim));

    std::cout << " - - - - - - FINISHED DIAGONALIZATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
    
    start = std::chrono::system_clock::now();
    const arma::vec E = this->ptr_to_model->get_eigenvalues();
    double E_av = arma::mean(E);

    auto i = min_element(begin(E), end(E), [=](double x, double y) {
        return abs(x - E_av) < abs(y - E_av);
    });
    const long Eav_idx = i - begin(E);
    const long Emin = this->boundary_conditions == 2? 0 : Eav_idx - size / 2;
    printSeparated(std::cout, "\t", 20, true, arma::trace(this->ptr_to_model->get_hamiltonian()) / double(dim), E_av, Eav_idx, Emin, dim);


    const auto _hilbert = this->ptr_to_model->get_model_ref().get_hilbert_space();
    const auto U = _hilbert.symmetry_rotation();
    
    std::cout << " - - - - - - FINISHED CREATING SYMMETRY TRANSFORMATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
    start = std::chrono::system_clock::now();

    auto subsystem_sizes = arma::conv_to<arma::Col<int>>::from(arma::linspace(0, this->L - 1, this->L));
    auto qs = arma::linspace(0.5, 3.0, 26);
    // auto subsystem_sizes = arma::Col<int>( { int(this->L) / 2} );
    std::cout << subsystem_sizes.t() << std::endl;

    arma::mat Scorr(size, subsystem_sizes.size(), arma::fill::zeros);
    arma::mat Scorr_site(size, subsystem_sizes.size(), arma::fill::zeros);
    arma::mat Purity(size, subsystem_sizes.size()+1, arma::fill::zeros);
    arma::mat Trace4(size, subsystem_sizes.size()+1, arma::fill::zeros);
    arma::mat Trace6(size, subsystem_sizes.size()+1, arma::fill::zeros);
    arma::vec NonGauss(size, arma::fill::zeros);
    
    arma::mat part_ratio(size, qs.size(), arma::fill::zeros);
    arma::mat info_ent(size, qs.size(), arma::fill::zeros);

    // outer_threads = this->thread_number;
    // omp_set_num_threads(1);

// #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
    for(int n = 0; n < size; n++){
        clk::time_point start_n = std::chrono::system_clock::now();
        // int idx = 0;
        // if(dim < dim_cut) idx = Emin;
        arma::Col<element_type> eigenstate = this->ptr_to_model->get_eigenState(Emin + n);
        arma::Col<element_type> state = U * eigenstate;
        
        // arma::cx_mat J_m_MB2(this->L, this->L, arma::fill::zeros);
        // for (u64 k = 0; k < dim; k++) {
        //     u64 base_state = _hilbert(k);

        //     for(int i = 0; i < this->L; i++)
        //     {
        //         auto [_spin, _] = operators::sigma_z<double>(base_state, this->L, i);
        //         if( _spin > 0){
        //             J_m_MB2(i, i) += std::conj(eigenstate(k)) * eigenstate(k);
        //         }
        //         for(int j = 0; j < this->L; j++)
        //         {
        //             if( j == i) continue;
        //             auto [val1, cm] = operators::fermions::spinless::anihilate<double>(base_state, this->L, j);
        //             auto [val2, cpcm] = operators::fermions::spinless::create<double>(cm, this->L, i);
        //             if(std::abs(val1 * val2) > 0)
        //             {
        //                 auto [_state_idx, sym_eig] = _hilbert.find_matrix_element(cpcm, _hilbert.get_norm(k));
        //                 auto _val_ = std::conj(eigenstate(_state_idx)) * eigenstate(k) * val1 * val2 * sym_eig;
        //                 J_m_MB2(i, j) += _val_;
        //                 // J_m_MB2(j, i) += std::conj(_val_);
        //             }
        //         }		
        //     }	
        // }

        arma::cx_mat J_m_MB(this->L, this->L, arma::fill::zeros);
        for(u64 base_state = 0; base_state < ULLPOW(this->L); base_state++)
        {
        // #pragma omp parallel for
            for(int i = 0; i < this->L; i++)
            {
                auto [_spin, _] = operators::sigma_z<double>(base_state, this->L, i);
                if( _spin > 0){
                    J_m_MB(i, i) += std::conj(state(base_state)) * state(base_state);
                }
                for(int j = i+1; j < this->L; j++)
                {
                    auto [val1, cm] = operators::fermions::spinless::anihilate<double>(base_state, this->L, j);
                    auto [val2, cpcm] = operators::fermions::spinless::create<double>(cm, this->L, i);
                    if(std::abs(val1 * val2) > 0)
                    {
                        auto _val_ = std::conj(state(cpcm)) * state(base_state) * val1 * val2;
                        J_m_MB(i, j) += _val_;
                        J_m_MB(j, i) += std::conj(_val_);
                    }
                }		
            }	
        }
        // std::cout << arma::abs(J_m_MB) << std::endl;
        // std::cout << arma::abs(J_m_MB2) << std::endl << std::endl;

        J_m_MB = 2.0 * J_m_MB - arma::eye(this->L, this->L);

        auto lambdas = arma::eig_sym(J_m_MB);
        NonGauss(n) = QHS::single_particle::entanglement::vonNeumann(lambdas);

        Purity(n, subsystem_sizes.size()) = std::real( arma::trace(J_m_MB * J_m_MB) );
        Trace4(n, subsystem_sizes.size()) = std::real( arma::trace(J_m_MB * J_m_MB * J_m_MB * J_m_MB) );
        Trace6(n, subsystem_sizes.size()) = std::real( arma::trace(J_m_MB * J_m_MB * J_m_MB * J_m_MB * J_m_MB * J_m_MB) );
    
    // #pragma omp parallel for
        for(int iiLA = 0; iiLA < subsystem_sizes.size(); iiLA++){
            int LA = subsystem_sizes[iiLA];

            arma::uvec row_idx = arma::regspace<arma::uvec>(0, LA-1);
            arma::uvec col_idx = arma::regspace<arma::uvec>(0, LA-1);
            arma::cx_mat J_m_VA = J_m_MB.submat(row_idx, col_idx);
            auto lambdas = arma::eig_sym(J_m_VA);
            Scorr(n, iiLA) = QHS::single_particle::entanglement::vonNeumann(lambdas);
            
            double lambda = std::real( J_m_MB(LA, LA) );
            Scorr_site(n, iiLA) = QHS::single_particle::entanglement::vonNeumann_helper(lambda);

            Purity(n, iiLA) = std::real( arma::trace(J_m_VA * J_m_VA) );
            Trace4(n, iiLA) = std::real( arma::trace(J_m_VA * J_m_VA * J_m_VA * J_m_VA) );
            Trace6(n, iiLA) = std::real( arma::trace(J_m_VA * J_m_VA * J_m_VA * J_m_VA * J_m_VA * J_m_VA) );
        }
        // std::cout << " - - - - - - Finished corr_mat in state n = " << n << " in: " << tim_s(start_n) << " seconds - - - - - - " << std::endl; // simulation end
        // start_n = std::chrono::system_clock::now();
    #pragma omp parallel for
        for(int iq = 0; iq < qs.size(); iq++)
        {
            if(qs(iq) == 1)
            {
                double _pr_ = 0;
                for (int k = 0; k < eigenstate.size(); k++) {
                    auto c_k = eigenstate(k);
                    double value = std::abs(std::conj(c_k) * c_k);
                    _pr_ += (std::abs(value) > 0) ? -value * std::log(value) : 0;
                }
                part_ratio(n, iq) = arma::norm(eigenstate);
                info_ent(n, iq) = _pr_;
            }
            else{
                double _pr_ = statistics::participation_ratio(eigenstate, qs(iq));
                part_ratio(n, iq) = _pr_;
                info_ent(n, iq) = std::log(_pr_) / (1 - qs(iq));
            }
        }
        std::cout << " - - - - - - Finished state n = " << n << " in: " << tim_s(start_n) << " seconds - - - - - - " << std::endl; // simulation end
    }
    std::cout << " - - - - - - FINISHED CORR MAT AND IPR CALCULATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
    
    // omp_set_num_threads(this->thread_number);
    // outer_threads = 1;
    
    E.save(arma::hdf5_name(dir + filename + ".hdf5", "energies"));
	// S.save(arma::hdf5_name(dir + filename + ".hdf5", "entropy", arma::hdf5_opts::append));
    subsystem_sizes.save(arma::hdf5_name(dir + filename + ".hdf5", "subsystem_sizes", arma::hdf5_opts::append));
    Scorr.save(arma::hdf5_name(dir + filename + ".hdf5", "entropy_corr_mat", arma::hdf5_opts::append));
    Scorr_site.save(arma::hdf5_name(dir + filename + ".hdf5", "entropy_single_site_corr_mat", arma::hdf5_opts::append));
    NonGauss.save(arma::hdf5_name(dir + filename + ".hdf5", "Non-Gaussianity", arma::hdf5_opts::append));
    Purity.save(arma::hdf5_name(dir + filename + ".hdf5", "Purity", arma::hdf5_opts::append));
    Trace4.save(arma::hdf5_name(dir + filename + ".hdf5", "Trace4", arma::hdf5_opts::append));
    Trace6.save(arma::hdf5_name(dir + filename + ".hdf5", "Trace6", arma::hdf5_opts::append));
    arma::uvec({dim}).save(arma::hdf5_name(dir + filename + ".hdf5", "D", arma::hdf5_opts::append));

    qs.save(arma::hdf5_name(dir + filename + ".hdf5", "qs", arma::hdf5_opts::append));
    part_ratio.save(arma::hdf5_name(dir + filename + ".hdf5", "part_ratio", arma::hdf5_opts::append));
    info_ent.save(arma::hdf5_name(dir + filename + ".hdf5", "info_ent", arma::hdf5_opts::append));
}
/// @brief 
/// @param skip 
/// @param sep 
/// @return 
std::string ui::set_info(std::vector<std::string> skip, std::string sep) const
{
        std::string name = "L=" + std::to_string(this->L) + \
            ",N=" + std::to_string(this->syms.N) + \
            ",t1=" + to_string_prec(this->t1) + \
            ",t2=" + to_string_prec(this->t2) + \
            ",V1=" + to_string_prec(this->V1) + \
            ",V2=" + to_string_prec(this->V2);
        
        if(this->boundary_conditions == 0)      name += ",k=" + std::to_string(this->syms.k_sym);
        if(this->k_real_sec(this->syms.k_sym))  name += ",p=" + std::to_string(this->syms.p_sym);
        if(this->use_flip_X())                  name += ",zx=" + std::to_string(this->syms.zx_sym);
        
        

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
    v_1d<double> k_sectors;
	v_1d<std::string> symms;
    auto kernel = [&](int k, int p, int zx)
    {
        auto symmetric_model = std::make_unique<QHS::QHamSolver<Fermions>>(this->boundary_conditions, this->L, this->syms.N, this->t1, this->t2, this->V1, this->V2, this->mu, k, p, zx, 1);
        if(symmetric_model->get_hilbert_size() > 0){
            symmetric_model->diagonalization(false);
            arma::vec E = symmetric_model->get_eigenvalues();
            
            printSeparated(std::cout, "\t", 20, true, "Sector:", k, p, zx, "Gap Ratio=", statistics::eigenlevel_statistics(E));
            std::vector<double> ksec(E.size(), k);
            k_sectors.insert(k_sectors.end(), std::make_move_iterator(ksec.begin()), std::make_move_iterator(ksec.end()));
            
            Esym.insert(Esym.end(), std::make_move_iterator(E.begin()), std::make_move_iterator(E.end()));
            v_1d<std::string> temp_str = v_1d<std::string>(E.size(), "k=" + std::to_string(k) + ",p=" + to_string(p) + ",zx=" + to_string(zx));
            symms.insert(symms.end(), std::make_move_iterator(temp_str.begin()), std::make_move_iterator(temp_str.end()));
        }
    };
    loopSymmetrySectors(kernel);

    auto full_model = std::make_unique<QHS::QHamSolver<Fermions>>(this->boundary_conditions, this->L, this->syms.N, this->t1, this->t2, this->V1, this->V2, this->mu, this->syms.k_sym, this->syms.p_sym, this->syms.zx_sym, 0);
    full_model->diagonalization(true);
    arma::vec E_dis = full_model->get_eigenvalues();
    arma::Mat<element_type> V = full_model->get_eigenvectors();

    auto U1sector = U1Hilbert(this->L, this->syms.N);
    auto Jsh = (QOps::_spin_flip_x_symmetry<QOps::particle::fermion>(this->L, this->syms.zx_sym)).to_reduced_matrix(U1sector);
    auto P = (QOps::_parity_symmetry<QOps::particle::fermion>(this->L, this->syms.p_sym)).to_reduced_matrix(U1sector);
    auto T = (QOps::_translation_symmetry<QOps::particle::fermion>(this->L, this->syms.k_sym)).to_reduced_matrix(U1sector);
    arma::cx_vec Jsh_value(E_dis.size()), P_value(E_dis.size()), T_value(E_dis.size());
    // for(u64 k = 0; k < E_dis.size(); k++){
    //     Jsh_value(k) = arma::cdot(V.col(k), Jsh * V.col(k));
    //     P_value(k) = arma::cdot(V.col(k), P * V.col(k));
    //     T_value(k) = arma::cdot(V.col(k), T * V.col(k));
    // }
    auto permut = sort_permutation(Esym, [](const double a, const double b)
								   { return a < b; });
	apply_permutation(Esym, permut);
	apply_permutation(symms, permut);
    apply_permutation(k_sectors, permut);
	std::cout << std::endl << Esym.size() << std::endl << E_dis.size() << std::endl;
	printSeparated(std::cout, "\t", 20, true, "symmetry sector", "Energy sym", "Energy total", "difference", "Shiba eigenvalue", "Parity eigenvalue", "Translation eigenvalue");
    // for( int q = 0; q < this->L; q++)
    {
        for (int k = 0; k < min((int)E_dis.size(), (int)Esym.size()); k++){
            // if(k_sectors[k] != q) continue;
            if(std::abs(Esym[k] - E_dis(k)) > 1e-18)
                printSeparated(std::cout, "\t", 20, true, symms[k], Esym[k], E_dis(k), Esym[k] - E_dis(k), std::abs(Jsh_value(k)), P_value(k), T_value(k));
        }
    }
}

/// @brief Compaer full hamiltonian to the reconstructed one from symmetry sectors
void ui::compare_hamiltonian()
{   
    // auto full_model = std::make_unique<QHS::QHamSolver<XXZ>>(this->boundary_conditions, this->L, this->t1, this->t2, this->V1, this->V2, this->hz, this->syms.Sz);
    // arma::sp_mat Hfull = full_model->get_hamiltonian();
    // const u64 dim = full_model->get_hilbert_size();
    // arma::sp_cx_mat H(dim, dim);
    // auto kernel = [&](int k, int p, int zx)
    // {
    //     auto symmetric_model = std::make_unique<QHS::QHamSolver<XXZsym>>(this->boundary_conditions, this->L, this->t1, this->t2, this->V1, this->V2, this->hz, k, p, zx, this->syms.Sz);
    //     auto U = symmetric_model->get_model_ref().get_hilbert_space().symmetry_rotation();
    //     arma::sp_cx_mat Hsym = cast_cx_sparse(symmetric_model->get_hamiltonian());
    //     H += U * Hsym * U.t();
    // };
    // loopSymmetrySectors(kernel);
    // arma::sp_cx_mat res = cast_cx_sparse(Hfull) - cast_cx_sparse(H);
	// printSeparated(std::cout, "\t", 20, true, "col", "row", "diff", "\t", "sym H");
    // for(int i = 0; i < dim; i++){
    //     for(int j = 0; j < dim; j++){
    //         cpx val = res(i, j);
    //         if(std::abs(val) > 1e-14)
    //             printSeparated(std::cout, "\t", 15, true, i, j, val, "\t", H(i,j));
    //     }
    // }
}

// /// @brief 
// void ui::check_symmetry_generators()
// {
//     v_1d<QOps::genOp> sym_group;
//     // parity symmetry
//     sym_group.emplace_back(QOps::_parity_symmetry(this->L, this->syms.p_sym));

//     // spin flips
//     if(this->hz == 0 && this->syms.Sz == 0.0)
//         sym_group.emplace_back(QOps::_spin_flip_x_symmetry(this->L, this->syms.zx_sym));
    
//     QHS::point_symmetric hilbert( this->L, sym_group, this->boundary_conditions, this->syms.k_sym, 0);
//     auto group = hilbert.get_symmetry_group();
//     for(auto& idx : {1, 130, 33, 71, 756}){
//         for(auto& G : group){
//             auto [state, val] = G(idx);
//             printSeparated(std::cout, "\t", 16, true, std::vector<bool>(this->L, idx), std::vector<bool>(this->L, state), val);
//         }
//         std::cout << std::endl;
//     }
// }


/// @brief Create energy current for this specific model
arma::sp_mat ui::energy_current(){

    const size_t dim_max = ULLPOW(this->L);
    auto check_spin = QOps::__builtins::get_digit(this->L);
    _assert_(this->t2 == 0.0 && this->V2 == 0, "Energy current implemented only for integrable case, no nearest neighbour terms yet!");
    double Jx = this->t1;
    double Jy = this->t1;
    double Jz = this->V1;
    arma::sp_mat jE(dim_max, dim_max);
    // printSeparated(std::cout, "\t", 20, true, "Start Current", Jx, Jy, Jz);
    // for(int i = 0; i < this->L; i++)
    // {
    //     int nei = (this->boundary_conditions)? i + 1 : (i + 1)%this->L;
    //     int nei2 = (this->boundary_conditions)? i + 2 : (i + 2)%this->L;
    //     // printSeparated(std::cout, "\t", 20, true, "site", i, nei, nei2);
    //     if(nei < this->L && nei2 < this->L){
    //         for(long k = 0; k < dim_max; k++)
    //         {
    //             double Si = double(check_spin(k, i)) - 0.5;
    //             double Snei = double(check_spin(k, nei)) - 0.5;
    //             double Snei2 = double(check_spin(k, nei2)) - 0.5;
    //             {
    //                 auto [val, state_tmp]   = operators::sigma_x(k, this->L, i);
    //                 auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei2);
    //                 jE(new_idx, k) += std::imag(Jx * Jy * Snei * val * val2);
    //             }{
    //                 auto [val, state_tmp]   = operators::sigma_x(k, this->L, nei2);
    //                 auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, i);
    //                 jE(new_idx, k) -= std::imag(Jx * Jy * Snei * val * val2);
    //             }{
    //                 auto [val, state_tmp]   = operators::sigma_x(k, this->L, nei);
    //                 auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, i);
    //                 jE(new_idx, k) += std::imag(Jz * Jy * Snei2 * val * val2);
    //             }{
    //                 auto [val, state_tmp]   = operators::sigma_x(k, this->L, i);
    //                 auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei);
    //                 jE(new_idx, k) -= std::imag(Jz * Jx * Snei2 * val * val2);
    //             }{
    //                 auto [val, state_tmp]   = operators::sigma_x(k, this->L, nei2);
    //                 auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei);
    //                 jE(new_idx, k) += std::imag(Jz * Jx * Si * val * val2);
    //             }{
    //                 auto [val, state_tmp]   = operators::sigma_x(k, this->L, nei);
    //                 auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei2);
    //                 jE(new_idx, k) -= std::imag(Jz * Jy * Si * val * val2);
    //             }
    //         }
    //     }
    // }

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
    _assert_(this->t2 == 0.0 && this->V2 == 0, "Energy current implemented only for integrable case, no nearest neighbour terms yet!");
    double Jx = this->t1;
    double Jy = this->t1;
    double Jz = this->V1;
   
    // int nei = (this->boundary_conditions)? i + 1 : (i + 1)%this->L;
    // int nei2 = (this->boundary_conditions)? i + 2 : (i + 2)%this->L;
    // if(nei < this->L && nei2 < this->L){
    //     double Si = double(check_spin(k, i)) - 0.5;
    //     double Snei = double(check_spin(k, nei)) - 0.5;
    //     double Snei2 = double(check_spin(k, nei2)) - 0.5;
    //     {
    //         auto [val, state_tmp]   = operators::sigma_x(k, this->L, i);
    //         auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei2);
    //         // jE(new_idx, k) += std::imag(Jx * Jy * Snei * val * val2);
    //         result += my_conjungate(state1(new_idx)) * std::imag(Jx * Jy * Snei * val * val2) * state2(k);
    //     }{
    //         auto [val, state_tmp]   = operators::sigma_x(k, this->L, nei2);
    //         auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, i);
    //         // jE(new_idx, k) -= std::imag(Jx * Jy * Snei * val * val2);
    //         result -= my_conjungate(state1(new_idx)) * std::imag(Jx * Jy * Snei * val * val2) * state2(k);
    //     }{
    //         auto [val, state_tmp]   = operators::sigma_x(k, this->L, nei);
    //         auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, i);
    //         // jE(new_idx, k) += std::imag(Jz * Jy * Snei2 * val * val2);
    //         result += my_conjungate(state1(new_idx)) * std::imag(Jz * Jy * Snei2 * val * val2) * state2(k);
    //     }{
    //         auto [val, state_tmp]   = operators::sigma_x(k, this->L, i);
    //         auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei);
    //         // jE(new_idx, k) -= std::imag(Jz * Jx * Snei2 * val * val2);
    //         result -= my_conjungate(state1(new_idx)) * std::imag(Jz * Jx * Snei2 * val * val2) * state2(k);
    //     }{
    //         auto [val, state_tmp]   = operators::sigma_x(k, this->L, nei2);
    //         auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei);
    //         // jE(new_idx, k) += std::imag(Jz * Jx * Si * val * val2);
    //         result += my_conjungate(state1(new_idx)) * std::imag(Jz * Jx * Si * val * val2) * state2(k);
    //     }{
    //         auto [val, state_tmp]   = operators::sigma_x(k, this->L, nei);
    //         auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei2);
    //         // jE(new_idx, k) -= std::imag(Jz * Jy * Si * val * val2);
    //         result -= my_conjungate(state1(new_idx)) * std::imag(Jz * Jy * Si * val * val2) * state2(k);
    //     }
    // }

    return result;
}

// -------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------- IMPLEMENTATION OF UI

/// @brief Create unique pointer to model with current parameters in class
typename ui::model_pointer ui::create_new_model_pointer(){
    return std::make_unique<QHS::QHamSolver<Fermions>>(this->boundary_conditions, this->L, this->syms.N, this->t1, this->t2, this->V1, this->V2, this->mu, this->syms.k_sym, this->syms.p_sym, this->syms.zx_sym);
}

/// @brief Reset member unique pointer to model with current parameters in class
void ui::reset_model_pointer(){
    return this->ptr_to_model.reset(new QHS::QHamSolver<Fermions>(this->boundary_conditions, this->L, this->syms.N, this->t1, this->t2, this->V1, this->V2, this->mu, this->syms.k_sym, this->syms.p_sym, this->syms.zx_sym));
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
    user_interface_sym<Fermions>::parse_cmd_options(argc, argv);

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
    set_param(t1);
    set_param(t2);
    set_param(V1);
    set_param(V2);

	// checmical potential
	choosen_option = "-mu";
	this->set_option(this->mu, argv, choosen_option, true);

    //<! SYMMETRIES
    choosen_option = "-k";
    this->set_option(this->syms.k_sym, argv, choosen_option);

    choosen_option = "-p";
    this->set_option(this->syms.p_sym, argv, choosen_option);
    
    choosen_option = "-zx";
    this->set_option(this->syms.zx_sym, argv, choosen_option);
    
    choosen_option = "-N";
    this->set_option(this->syms.N, argv, choosen_option);

    //<! FOLDER
    std::string folder = "." + kPSep + "results" + kPSep;
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
    user_interface_sym<Fermions>::set_default();
    this->t1 = 1.0;
	this->t1s = 0.0;
	this->t1n = 1;
    this->t2 = 0.0;
	this->t2s = 0.0;
	this->t2n = 1;

    this->V1 = 1.0;
	this->V1s = 0.0;
	this->V1n = 1;
    this->V2 = 0.0;
	this->V2s = 0.0;
	this->V2n = 1;
    this->mu = 0;

    this->syms.k_sym = 0;
    this->syms.p_sym = 1;
    this->syms.zx_sym = 1;
    this->syms.N = this->L / 2;
}

/// @brief 
void ui::print_help() const {
    user_interface_sym<Fermions>::print_help();

    printf(" Flags for XXZ model:\n");
    printSeparated(std::cout, "\t", 20, true, "-t1", "(double)", "nearest neighbour coupling strength");
    printSeparated(std::cout, "\t", 20, true, "-t1s", "(double)", "step in nearest neighbour coupling strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-t1n", "(int)", "number of nearest neighbour couplings in the sweep");
    printSeparated(std::cout, "\t", 20, true, "-t2", "(double)", "next-nearest neighbour coupling strength");
    printSeparated(std::cout, "\t", 20, true, "-t2s", "(double)", "step in next-nearest neighbour coupling strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-t2n", "(int)", "number of next-nearest neighbour couplings in the sweep");
    
    printSeparated(std::cout, "\t", 20, true, "-V1", "(double)", "nearest neighbour interaction strength");
    printSeparated(std::cout, "\t", 20, true, "-V1s", "(double)", "step in nearest neighbour interaction strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-V1n", "(int)", "number of nearest neighbour interaction in the sweep");
    printSeparated(std::cout, "\t", 20, true, "-V2", "(double)", "next-nearest neighbour interaction strength");
    printSeparated(std::cout, "\t", 20, true, "-V2s", "(double)", "step in next-nearest neighbour interaction strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-V2n", "(int)", "number of next-nearest neighbour interaction in the sweep");
    
    printSeparated(std::cout, "\t", 20, true, "-mu", "(double)", "chemical potential shift");

    printSeparated(std::cout, "\t", 20, true, "-k", "(int)", "quasimomentum symmetry sector");
    printSeparated(std::cout, "\t", 20, true, "-p", "(int)", "parity symmetry sector");
    printSeparated(std::cout, "\t", 20, true, "-zx", "(int)", "spin flip in X direction symmetry sector");
    printSeparated(std::cout, "\t", 20, true, "-Sz", "(float)", "magnetization sector");
	std::cout << std::endl;
}

/// @brief 
void ui::printAllOptions() const{
    user_interface_sym<Fermions>::printAllOptions();
    std::cout << "H = \u03A3_r \u03A3_i t_r[ c^+_i c_i+1 + h.c ] + \u0394_r n_i n_i+1" << std::endl << std::endl;

	std::cout << "------------------------------ CHOSEN XXZ OPTIONS:" << std::endl;
    std::cout 
		  << "t1  = " << this->t1 << std::endl
		  << "t1n = " << this->t1n << std::endl
		  << "t1s = " << this->t1s << std::endl

		  << "t2  = " << this->t2 << std::endl
		  << "t2n = " << this->t2n << std::endl
		  << "t2s = " << this->t2s << std::endl

		  << "V1  = " << this->V1 << std::endl
		  << "V1n = " << this->V1n << std::endl
		  << "V1s = " << this->V1s << std::endl
		  
		  << "V2  = " << this->V2 << std::endl
		  << "V2n = " << this->V2n << std::endl
		  << "V2s = " << this->V2s << std::endl

		  << "mu = " << this->mu << std::endl;

    if(this->boundary_conditions == 0)        std::cout << "k  = " << this->syms.k_sym << std::endl;
    if(this->k_real_sec(this->syms.k_sym))    std::cout << "p  = " << this->syms.p_sym << std::endl;
    if(this->use_flip_X())                    std::cout << "zx  = " << this->syms.zx_sym << std::endl;
    std::cout << "N = " << this->syms.N << std::endl;

    std::cout << std::endl;
    printSeparated(std::cout, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
}   





    // arma::vec empty;
    // empty.save(arma::hdf5_name(this->saving_dir + this->set_info({"k", "p", "zx", "zz"}) + ".hdf5", "(empty)"));
    // auto kernel = [&](int k, int p, int zx, int zz)
    // {
    //     auto symmetric_model = std::make_unique<QHamSolver<XXZsym>>(this->boundary_conditions, this->L, this->t1, this->t2, this->V1, this->V2, this->eta1, this->eta2,
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