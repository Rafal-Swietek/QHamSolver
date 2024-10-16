#include "includes/ConstrainedXXZ_UI.hpp"

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

namespace ConstrainedXXZ_UI{



void ui::make_sim(){
    printAllOptions();

    this->ptr_to_model = create_new_model_pointer();

    compare_energies(); return;
    
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
	default:
		#define generate_scaling_array(name) arma::linspace(this->name, this->name + this->name##s * (this->name##n - 1), this->name##n)
        #define for_loop(param, var) for (auto& param : generate_scaling_array(var))

        for_loop(system_size, L){ 
            for_loop(Jx, J){    
        {
            this->L = system_size;
            this->J = Jx;
            this->site = this->L / 2.;
            const auto start_loop = std::chrono::system_clock::now();
            std::cout << " - - START NEW ITERATION:\t\t par = "; // simulation end
            printSeparated(std::cout, "\t", 16, true, this->L, this->J);
            
            auto kernel = [&](int k, int p)
                                    {
                                        v_1d<QOps::genOp> symmetry_generators;
                                        this->syms.k_sym = k;
                                        this->syms.p_sym = p;
                                        auto Translate = QOps::__builtins::translation(this->L, 1);
                                        auto flip = QOps::__builtins::spin_flip_x(this->L);
                                        auto some_kernel = [&Translate, &flip](u64 n){
                                            n = std::get<0>(flip(n));
                                            return !( (n) & std::get<0>( Translate(n) ) );
                                        };
                                        symmetry_generators.emplace_back(QOps::_parity_symmetry(this->L, this->syms.p_sym));
                                        auto _hilbert_GoldenChain = QHS::constrained_hilbert_space(this->L, std::move(some_kernel));
                                        auto _second_hilbert = QHS::point_symmetric( this->L, symmetry_generators, this->boundary_conditions, this->syms.k_sym, 0);
                                                                                        
                                        
                                        auto _hilbert_space = tensor(_second_hilbert, _hilbert_GoldenChain);
                                        // auto _hilbert_space = _second_hilbert;
                                        u64 dim = _hilbert_space.get_hilbert_space_size();
                                        // this->reset_model_pointer();
                                        // this->diagonal_matrix_elements();
                                        // this->diagonalize();
                                        // this->eigenstate_entanglement();
                                        // this->eigenstate_entanglement_degenerate();
                                        printSeparated(std::cout, "\t", 16, true, this->L, k, p, dim);
                                    };
            loopSymmetrySectors(kernel); continue;
            this->reset_model_pointer();

            this->eigenstate_entanglement_degenerate(); 
            continue;

            diagonal_matrix_elements();
            std::cout << "\t\t - - - - - - FINISHED ITERATION IN : " << tim_s(start_loop) << " seconds\n\t\t\t Total time : " << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
        }}}
        std::cout << "Add default function" << std::endl;
	}
	std::cout << " - - - - - - FINISHED CALCULATIONS IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
}



// -------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------- MODEL DEPENDENT FUNCTIONS

void ui::eigenstate_entanglement(){
    clk::time_point start = std::chrono::system_clock::now();
	
	std::string dir = this->saving_dir + "Entropy" + kPSep + "Eigenstate" + kPSep;
	createDirs(dir);
	
	std::string info = this->set_info();
	std::string filename = info;// + "_subsize=" + std::to_string(LA);
    
    int LA = this->site;
	size_t dim = this->ptr_to_model->get_hilbert_size();
	
    arma::vec emtpy_vec(1);
    if(dim == 0){
        emtpy_vec.save(arma::hdf5_name(dir + filename + ".hdf5", "nope"));
        return;
    }
    
    #ifdef ARMA_USE_SUPERLU
        const int size = this->ch? 500 : dim;
        if(this->ch){
            this->ptr_to_model->hamiltonian();
            this->ptr_to_model->diag_sparse(true);
        } else
            this->ptr_to_model->diagonalization();
    
    #else
        const int size = dim;
        this->ptr_to_model->diagonalization();
    #endif

    std::cout << " - - - - - - FINISHED DIAGONALIZATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
    
    start = std::chrono::system_clock::now();
    const arma::vec E = this->ptr_to_model->get_eigenvalues();
    
    const auto U = this->ptr_to_model->get_model_ref().get_hilbert_space().symmetry_rotation();
    
    std::cout << " - - - - - - FINISHED CREATING SYMMETRY TRANSFORMATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
    start = std::chrono::system_clock::now();

    auto subsystem_sizes = arma::conv_to<arma::Col<int>>::from(arma::linspace(0, this->L / 2, this->L / 2 + 1));
    // auto subsystem_sizes = arma::Col<int>( { int(this->L) / 2} );
    std::cout << subsystem_sizes.t() << std::endl;

    // outer_threads = this->thread_number;
    // omp_set_num_threads(1);
    // std::cout << th_num << "\t\t" << omp_get_num_threads() << std::endl;
    
        // auto start_LA = std::chrono::system_clock::now();
    double E_av = arma::mean(E);
    auto i = min_element(begin(E), end(E), [=](double x, double y) {
        return abs(x - E_av) < abs(y - E_av);
    });
    const long E_av_idx = i - begin(E);
    long int E_min = dim > 1e5? 0 : E_av_idx - 5;
    long int E_max = dim > 1e5? dim : E_av_idx + 5;
    printSeparated(std::cout, "\t", 16, true, "Mean Energy:", E_av, E_av_idx, E_min, E_max);
    const auto new_size = E_max - E_min;
    arma::mat S(new_size, subsystem_sizes.size(), arma::fill::zeros);
    arma::vec Ecut(new_size, arma::fill::zeros);
// #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
    for(int n = E_min; n < E_max; n++){
        Ecut(n - E_min) = E(n);
        auto eigenstate = this->ptr_to_model->get_eigenState(n);
        arma::Col<element_type> state = U * eigenstate;
        
        for(int iiLA = 0; iiLA < subsystem_sizes.size(); iiLA++){
            int LA = subsystem_sizes[iiLA];
            S(n - E_min, iiLA) = entropy::schmidt_decomposition(state, LA, this->L);
        }
    }
    std::cout << " - - - - - - FINISHED ENTROPY CALCULATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
    
    // omp_set_num_threads(this->thread_number);
    // outer_threads = 1;
    
    E.save(arma::hdf5_name(dir + filename + ".hdf5", "energies"));
    Ecut.save(arma::hdf5_name(dir + filename + ".hdf5", "energies cropped"));
	S.save(arma::hdf5_name(dir + filename + ".hdf5", "entropy", arma::hdf5_opts::append));
}

/// @brief 
/// @param skip 
/// @param sep 
/// @return 
std::string ui::set_info(std::vector<std::string> skip, std::string sep) const
{
        std::string name = "L=" + std::to_string(this->L) + \
            ",J=" + to_string_prec(this->J) + \
            ",delta=" + to_string_prec(this->delta);
        // #ifdef USE_SYMMETRIES
        name += ",Sz=" + std::to_string(this->syms.Sz);
        if(this->boundary_conditions == 0)      name += ",k=" + std::to_string(this->syms.k_sym);
        if(this->k_real_sec(this->syms.k_sym))  name += ",p=" + std::to_string(this->syms.p_sym);
        // #else
        //     name += ",w=" + to_string_prec(this->w) + \
        //             ",edge=" + std::to_string((int)this->add_edge_fields) + \
        //             ",pb=" + std::to_string((int)this->add_parity_breaking);
        // #endif

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

    auto full_model = std::make_unique<QHS::QHamSolver<ConstrainedXXZ>>(this->boundary_conditions, this->L, this->J, this->delta, this->syms.k_sym, this->syms.p_sym, this->syms.Sz, false);
    full_model->diagonalization(false);
    arma::vec E_dis = full_model->get_eigenvalues();// + this->J1 * (this->L - int(this->boundary_conditions)) * (3 + this->eta1 * this->eta1) / 8.;
    
    v_1d<double> Esym;
	v_1d<std::string> symms;
    auto kernel = [&Esym, &symms, this](int k, int p)
    {
        auto symmetric_model = std::make_unique<QHS::QHamSolver<ConstrainedXXZ>>(this->boundary_conditions, this->L, this->J, this->delta, k, p, this->syms.Sz, true);
        symmetric_model->diagonalization(false);
        arma::vec E = symmetric_model->get_eigenvalues();
        
        Esym.insert(Esym.end(), std::make_move_iterator(E.begin()), std::make_move_iterator(E.end()));
        v_1d<std::string> temp_str = v_1d<std::string>(E.size(), "k=" + std::to_string(k) + ",p=" + to_string(p));
		symms.insert(symms.end(), std::make_move_iterator(temp_str.begin()), std::make_move_iterator(temp_str.end()));
    };
    loopSymmetrySectors(kernel);

    
    auto permut = sort_permutation(Esym, [](const double a, const double b)
								   { return a < b; });
	apply_permutation(Esym, permut);
	apply_permutation(symms, permut);
	std::cout << std::endl << Esym.size() << std::endl << E_dis.size() << std::endl;
	printSeparated(std::cout, "\t", 20, true, "symmetry sector", "Energy sym", "Energy total", "difference");
	for (int k = 0; k < min((int)E_dis.size(), (int)Esym.size()); k++){
        // if(std::abs(Esym[k] - E_dis(k)) > 1e-13)
		    printSeparated(std::cout, "\t", 20, true, symms[k], Esym[k], E_dis(k), Esym[k] - E_dis(k));
    }
}

// -------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------- IMPLEMENTATION OF UI

/// @brief Create unique pointer to model with current parameters in class
typename ui::model_pointer ui::create_new_model_pointer(){
    return std::make_unique<QHS::QHamSolver<ConstrainedXXZ>>(this->boundary_conditions, this->L, this->J, this->delta, this->syms.k_sym, this->syms.p_sym, this->syms.Sz);
}

/// @brief Reset member unique pointer to model with current parameters in class
void ui::reset_model_pointer(){
    this->ptr_to_model.reset(new QHS::QHamSolver<ConstrainedXXZ>(this->boundary_conditions, this->L, this->J, this->delta, this->syms.k_sym, this->syms.p_sym, this->syms.Sz) );
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
    user_interface_sym<ConstrainedXXZ>::parse_cmd_options(argc, argv);

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
    set_param(J);
    set_param(delta);

    //<! SYMMETRIES
    choosen_option = "-k";
    this->set_option(this->syms.k_sym, argv, choosen_option);

    choosen_option = "-p";
    this->set_option(this->syms.p_sym, argv, choosen_option);

    choosen_option = "-Sz";
    this->set_option(this->syms.Sz, argv, choosen_option);
    if(this->L % 2 == 1 && this->syms.Sz == 0.0)
        this->syms.Sz = _Spin;

    //<! FOLDER
    std::string folder = this->dir_prefix + kPSep + "results" + kPSep;
    // #ifdef USE_SYMMETRIES
    //     folder += "symmetries" + kPSep;
    // #else
    //     folder += "disorder" + kPSep;
    // #endif
    switch(this->boundary_conditions){
        case 0: folder += "PBC" + kPSep; break;
        case 1: folder += "OBC" + kPSep; break;
        case 2: folder += "ABC" + kPSep; break;
        default:
            folder += "PBC" + kPSep; 
            break;
        
    }
    if (fs::create_directories(folder) || fs::is_directory(folder)) // creating the directory for saving the files with results
    	this->saving_dir = folder;									// if can create dir this is is
}


/// @brief 
void ui::set_default(){
    user_interface_sym<ConstrainedXXZ>::set_default();
    this->J = 1.0;
	this->Js = 0.0;
	this->Jn = 1;
    
    this->delta = 0.0;
	this->deltas = 0.0;
	this->deltan = 1;

    this->syms.k_sym = 0;
    this->syms.p_sym = 1;
    this->syms.Sz = (this->L%2) * _Spin;
}

/// @brief 
void ui::print_help() const {
    user_interface_sym<ConstrainedXXZ>::print_help();

    printf(" Flags for ConstrainedXXZ model:\n");
    printSeparated(std::cout, "\t", 20, true, "-J", "(double)", "some coupling strength");
    printSeparated(std::cout, "\t", 20, true, "-Js", "(double)", "step in coupling strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-Jn", "(int)", "number of couplings in the sweep");
    printSeparated(std::cout, "\t", 20, true, "-delta", "(double)", "coupling to Q5 charge");
    printSeparated(std::cout, "\t", 20, true, "-deltas", "(double)", "step in coupling to Q5 charge sweep");
    printSeparated(std::cout, "\t", 20, true, "-deltan", "(int)", "number of couplings to Q5 in the sweep");
    // #ifdef USE_SYMMETRIES
        printSeparated(std::cout, "\t", 20, true, "-k", "(int)", "quasimomentum symmetry sector");
        printSeparated(std::cout, "\t", 20, true, "-p", "(int)", "parity symmetry sector");
        printSeparated(std::cout, "\t", 20, true, "-Sz", "(float)", "magnetization sector");

    // #else
    //     printSeparated(std::cout, "\t", 20, true, "-w", "(double)", "disorder strength from uniform distribution");
    //     printSeparated(std::cout, "\t", 20, true, "-ws", "(double)", "step in disorder strength sweep");
    //     printSeparated(std::cout, "\t", 20, true, "-wn", "(int)", "number of disorder in the sweep");

    //     printSeparated(std::cout, "\t", 20, true, "-seed", "(u64)", "randomness in position for coupling to grain");
    //     printSeparated(std::cout, "\t", 20, true, "-edge", "(boolean)", "add edge fields for SUSY (when no disorder, i.e. w=0)");
    //     printSeparated(std::cout, "\t", 20, true, "-pb", "(boolean)", "add parity breaking term on edge (when no disorder, i.e. w=0)");
    // #endif
	std::cout << std::endl;
}

/// @brief 
void ui::printAllOptions() const{
    user_interface_sym<ConstrainedXXZ>::printAllOptions();
    std::cout << "H = \u03A3_j H_j" << std::endl << std::endl;
	std::cout << "H_j = P_j (X_j X_j+1 + Y_j Y_j+1 + \u0394 Z_j Z_j+1) P_j+1" << std::endl;

	std::cout << "------------------------------ CHOSEN ConstrainedXXZ OPTIONS:" << std::endl;
    std::cout 
		  << "J  = " << this->J << std::endl
		  << "Jn = " << this->Jn << std::endl
		  << "Js = " << this->Js << std::endl
		  << "\u0394  = " << this->delta << std::endl
		  << "\u0394n = " << this->deltan << std::endl
		  << "\u0394s = " << this->deltas << std::endl;

    std::cout << "k  = " << this->syms.k_sym << std::endl;
    std::cout << "p  = " << this->syms.p_sym << std::endl;
    std::cout << "Sz  = " << this->syms.Sz << std::endl;
		                                            
        std::cout << std::endl;
    printSeparated(std::cout, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
}   

};