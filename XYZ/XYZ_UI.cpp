#include "includes/XYZ_UI.hpp"

int outer_threads = 1;
int num_of_threads = 1;

namespace XYZ_UI{

void ui::make_sim(){
    printAllOptions();
    
    check_symmetry_generators();
    //compare_hamiltonian();
    //compare_energies();
    return;
//
    this->ptr_to_model = create_new_model_pointer();
	
	clk::time_point start = std::chrono::system_clock::now();
    switch (this->fun)
	{
	case 0: 
		diagonalize(); 
		break;
	case 1:
		eigenstate_entanglement();
		break;
	default:
		#define generate_scaling_array(name) arma::linspace(this->name, this->name + this->name##s * (this->name##n - 1), this->name##n);
		auto L_list = generate_scaling_array(L);
		auto J1_list = generate_scaling_array(J1);
		auto J2_list = generate_scaling_array(J2);
		auto delta1_list = generate_scaling_array(delta1);
		auto delta2_list = generate_scaling_array(delta2);
		auto hz_list = generate_scaling_array(hz);
        auto hx_list = generate_scaling_array(hz);
		auto w_list = generate_scaling_array(w);

		for (auto& system_size : L_list){
			for (auto& J1x : J1_list){
                for (auto& J2x : J2_list){
			        for (auto& delta1x : delta1_list){
                        for (auto& delta2x : delta2_list){
				            for (auto& hxx : hx_list){
				            	for(auto& hzx : hz_list){
				            		for(auto& wx : w_list){
                                        this->L = system_size;
                                        this->J1 = J1x;
                                        this->J2 = J2x;
                                        this->delta1 = delta1x;
                                        this->delta2 = delta2x;
                                        this->hx = hxx;
                                        this->hz = hzx;
                                        this->w = wx;
                                        this->site = this->L / 2.;
                                        const auto start_loop = std::chrono::system_clock::now();
                                        std::cout << " - - START NEW ITERATION:\t\t par = "; // simulation end
                                        printSeparated(std::cout, "\t", 16, true, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2, this->hz, this->hx, this->w);
                                        
                                        auto kernel = [&](int k, int p, int zx, int zz)
                                                                {
                                                                    this->syms.k_sym = k;
                                                                    this->syms.p_sym = p;
                                                                    this->syms.zx_sym = zx;
                                                                    this->syms.zz_sym = zz;
                                                                    
                                                                    this->reset_model_pointer();
                                                                    //this->diagonalize();
                                                                    this->eigenstate_entanglement();

                                                                };
                                        loopSymmetrySectors(kernel);


                                        this->reset_model_pointer();

                                        std::cout << "\t\t - - - - - - FINISHED ITERATION IN : " << tim_s(start_loop) << " seconds\n\t\t\t Total time : " << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
						}}}}}}}
            }
        std::cout << "Add default function" << std::endl;
	}
	std::cout << " - - - - - - FINISHED CALCULATIONS IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
}



// -------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------- MODEL DEPENDENT FUNCTIONS
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
            ",e1=" + to_string_prec(this->eta1) + \
            ",e2=" + to_string_prec(this->eta2) + \
            ",hx=" + to_string_prec(this->hx) + \
            ",hz=" + to_string_prec(this->hz);
        #ifdef USE_SYMMETRIES
            if(this->boundary_conditions == 0)      name += ",k=" + std::to_string(this->syms.k_sym);
            if(this->k_real_sec(this->syms.k_sym))  name += ",p=" + std::to_string(this->syms.p_sym);
            if(this->use_flip_X())                  name += ",zx=" + std::to_string(this->syms.zx_sym);
            if(this->use_flip_Z())                  name += ",zz=" + std::to_string(this->syms.zz_sym);
        #else
            name += ",w=" + to_string_prec(this->w) + \
                    ",pb=" + std::to_string((int)this->add_parity_breaking);
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
    auto kernel = [&](int k, int p, int zx, int zz)
    {
        auto symmetric_model = std::make_unique<QHamSolver<XYZsym>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
                                                                        this->hx, this->hz, k, p, zx, zz);
        symmetric_model->diagonalization(false);
        arma::vec E = symmetric_model->get_eigenvalues();
        Esym.insert(Esym.end(), std::make_move_iterator(E.begin()), std::make_move_iterator(E.end()));
        v_1d<std::string> temp_str = v_1d<std::string>(E.size(), "k=" + std::to_string(k) + ",p=" + to_string(p) + ",zx=" + to_string(zx) + ",zz=" + to_string(zz));
		symms.insert(symms.end(), std::make_move_iterator(temp_str.begin()), std::make_move_iterator(temp_str.end()));
    };
    loopSymmetrySectors(kernel);

    auto full_model = std::make_unique<QHamSolver<XYZ>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
                                                            this->hx, this->hz, 0, 0, 0);
    full_model->diagonalization(false);
    arma::vec E_dis = full_model->get_eigenvalues();
    auto permut = sort_permutation(Esym, [](const double a, const double b)
								   { return a < b; });
	apply_permutation(Esym, permut);
	apply_permutation(symms, permut);
	std::cout << std::endl << Esym.size() << std::endl;
	std::cout << "symmetry sector\t\tEnergy sym\t\tEnergy total\t\tdifference" << endl;
	printSeparated(std::cout, "\t", 20, true, "symmetry sector", "Energy sym", "Energy total", "difference");
	for (int k = 0; k < min((int)E_dis.size(), (int)Esym.size()); k++)
		printSeparated(std::cout, "\t", 20, true, symms[k], Esym[k], E_dis(k), Esym[k] - E_dis(k));
}

/// @brief Compaer full hamiltonian to the reconstructed one from symmetry sectors
void ui::compare_hamiltonian()
{   
    auto full_model = std::make_unique<QHamSolver<XYZ>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
                                                            this->hx, this->hz, 0, 0, 0);
    arma::sp_mat Hfull = full_model->get_hamiltonian();
    const u64 dim = full_model->get_hilbert_size();
    arma::sp_cx_mat H(dim, dim);
    auto kernel = [&](int k, int p, int zx, int zz)
    {
        auto symmetric_model = std::make_unique<QHamSolver<XYZsym>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
                                                                        this->hx, this->hz, k, p, zx, zz);
        auto U = symmetric_model->get_model_ref().get_hilbert_space().symmetry_rotation();
        arma::sp_cx_mat Hsym = cast_cx_sparse(symmetric_model->get_hamiltonian());
        H += U * Hsym * U.t();
    };
    loopSymmetrySectors(kernel);
    arma::sp_cx_mat res = cast_cx_sparse(Hfull) - cast_cx_sparse(H);
    for(int i = 0; i < dim; i++){
        for(int j = 0; j < dim; j++){
            cpx val = res(i, j);
            if(std::abs(val) > 1e-15)
                printSeparated(std::cout, "\t", 15, true, i, j, val);
        }
    }
}

/// @brief 
void ui::check_symmetry_generators()
{
    v_1d<op::genOp> sym_group;
    // parity symmetry
    sym_group.emplace_back(op::_parity_symmetry(this->L, this->syms.p_sym));

    // spin flips
    if(this->hz == 0)
        sym_group.emplace_back(op::_spin_flip_x_symmetry(this->L, this->syms.zx_sym));
    if(this->hx == 0  && (this->L % 2 == 0 || this->hz != 0))
        sym_group.emplace_back(op::_spin_flip_z_symmetry(this->L, this->syms.zz_sym));
    
    point_symmetric hilbert( this->L, sym_group, this->boundary_conditions, this->syms.k_sym, 0);
    auto group = hilbert.get_symmetry_group();
    for(auto& idx : {1, 130, 33, 71, 756}){
        for(auto& G : group){
            auto [state, val] = G(idx);
            printSeparated(std::cout, "\t", 16, true, std::bitset<11>(idx), std::bitset<11>(state), val);
        }
        std::cout << std::endl;
    }
}

/// @brief 
void ui::eigenstate_entanglement()
{
    clk::time_point start = std::chrono::system_clock::now();
	
	std::string dir = this->saving_dir + "Entropy" + kPSep + "Eigenstate" + kPSep;
	createDirs(dir);
	
    int LA = this->site;
	size_t dim = this->ptr_to_model->get_hilbert_size();
	
	std::string info = this->set_info();
	std::string filename = info;// + "_subsize=" + std::to_string(LA);

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
    
    #ifdef USE_SYMMETRIES
        const auto U = this->ptr_to_model->get_model_ref().get_hilbert_space().symmetry_rotation();
        auto transform = [&](const auto& state)
                                { return U * state; };
    #else
        auto transform = [&](const auto& state)
                                { return state; };
    #endif
    std::cout << " - - - - - - FINISHED CREATING SYMMETRY TRANSFORMATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end

    arma::mat S(size, this->L / 2 + 1, arma::fill::zeros);
    int th_num = this->thread_number;
    omp_set_num_threads(1);
    std::cout << th_num << "\t\t" << omp_get_num_threads() << std::endl;
    
    auto subsystem_sizes = arma::conv_to<arma::Col<int>>::from(arma::linspace(0, this->L / 2, this->L / 2 + 1));
    std::cout << subsystem_sizes.t() << std::endl;
    for(auto& LA : subsystem_sizes){
        start = std::chrono::system_clock::now();
    #pragma omp parallel for num_threads(th_num) schedule(dynamic)
        for(int n = 0; n < size; n++){
            auto eigenstate = this->ptr_to_model->get_eigenState(n);
            
            arma::Col<element_type> state = transform(eigenstate);
            S(n, LA) = entropy::schmidt_decomposition(state, LA, this->L);
            //double entro = entropy::vonNeumann(state, LA, this->L);
            //if(std::abs(entro - S(n)) > 1e-14)
            //printSeparated(std::cout, "\t", 16, true, E(n), entro, S(n), std::abs(entro - S(n)));
        }
    }
    std::cout << " - - - - - - FINISHED ENTROPY CALCULATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
    
    E.save(arma::hdf5_name(dir + filename + ".hdf5", "energies"));
	S.save(arma::hdf5_name(dir + filename + ".hdf5", "entropy", arma::hdf5_opts::append));
    
        
    #ifdef ARMA_USE_SUPERLU
        arma::Mat<element_type> V;
        if(this->ch) V = this->ptr_to_model->get_eigenvectors();
        else         V = this->ptr_to_model->get_eigenvectors().submat(0, this->ptr_to_model->E_av_idx - 50, dim - 1, this->ptr_to_model->E_av_idx + 50);
    #else
        arma::Mat<element_type> V = this->ptr_to_model->get_eigenvectors().submat(0, this->ptr_to_model->E_av_idx - 50, dim - 1, this->ptr_to_model->E_av_idx + 50);
    #endif
    V.save(arma::hdf5_name(dir + filename + ".hdf5", "eigenvectors",arma::hdf5_opts::append));
}

// -------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------- IMPLEMENTATION OF UI

/// @brief Create unique pointer to model with current parameters in class
typename ui::model_pointer ui::create_new_model_pointer(){
    #ifdef USE_SYMMETRIES
        return std::make_unique<QHamSolver<XYZsym>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
                                                    this->hx, this->hz, this->syms.k_sym, this->syms.p_sym, this->syms.zx_sym, this->syms.zz_sym);
    #else
        return std::make_unique<QHamSolver<XYZ>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
                                                    this->hx, this->hz, this->add_parity_breaking, this->w, this->seed); 
    #endif
}

/// @brief Reset member unique pointer to model with current parameters in class
void ui::reset_model_pointer(){
    #ifdef USE_SYMMETRIES
        return this->ptr_to_model.reset(new QHamSolver<XYZsym>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
                                                    this->hx, this->hz, this->syms.k_sym, this->syms.p_sym, this->syms.zx_sym, this->syms.zz_sym)); 
    #else
        return this->ptr_to_model.reset(new QHamSolver<XYZ>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
                                                            this->hx, this->hz, this->add_parity_breaking, this->w, this->seed)); 
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
    XYZUIparent::parse_cmd_options(argc, argv);

    #ifdef USE_SYMMETRIES
        //<! set averaging to one for symmetric model 
        this->realisations = 1;
        this->jobid = 1;
    #endif
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
    set_param(hx);
    set_param(w);

    choosen_option = "-eta1";
    this->set_option(this->eta1, argv, choosen_option);

    choosen_option = "-eta2";
    this->set_option(this->eta2, argv, choosen_option);
    
    choosen_option = "-pb";
    this->set_option(this->add_parity_breaking, argv, choosen_option);

    //<! SYMMETRIES
    choosen_option = "-k";
    this->set_option(this->syms.k_sym, argv, choosen_option);

    choosen_option = "-p";
    this->set_option(this->syms.p_sym, argv, choosen_option);
    
    choosen_option = "-zx";
    this->set_option(this->syms.zx_sym, argv, choosen_option);
    
    choosen_option = "-zz";
    this->set_option(this->syms.zz_sym, argv, choosen_option);

    //<! FOLDER
    std::string folder = "." + kPSep + "results" + kPSep;
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
    if (fs::create_directories(folder) || fs::is_directory(folder)) // creating the directory for saving the files with results
    	this->saving_dir = folder;									// if can create dir this is is
}


/// @brief 
void ui::set_default(){
    XYZUIparent::set_default();
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

    this->hx = 1.0;
	this->hxs = 0.0;
	this->hxn = 1;
    this->hz = 0.0;
	this->hzs = 0.0;
	this->hzn = 1;
	this->w = 0.0;
	this->ws = 0.0;
	this->wn = 1;

	this->eta1 = 0.5;
	this->eta2 = 0.0;

    this->add_parity_breaking = 1;

    this->syms.k_sym = 0;
    this->syms.p_sym = 1;
    this->syms.zx_sym = 1;
    this->syms.zz_sym = 1;
}

/// @brief 
void ui::print_help() const {
    XYZUIparent::print_help();

    printf(" Flags for XYZ model:\n");
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

    printSeparated(std::cout, "\t", 20, true, "-eta1", "(double)", "nearest neighbour anisotropy");
    printSeparated(std::cout, "\t", 20, true, "-eta2", "(double)", "next-nearest neighbour anisotropy");

    printSeparated(std::cout, "\t", 20, true, "-hz", "(double)", "uniform longitudinal field on spins");
    printSeparated(std::cout, "\t", 20, true, "-hzs", "(double)", "step in Z-field strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-hzn", "(int)", "number of Z-field values in the sweep");
    printSeparated(std::cout, "\t", 20, true, "-hx", "(double)", "uniform transverse field on spins");
    printSeparated(std::cout, "\t", 20, true, "-hxs", "(double)", "step in X-field strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-hxn", "(int)", "number of X-field values in the sweep");
    #ifdef USE_SYMMETRIES
        printSeparated(std::cout, "\t", 20, true, "-k", "(int)", "quasimomentum symmetry sector");
        printSeparated(std::cout, "\t", 20, true, "-p", "(int)", "parity symmetry sector");
        printSeparated(std::cout, "\t", 20, true, "-zx", "(int)", "spin flip in X direction symmetry sector");
        printSeparated(std::cout, "\t", 20, true, "-zz", "(int)", "spin flip in Z direction symmetry sector");

    #else
        printSeparated(std::cout, "\t", 20, true, "-w", "(double)", "disorder strength from uniform distribution");
        printSeparated(std::cout, "\t", 20, true, "-ws", "(double)", "step in disorder strength sweep");
        printSeparated(std::cout, "\t", 20, true, "-wn", "(int)", "number of disorder in the sweep");

        printSeparated(std::cout, "\t", 20, true, "-seed", "(u64)", "randomness in position for coupling to grain");
        printSeparated(std::cout, "\t", 20, true, "-pb", "(boolean)", "add parity breaking term on edge (when no disorder, i.e. w=0)");
    #endif
	std::cout << std::endl;
}

/// @brief 
void ui::printAllOptions() const{
    XYZUIparent::printAllOptions();
    std::cout << "H = \u03A3_r J_r\u03A3_i [ (1-\u03B7_r) S^x_i S^x_i+1 + (1+\u03B7_r) S^y_i S^y_i+1) + \u0394_r S^z_iS^z_i+1] + \u03A3_i h^z_i S^z_i + h^x\u03A3_i S^x_i" << std::endl << std::endl;
	std::cout << "h_i \u03B5 [hz - w, hz + w]" << std::endl;

	std::cout << "------------------------------ CHOSEN XYZ OPTIONS:" << std::endl;
    std::cout 
		  << "J1  = " << this->J1 << std::endl
		  << "J1n = " << this->J1n << std::endl
		  << "J1s = " << this->J1s << std::endl

		  << "J2  = " << this->J2 << std::endl
		  << "J2n = " << this->J2n << std::endl
		  << "J2s = " << this->J2s << std::endl

          << "\u03B7_1  = " << this->eta1 << std::endl
		  << "\u03B7_2  = " << this->eta2 << std::endl

		  << "delta1  = " << this->delta1 << std::endl
		  << "delta1n = " << this->delta1n << std::endl
		  << "delta1s = " << this->delta1s << std::endl
		  
		  << "delta2  = " << this->delta2 << std::endl
		  << "delta2n = " << this->delta2n << std::endl
		  << "delta2s = " << this->delta2s << std::endl

		  << "hx  = " << this->hx << std::endl
		  << "hxn = " << this->hxn << std::endl
		  << "hxs = " << this->hxs << std::endl

		  << "hz  = " << this->hz << std::endl
		  << "hzn = " << this->hzn << std::endl
		  << "hzs = " << this->hzs << std::endl;
    #ifdef USE_SYMMETRIES
		  if(this->boundary_conditions == 0)        std::cout << "k  = " << this->syms.k_sym << std::endl;
		  if(this->k_real_sec(this->syms.k_sym))    std::cout << "p  = " << this->syms.p_sym << std::endl;
		  if(this->use_flip_X())                    std::cout << "zx  = " << this->syms.zx_sym << std::endl;
		  if(this->use_flip_Z())                    std::cout << "zz  = " << this->syms.zz_sym << std::endl;
    #else
		 std::cout  << "seed  = " << this->seed << std::endl
		  << "realisations  = " << this->realisations << std::endl
		  << "honid  = " << this->jobid << std::endl
		  << "w  = " << this->w << std::endl
		  << "ws = " << this->ws << std::endl
		  << "wn = " << this->wn << std::endl
		  << "add parity breaking term = " << this->add_parity_breaking << std::endl;
    #endif
          std::cout << std::endl;
    printSeparated(std::cout, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
}   










};