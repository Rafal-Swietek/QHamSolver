#include "includes/XXZ_UI.hpp"

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

namespace XXZ_UI{

void ui::make_sim(){
    printAllOptions();

    this->ptr_to_model = create_new_model_pointer();
	
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
	default:
		#define generate_scaling_array(name) arma::linspace(this->name, this->name + this->name##s * (this->name##n - 1), this->name##n)
        #define for_loop(param, var) for (auto& param : generate_scaling_array(var))

        for_loop(system_size, L){ 
            for_loop(J1x, J1){           
                for_loop(J2x, J2){ 
                    for_loop(delta1x, delta1){   
                        for_loop(delta2x, delta2){       
                                for_loop(hzx, hz){ 
                                    for_loop(wx, w)
        {
            this->L = system_size;
            this->J1 = J1x;
            this->J2 = J2x;
            this->delta1 = delta1x;
            this->delta2 = delta2x;
            this->hz = hzx;
            this->w = wx;
            this->site = this->L / 2.;
            const auto start_loop = std::chrono::system_clock::now();
            std::cout << " - - START NEW ITERATION:\t\t par = "; // simulation end
            printSeparated(std::cout, "\t", 16, true, this->L, this->J1, this->J2, this->delta1, this->delta2, this->hz, this->w);
            
            auto kernel = [&](int k, int p, int zx)
                                    {
                                        this->syms.k_sym = k;
                                        this->syms.p_sym = p;
                                        this->syms.zx_sym = zx;
                                        
                                        this->reset_model_pointer();
                                        //diagonal_matrix_elements();
                                        // this->diagonalize();
                                        this->eigenstate_entanglement();
                                        // this->eigenstate_entanglement_degenerate();

                                    };
            loopSymmetrySectors(kernel); continue;
            this->reset_model_pointer();

            this->eigenstate_entanglement_degenerate(); 
            continue;

            diagonal_matrix_elements();
            std::cout << "\t\t - - - - - - FINISHED ITERATION IN : " << tim_s(start_loop) << " seconds\n\t\t\t Total time : " << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
        }}}}}}
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
    auto U1sector = U1Hilbert(this->L, this->syms.Sz);
    arma::Col<ui::element_type> full_state(ULLPOW(this->L), arma::fill::zeros);
    for(int i = 0; i < U1sector.get_hilbert_space_size(); i++)
        full_state(U1sector(i)) = state(i);
    return full_state;
}

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
            ",hz=" + to_string_prec(this->hz);
        #ifdef USE_SYMMETRIES
            if(this->boundary_conditions == 0)      name += ",k=" + std::to_string(this->syms.k_sym);
            if(this->k_real_sec(this->syms.k_sym))  name += ",p=" + std::to_string(this->syms.p_sym);
            if(this->use_flip_X())                  name += ",zx=" + std::to_string(this->syms.zx_sym);
            name += ",Sz=" + to_string_prec(this->syms.Sz, 1);
            // name += ",edge=" + std::to_string((int)this->add_edge_fields);
        #else
            name += ",w=" + to_string_prec(this->w) + \
                    ",pb=" + std::to_string((int)this->add_parity_breaking);
                    // ",edge=" + std::to_string((int)this->add_edge_fields) + 
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
    auto kernel = [&](int k, int p, int zx)
    {
        auto symmetric_model = std::make_unique<QHamSolver<XXZsym>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->hz, k, p, zx, this->syms.Sz);
        if(symmetric_model->get_hilbert_size() > 0){
            symmetric_model->diagonalization(false);
            arma::vec E = symmetric_model->get_eigenvalues();
            
            Esym.insert(Esym.end(), std::make_move_iterator(E.begin()), std::make_move_iterator(E.end()));
            v_1d<std::string> temp_str = v_1d<std::string>(E.size(), "k=" + std::to_string(k) + ",p=" + to_string(p) + ",zx=" + to_string(zx));
            symms.insert(symms.end(), std::make_move_iterator(temp_str.begin()), std::make_move_iterator(temp_str.end()));
        }
    };
    loopSymmetrySectors(kernel);

    auto full_model = std::make_unique<QHamSolver<XXZ>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->hz, this->syms.Sz);
    full_model->diagonalization(false);
    arma::vec E_dis = full_model->get_eigenvalues();
    
    auto permut = sort_permutation(Esym, [](const double a, const double b)
								   { return a < b; });
	apply_permutation(Esym, permut);
	apply_permutation(symms, permut);
	std::cout << std::endl << Esym.size() << std::endl << E_dis.size() << std::endl;
	printSeparated(std::cout, "\t", 20, true, "symmetry sector", "Energy sym", "Energy total", "difference");
	for (int k = 0; k < min((int)E_dis.size(), (int)Esym.size()); k++)
        if(std::abs(Esym[k] - E_dis(k)) > 1e-14)
		    printSeparated(std::cout, "\t", 20, true, symms[k], Esym[k], E_dis(k), Esym[k] - E_dis(k));
}

/// @brief Compaer full hamiltonian to the reconstructed one from symmetry sectors
void ui::compare_hamiltonian()
{   
    auto full_model = std::make_unique<QHamSolver<XXZ>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->hz, this->syms.Sz);
    arma::sp_mat Hfull = full_model->get_hamiltonian();
    const u64 dim = full_model->get_hilbert_size();
    arma::sp_cx_mat H(dim, dim);
    auto kernel = [&](int k, int p, int zx)
    {
        auto symmetric_model = std::make_unique<QHamSolver<XXZsym>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->hz, k, p, zx, this->syms.Sz);
        auto U = symmetric_model->get_model_ref().get_hilbert_space().symmetry_rotation();
        arma::sp_cx_mat Hsym = cast_cx_sparse(symmetric_model->get_hamiltonian());
        H += U * Hsym * U.t();
    };
    loopSymmetrySectors(kernel);
    arma::sp_cx_mat res = cast_cx_sparse(Hfull) - cast_cx_sparse(H);
	printSeparated(std::cout, "\t", 20, true, "col", "row", "diff", "\t", "sym H");
    for(int i = 0; i < dim; i++){
        for(int j = 0; j < dim; j++){
            cpx val = res(i, j);
            if(std::abs(val) > 1e-14)
                printSeparated(std::cout, "\t", 15, true, i, j, val, "\t", H(i,j));
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
    if(this->hz == 0 && this->syms.Sz == 0.0)
        sym_group.emplace_back(op::_spin_flip_x_symmetry(this->L, this->syms.zx_sym));
    
    point_symmetric hilbert( this->L, sym_group, this->boundary_conditions, this->syms.k_sym, 0);
    auto group = hilbert.get_symmetry_group();
    for(auto& idx : {1, 130, 33, 71, 756}){
        for(auto& G : group){
            auto [state, val] = G(idx);
            printSeparated(std::cout, "\t", 16, true, std::vector<bool>(this->L, idx), std::vector<bool>(this->L, state), val);
        }
        std::cout << std::endl;
    }
}


/// @brief Create energy current for this specific model
arma::sp_mat ui::energy_current(){

    const size_t dim_max = ULLPOW(this->L);
    auto check_spin = op::__builtins::get_digit(this->L);
    if(this->J2 != 0.0 || this->delta2 != 0)
        assert(false && "Energy current implemented only for integrable case, no nearest neighbour terms yet!");
    double Jx = 1;
    double Jy = 1;
    double Jz = this->delta1;
    arma::sp_mat jE(dim_max, dim_max);
    printSeparated(std::cout, "\t", 20, true, "Start Current", Jx, Jy, Jz);
    for(int i = 0; i < this->L; i++)
    {
        int nei = (this->boundary_conditions)? i + 1 : (i + 1)%this->L;
        int nei2 = (this->boundary_conditions)? i + 2 : (i + 2)%this->L;
        // printSeparated(std::cout, "\t", 20, true, "site", i, nei, nei2);
        if(nei < this->L && nei2 < this->L){
            for(long k = 0; k < dim_max; k++)
            {
                double Si = double(check_spin(k, i)) - 0.5;
                double Snei = double(check_spin(k, nei)) - 0.5;
                double Snei2 = double(check_spin(k, nei2)) - 0.5;
                {
                    auto [val, state_tmp]   = operators::sigma_x(k, this->L, i);
                    auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei2);
                    jE(new_idx, k) += std::imag(Jx * Jy * Snei * val * val2);
                }{
                    auto [val, state_tmp]   = operators::sigma_x(k, this->L, nei2);
                    auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, i);
                    jE(new_idx, k) -= std::imag(Jx * Jy * Snei * val * val2);
                }{
                    auto [val, state_tmp]   = operators::sigma_x(k, this->L, nei);
                    auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, i);
                    jE(new_idx, k) += std::imag(Jz * Jy * Snei2 * val * val2);
                }{
                    auto [val, state_tmp]   = operators::sigma_x(k, this->L, i);
                    auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei);
                    jE(new_idx, k) -= std::imag(Jz * Jx * Snei2 * val * val2);
                }{
                    auto [val, state_tmp]   = operators::sigma_x(k, this->L, nei2);
                    auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei);
                    jE(new_idx, k) += std::imag(Jz * Jx * Si * val * val2);
                }{
                    auto [val, state_tmp]   = operators::sigma_x(k, this->L, nei);
                    auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei2);
                    jE(new_idx, k) -= std::imag(Jz * Jy * Si * val * val2);
                }
            }
        }
    }

    return jE / double(this->L);
}


// -------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------- IMPLEMENTATION OF UI

/// @brief Create unique pointer to model with current parameters in class
typename ui::model_pointer ui::create_new_model_pointer(){
    #ifdef USE_SYMMETRIES
        return std::make_unique<QHamSolver<XXZsym>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->hz, this->syms.k_sym, this->syms.p_sym, this->syms.zx_sym, this->syms.Sz);
    #else
        return std::make_unique<QHamSolver<XXZ>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->hz, this->syms.Sz, this->add_parity_breaking, this->w, this->seed);
    #endif
}

/// @brief Reset member unique pointer to model with current parameters in class
void ui::reset_model_pointer(){
    #ifdef USE_SYMMETRIES
        return this->ptr_to_model.reset(new QHamSolver<XXZsym>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->hz, this->syms.k_sym, this->syms.p_sym, this->syms.zx_sym, this->syms.Sz));
    #else
        return this->ptr_to_model.reset(new QHamSolver<XXZ>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->hz, this->syms.Sz, this->add_parity_breaking, this->w, this->seed));
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
    XXZUIparent::parse_cmd_options(argc, argv);

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
    set_param(w);

    choosen_option = "-pb";
    this->set_option(this->add_parity_breaking, argv, choosen_option);

    // choosen_option = "-edge";
    // this->set_option(this->add_edge_fields, argv, choosen_option);

    //<! SYMMETRIES
    choosen_option = "-k";
    this->set_option(this->syms.k_sym, argv, choosen_option);

    choosen_option = "-p";
    this->set_option(this->syms.p_sym, argv, choosen_option);
    
    choosen_option = "-zx";
    this->set_option(this->syms.zx_sym, argv, choosen_option);
    
    choosen_option = "-Sz";
    this->set_option(this->syms.Sz, argv, choosen_option);

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
    XXZUIparent::set_default();
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

    this->hz = 0.0;
	this->hzs = 0.0;
	this->hzn = 1;
	this->w = 0.0;
	this->ws = 0.0;
	this->wn = 1;


    // this->add_edge_fields = 0;
    this->add_parity_breaking = 0;

    this->syms.k_sym = 0;
    this->syms.p_sym = 1;
    this->syms.zx_sym = 1;
    this->syms.Sz = 0.0;
}

/// @brief 
void ui::print_help() const {
    XXZUIparent::print_help();

    printf(" Flags for XXZ model:\n");
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

    printSeparated(std::cout, "\t", 20, true, "-hz", "(double)", "uniform longitudinal field on spins");
    printSeparated(std::cout, "\t", 20, true, "-hzs", "(double)", "step in Z-field strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-hzn", "(int)", "number of Z-field values in the sweep");
    #ifdef USE_SYMMETRIES
        printSeparated(std::cout, "\t", 20, true, "-k", "(int)", "quasimomentum symmetry sector");
        printSeparated(std::cout, "\t", 20, true, "-p", "(int)", "parity symmetry sector");
        printSeparated(std::cout, "\t", 20, true, "-zx", "(int)", "spin flip in X direction symmetry sector");
        printSeparated(std::cout, "\t", 20, true, "-Sz", "(float)", "magnetization sector");

    #else
        printSeparated(std::cout, "\t", 20, true, "-w", "(double)", "disorder strength from uniform distribution");
        printSeparated(std::cout, "\t", 20, true, "-ws", "(double)", "step in disorder strength sweep");
        printSeparated(std::cout, "\t", 20, true, "-wn", "(int)", "number of disorder in the sweep");

        printSeparated(std::cout, "\t", 20, true, "-seed", "(u64)", "randomness in position for coupling to grain");
        // printSeparated(std::cout, "\t", 20, true, "-edge", "(boolean)", "add edge fields for SUSY (when no disorder, i.e. w=0)");
        printSeparated(std::cout, "\t", 20, true, "-pb", "(boolean)", "add parity breaking term on edge (when no disorder, i.e. w=0)");
    #endif
	std::cout << std::endl;
}

/// @brief 
void ui::printAllOptions() const{
    XXZUIparent::printAllOptions();
    std::cout << "H = \u03A3_r J_r\u03A3_i [ (1-\u03B7_r) S^x_i S^x_i+1 + (1+\u03B7_r) S^y_i S^y_i+1 + \u0394_r S^z_iS^z_i+1] + \u03A3_i h^z_i S^z_i + h^x\u03A3_i S^x_i" << std::endl << std::endl;
	std::cout << "h_i \u03B5 [hz - w, hz + w]" << std::endl;

	std::cout << "------------------------------ CHOSEN XXZ OPTIONS:" << std::endl;
    std::cout 
		  << "J1  = " << this->J1 << std::endl
		  << "J1n = " << this->J1n << std::endl
		  << "J1s = " << this->J1s << std::endl

		  << "J2  = " << this->J2 << std::endl
		  << "J2n = " << this->J2n << std::endl
		  << "J2s = " << this->J2s << std::endl

		  << "\u03941  = " << this->delta1 << std::endl
		  << "\u03941n = " << this->delta1n << std::endl
		  << "\u03941s = " << this->delta1s << std::endl
		  
		  << "\u03942  = " << this->delta2 << std::endl
		  << "\u03942n = " << this->delta2n << std::endl
		  << "\u03942s = " << this->delta2s << std::endl

		  << "hz  = " << this->hz << std::endl
		  << "hzn = " << this->hzn << std::endl
		  << "hzs = " << this->hzs << std::endl;
    #ifdef USE_SYMMETRIES
		  if(this->boundary_conditions == 0)        std::cout << "k  = " << this->syms.k_sym << std::endl;
		  if(this->k_real_sec(this->syms.k_sym))    std::cout << "p  = " << this->syms.p_sym << std::endl;
		  if(this->use_flip_X())                    std::cout << "zx  = " << this->syms.zx_sym << std::endl;
		  std::cout << "Sz = " << this->syms.Sz << std::endl;
    #else
		 std::cout  << "seed  = " << this->seed << std::endl
		  << "realisations  = " << this->realisations << std::endl
		  << "honid  = " << this->jobid << std::endl
		  << "w  = " << this->w << std::endl
		  << "ws = " << this->ws << std::endl
		  << "wn = " << this->wn << std::endl
		  << "add parity breaking term = " << this->add_parity_breaking << std::endl;
		//   << "add edge fields = " << this->add_edge_fields << std::endl;
    #endif
          std::cout << std::endl;
    printSeparated(std::cout, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
}   





    // arma::vec empty;
    // empty.save(arma::hdf5_name(this->saving_dir + this->set_info({"k", "p", "zx", "zz"}) + ".hdf5", "(empty)"));
    // auto kernel = [&](int k, int p, int zx, int zz)
    // {
    //     auto symmetric_model = std::make_unique<QHamSolver<XXZsym>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
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