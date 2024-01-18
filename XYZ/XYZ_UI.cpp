#include "includes/XYZ_UI.hpp"

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

namespace XYZ_UI{



void ui::make_sim(){
    printAllOptions();

    this->ptr_to_model = create_new_model_pointer();

    compare_energies();
    return;
    // // std::cout << arma::linspace(0, ULLPOW(this->L)-1, ULLPOW(this->L)).t() << std::endl;
	// // std::cout << arma::mat(this->energy_current()) << std::endl;
    // // return;

    // this->ptr_to_model->diagonalization();
    auto H = this->ptr_to_model->get_hamiltonian();

    // auto lancz = lanczos::Lanczos<ui::element_type>(H, this->l_steps, this->l_realis, this->seed, this->reorthogonalize);
    // lancz.convergence(this->saving_dir, this->set_info());
    // auto lancz_block = lanczos::BlockLanczos<ui::element_type>(H, this->l_steps, this->l_realis, this->l_bundle, this->seed, this->reorthogonalize);
    // lancz_block.convergence(this->saving_dir, this->set_info());
    // for(this->L = 10; this->L < 18; this->L++){
    //     this->reset_model_pointer();
    //     auto H = this->ptr_to_model->get_dense_hamiltonian();
    //     int counter = 0;
    //     auto N = this->ptr_to_model->get_hilbert_size();
    //     for(int n = 0; n < N; n++)
    //         for(int m = 0; m < n; m++)
    //             if(std::abs(H(n,m)) > 1e-14)
    //                 counter++;
    //     std::cout << this->L << "\t\t" << counter << std::endl;
    // }
    
    
    
    return;

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
                    for_loop(eta1x, eta1){       
                        for_loop(eta2x, eta2){
                            for_loop(delta1x, delta1){   
                                for_loop(delta2x, delta2){ 
                                    for_loop(hxx, hx){           
                                        for_loop(hzx, hz){ 
                                            for_loop(wx, w)
        {
            this->L = system_size;
            this->J1 = J1x;
            this->J2 = J2x;
            this->delta1 = delta1x;
            this->delta2 = delta2x;
            this->eta1 = eta1x;
            this->eta2 = eta2x;
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
                                        // this->diagonal_matrix_elements();
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
        }}}}}}}}}
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
            name += ",edge=" + std::to_string((int)this->add_edge_fields);
        #else
            name += ",w=" + to_string_prec(this->w) + \
                    ",edge=" + std::to_string((int)this->add_edge_fields) + \
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
        auto symmetric_model = std::make_unique<QHS::QHamSolver<XYZsym>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
                                                                        this->hx, this->hz, k, p, zz, zx, this->add_edge_fields);
        symmetric_model->diagonalization(false);
        arma::vec E = symmetric_model->get_eigenvalues();
        
        Esym.insert(Esym.end(), std::make_move_iterator(E.begin()), std::make_move_iterator(E.end()));
        v_1d<std::string> temp_str = v_1d<std::string>(E.size(), "k=" + std::to_string(k) + ",p=" + to_string(p) + ",zx=" + to_string(zx) + ",zz=" + to_string(zz));
		symms.insert(symms.end(), std::make_move_iterator(temp_str.begin()), std::make_move_iterator(temp_str.end()));
    };
    loopSymmetrySectors(kernel);

    auto full_model = std::make_unique<QHS::QHamSolver<XYZ>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
                                                            this->hx, this->hz, 0, this->add_edge_fields);
    full_model->diagonalization(false);
    arma::vec E_dis = full_model->get_eigenvalues();// + this->J1 * (this->L - int(this->boundary_conditions)) * (3 + this->eta1 * this->eta1) / 8.;
    
    auto permut = sort_permutation(Esym, [](const double a, const double b)
								   { return a < b; });
	apply_permutation(Esym, permut);
	apply_permutation(symms, permut);
	std::cout << std::endl << Esym.size() << std::endl << E_dis.size() << std::endl;
	printSeparated(std::cout, "\t", 20, true, "symmetry sector", "Energy sym", "Energy total", "difference");
	for (int k = 0; k < min((int)E_dis.size(), (int)Esym.size()); k++)
        if(std::abs(Esym[k] - E_dis(k)) > 1e-17)
		    printSeparated(std::cout, "\t", 20, true, symms[k], Esym[k], E_dis(k), Esym[k] - E_dis(k));
}

/// @brief Compaer full hamiltonian to the reconstructed one from symmetry sectors
void ui::compare_hamiltonian()
{   
    auto full_model = std::make_unique<QHS::QHamSolver<XYZ>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
                                                            this->hx, this->hz, 0, this->add_edge_fields);
    arma::sp_mat Hfull = full_model->get_hamiltonian();
    const u64 dim = full_model->get_hilbert_size();
    arma::sp_cx_mat H(dim, dim);
    auto kernel = [&](int k, int p, int zx, int zz)
    {
        auto symmetric_model = std::make_unique<QHS::QHamSolver<XYZsym>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
                                                                        this->hx, this->hz, k, p, zz, zx, this->add_edge_fields);
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
    v_1d<QOps::genOp> sym_group;
    // parity symmetry
    sym_group.emplace_back(QOps::_parity_symmetry(this->L, this->syms.p_sym));

    // spin flips
    if(this->hz == 0)
        sym_group.emplace_back(QOps::_spin_flip_x_symmetry(this->L, this->syms.zx_sym));
    if(this->hx == 0  && (this->L % 2 == 0 || this->hz != 0))
        sym_group.emplace_back(QOps::_spin_flip_z_symmetry(this->L, this->syms.zz_sym));
    
    QHS::point_symmetric hilbert( this->L, sym_group, this->boundary_conditions, this->syms.k_sym, 0);
    auto group = hilbert.get_symmetry_group();
    for(auto& idx : {1, 130, 33, 71, 756}){
        for(auto& G : group){
            auto [state, val] = G(idx);
            printSeparated(std::cout, "\t", 16, true, std::vector<bool>(this->L, idx), std::vector<bool>(this->L, state), val);
        }
        std::cout << std::endl;
    }
}


/// @brief Create supersymmetry charge Q or Q†
/// @param dagger choose Q (false) or Q†(true)
/// @return supercharge (sparse matrix)
arma::SpMat<ui::element_type> ui::create_supercharge(bool dagger){
    // arma::sp_mat q(4, 2);   q(0, 1) = 1;            q(3, 1) = dzeta;
    // arma::sp_mat q2(4, 2);  q2(0, 0) = dzeta;       q2(3, 0) = 1;
    // arma::sp_mat qobc = q;  qobc(1, 0) = -dzeta;    qobc(2, 0) = -dzeta;    q(3, 1) = -dzeta;
    #ifdef USE_SYMMETRIES
        arma::SpMat<ui::element_type> q(4, 2);  
        q(0, 0) = this->eta1;       
        q(3, 0) = 1;
        const int size = this->L - int(dagger);
        const int p_sym = ( dagger && this->L % 2 )? -this->syms.p_sym : this->syms.p_sym;
        const int z_sym = dagger?                    -this->syms.zz_sym : this->syms.zz_sym;

        int k_sec = size % 2 == 0? size / 2 : 0;

        std::vector<QOps::genOp> sym_gen;
        sym_gen.emplace_back(QOps::_parity_symmetry(size, p_sym));
        sym_gen.emplace_back(QOps::_spin_flip_z_symmetry(size, z_sym));
        
        auto _hilbert_space_1 = QHS::point_symmetric(size, sym_gen, this->boundary_conditions, k_sec, 0);

        sym_gen = std::vector<QOps::genOp>();
        sym_gen.emplace_back(QOps::_parity_symmetry(size + 1, size % 2 == 0? -p_sym : p_sym));
        sym_gen.emplace_back(QOps::_spin_flip_z_symmetry(size + 1, -z_sym));
        
        k_sec = (size + 1) % 2 == 0? (size + 1) / 2 : 0;
        auto _hilbert_space_2 = QHS::point_symmetric(size + 1, sym_gen, this->boundary_conditions, k_sec, 0);
        
        return susy::create_supercharge(size, q, this->boundary_conditions, _hilbert_space_1, _hilbert_space_2);
    #else
        return arma::SpMat<ui::element_type>();
    #endif
}

/// @brief Create energy current for this specific model
arma::sp_mat ui::energy_current(){

    const size_t dim_max = ULLPOW(this->L);
    auto check_spin = QOps::__builtins::get_digit(this->L);
    if(this->J2 != 0.0 || this->delta2 != 0)
        assert(false && "Energy current implemented only for integrable case, no nearest neighbour terms yet!");
    double Jx = this->J1 * (1 - this->eta1);
    double Jy = this->J1 * (1 + this->eta1);
    double Jz = this->J1 * this->delta1;
    arma::sp_mat jE(dim_max, dim_max);
    // printSeparated(std::cout, "\t", 20, true, "Start Current", Jx, Jy, Jz);
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

    return jE;
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
    if(this->J2 != 0.0 || this->delta2 != 0)
        assert(false && "Energy current implemented only for integrable case, no nearest neighbour terms yet!");
    double Jx = this->J1 * (1 - this->eta1);
    double Jy = this->J1 * (1 + this->eta1);
    double Jz = this->J1 * this->delta1;
   
    int nei = (this->boundary_conditions)? i + 1 : (i + 1)%this->L;
    int nei2 = (this->boundary_conditions)? i + 2 : (i + 2)%this->L;
    if(nei < this->L && nei2 < this->L){
        double Si = double(check_spin(k, i)) - 0.5;
        double Snei = double(check_spin(k, nei)) - 0.5;
        double Snei2 = double(check_spin(k, nei2)) - 0.5;
        {
            auto [val, state_tmp]   = operators::sigma_x(k, this->L, i);
            auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei2);
            // jE(new_idx, k) += std::imag(Jx * Jy * Snei * val * val2);
            result += my_conjungate(state1(new_idx)) * std::imag(Jx * Jy * Snei * val * val2) * state2(k);
        }{
            auto [val, state_tmp]   = operators::sigma_x(k, this->L, nei2);
            auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, i);
            // jE(new_idx, k) -= std::imag(Jx * Jy * Snei * val * val2);
            result -= my_conjungate(state1(new_idx)) * std::imag(Jx * Jy * Snei * val * val2) * state2(k);
        }{
            auto [val, state_tmp]   = operators::sigma_x(k, this->L, nei);
            auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, i);
            // jE(new_idx, k) += std::imag(Jz * Jy * Snei2 * val * val2);
            result += my_conjungate(state1(new_idx)) * std::imag(Jz * Jy * Snei2 * val * val2) * state2(k);
        }{
            auto [val, state_tmp]   = operators::sigma_x(k, this->L, i);
            auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei);
            // jE(new_idx, k) -= std::imag(Jz * Jx * Snei2 * val * val2);
            result -= my_conjungate(state1(new_idx)) * std::imag(Jz * Jx * Snei2 * val * val2) * state2(k);
        }{
            auto [val, state_tmp]   = operators::sigma_x(k, this->L, nei2);
            auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei);
            // jE(new_idx, k) += std::imag(Jz * Jx * Si * val * val2);
            result += my_conjungate(state1(new_idx)) * std::imag(Jz * Jx * Si * val * val2) * state2(k);
        }{
            auto [val, state_tmp]   = operators::sigma_x(k, this->L, nei);
            auto [val2, new_idx]    = operators::sigma_y(state_tmp, this->L, nei2);
            // jE(new_idx, k) -= std::imag(Jz * Jy * Si * val * val2);
            result -= my_conjungate(state1(new_idx)) * std::imag(Jz * Jy * Si * val * val2) * state2(k);
        }
    }

    return result;
}

// -------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------- IMPLEMENTATION OF UI

/// @brief Create unique pointer to model with current parameters in class
typename ui::model_pointer ui::create_new_model_pointer(){
    #ifdef USE_SYMMETRIES
        return std::make_unique<QHS::QHamSolver<XYZsym>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
                                                    this->hx, this->hz, this->syms.k_sym, this->syms.p_sym, this->syms.zx_sym, this->syms.zz_sym, this->add_edge_fields);
    #else
        return std::make_unique<QHS::QHamSolver<XYZ>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
                                                    this->hx, this->hz, this->add_parity_breaking, this->add_edge_fields, this->w, this->seed); 
    #endif
}

/// @brief Reset member unique pointer to model with current parameters in class
void ui::reset_model_pointer(){
    #ifdef USE_SYMMETRIES
        return this->ptr_to_model.reset(new QHS::QHamSolver<XYZsym>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
                                                    this->hx, this->hz, this->syms.k_sym, this->syms.p_sym, this->syms.zx_sym, this->syms.zz_sym, this->add_edge_fields)); 
    #else
        return this->ptr_to_model.reset(new QHS::QHamSolver<XYZ>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
                                                            this->hx, this->hz, this->add_parity_breaking, this->add_edge_fields, this->w, this->seed)); 
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
    set_param(eta1);
    set_param(eta2);
    set_param(hz);
    set_param(hx);
    set_param(w);

    choosen_option = "-pb";
    this->set_option(this->add_parity_breaking, argv, choosen_option);

    choosen_option = "-edge";
    this->set_option(this->add_edge_fields, argv, choosen_option);

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

	this->eta1 = 0.5;
    this->eta1s = 0.0;
    this->eta1n = 1;
	this->eta2 = 0.0;
	this->eta2s = 0.0;
	this->eta2n = 1;
    
    this->hx = 1.0;
	this->hxs = 0.0;
	this->hxn = 1;
    this->hz = 0.0;
	this->hzs = 0.0;
	this->hzn = 1;
	this->w = 0.0;
	this->ws = 0.0;
	this->wn = 1;


    this->add_edge_fields = 0;
    this->add_parity_breaking = 0;

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
    printSeparated(std::cout, "\t", 20, true, "-eta1s", "(double)", "step in nearest neighbour anisotropy strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-eta1n", "(int)", "number of nearest neighbour anisotropy in the sweep");
    printSeparated(std::cout, "\t", 20, true, "-eta2", "(double)", "next-nearest neighbour anisotropy");
    printSeparated(std::cout, "\t", 20, true, "-eta2s", "(double)", "step in next-nearest neighbour anisotropy strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-eta2n", "(int)", "number of next-nearest neighbour anisotropy in the sweep");

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
        printSeparated(std::cout, "\t", 20, true, "-edge", "(boolean)", "add edge fields for SUSY (when no disorder, i.e. w=0)");
        printSeparated(std::cout, "\t", 20, true, "-pb", "(boolean)", "add parity breaking term on edge (when no disorder, i.e. w=0)");
    #endif
	std::cout << std::endl;
}

/// @brief 
void ui::printAllOptions() const{
    XYZUIparent::printAllOptions();
    std::cout << "H = \u03A3_r J_r\u03A3_i [ (1-\u03B7_r) S^x_i S^x_i+1 + (1+\u03B7_r) S^y_i S^y_i+1 + \u0394_r S^z_iS^z_i+1] + \u03A3_i h^z_i S^z_i + h^x\u03A3_i S^x_i" << std::endl << std::endl;
	std::cout << "h_i \u03B5 [hz - w, hz + w]" << std::endl;

	std::cout << "------------------------------ CHOSEN XYZ OPTIONS:" << std::endl;
    std::cout 
		  << "J1  = " << this->J1 << std::endl
		  << "J1n = " << this->J1n << std::endl
		  << "J1s = " << this->J1s << std::endl

		  << "J2  = " << this->J2 << std::endl
		  << "J2n = " << this->J2n << std::endl
		  << "J2s = " << this->J2s << std::endl

          << "\u03B71  = " << this->eta1 << std::endl
          << "\u03B71s  = " << this->eta1s << std::endl
          << "\u03B71n  = " << this->eta1n << std::endl
		  << "\u03B72  = " << this->eta2 << std::endl
          << "\u03B72s  = " << this->eta2s << std::endl
          << "\u03B72n  = " << this->eta2n << std::endl

		  << "\u03941  = " << this->delta1 << std::endl
		  << "\u03941n = " << this->delta1n << std::endl
		  << "\u03941s = " << this->delta1s << std::endl
		  
		  << "\u03942  = " << this->delta2 << std::endl
		  << "\u03942n = " << this->delta2n << std::endl
		  << "\u03942s = " << this->delta2s << std::endl

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
		  std::cout << "add edge fields = " << this->add_edge_fields << std::endl;
    #else
		 std::cout  << "seed  = " << this->seed << std::endl
		  << "realisations  = " << this->realisations << std::endl
		  << "jonid  = " << this->jobid << std::endl
		  << "w  = " << this->w << std::endl
		  << "ws = " << this->ws << std::endl
		  << "wn = " << this->wn << std::endl
		  << "add parity breaking term = " << this->add_parity_breaking << std::endl
		  << "add edge fields = " << this->add_edge_fields << std::endl;
    #endif
          std::cout << std::endl;
    printSeparated(std::cout, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
}   





    // arma::vec empty;
    // empty.save(arma::hdf5_name(this->saving_dir + this->set_info({"k", "p", "zx", "zz"}) + ".hdf5", "(empty)"));
    // auto kernel = [&](int k, int p, int zx, int zz)
    // {
    //     auto symmetric_model = std::make_unique<QHS::QHamSolver<XYZsym>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
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


// int Gamma = this->Ln;
    // CUE random_matrix(this->seed);
    // auto disorder_generator = disorder<double>(this->seed);
    // for(int t = 0; t < 50; t++){
    //     arma::Col<int> base_states = disorder_generator.create_random_vec<int>(Gamma, 0, ULLPOW(this->L)-1);
    //     // std::cout << base_states.t() << std::endl;
    //     auto mask1 = boost::dynamic_bitset<>(this->L, ULLPOW(this->L/2)-1);
    //     auto mask2 = boost::dynamic_bitset<>(this->L, ULLPOW(this->L) - ULLPOW(this->L/2));
        
    //     for(int id = 0; id < Gamma; id++)
    //         printSeparated(std::cout, "\t", 20, false, boost::dynamic_bitset<>(this->L, base_states(id)));

    //     double S = 0;
    //     int counter = 0;
    //     for(int r = 0; r < this->num_of_points; r++){
    //         arma::cx_vec state(ULLPOW(this->L), arma::fill::zeros);
    //         arma::cx_mat U = random_matrix.generate_matrix(Gamma);
    //         arma::cx_vec coeff = U.col(0) / std::sqrt(arma::cdot(U.col(0), U.col(0)));
            
    //         for(int id = 0; id < Gamma; id++)
    //             state(base_states(id)) += coeff(id);

    //         S += entropy::schmidt_decomposition(state, this->L/2, this->L);
    //         counter++;
    //     }
    //     printSeparated(std::cout, "\t", 20, true, S / double(counter));
    // }
    


    // return;
//     arma::vec single_particle_energies;   
//     bool loaded = single_particle_energies.load(arma::hdf5_name(this->saving_dir + "free_electron_spectra/L=" + std::to_string(this->L) + ".hdf5", "eigenvalues/dataset"));
//     std::cout << single_particle_energies.t() << std::endl;
//     std::cout << arma::sort(arma::cos(two_pi * arma::linspace(0, this->L-1, this->L) / double(this->L))).t() << std::endl;
    
//     v_1d<double> Esym;
// 	v_1d<std::string> symms;
//     auto check_spin = QOps::__builtins::get_digit(this->L);
//     auto kernel = [&](int k, int p, int zx, int zz)
//     {
//         std::vector<QOps::genOp> sym_gen;
//         sym_gen.emplace_back(QOps::_parity_symmetry(this->L, p));
//         if(this->hz == 0 && (this->L % 2 == 0 || this->hx != 0)) sym_gen.emplace_back(QOps::_spin_flip_x_symmetry(this->L, zx));
//         if(this->hx == 0) sym_gen.emplace_back(QOps::_spin_flip_z_symmetry(this->L, zz));
        
//         auto _hilbert_space = QHS::point_symmetric(this->L, sym_gen, this->boundary_conditions, k, 0);
//         auto G = _hilbert_space.get_symmetry_group();
//         u64 dim = _hilbert_space.get_hilbert_space_size();
//         arma::vec E(dim);
//         printSeparated(std::cout, "\t", 16, false, k, p, zz);
//     //#pragma omp parallel for
//         for(long k = 0; k < dim; k++){
//             u64 state = _hilbert_space(k);
//             printSeparated(std::cout, "\t", 16, false, to_binary(state, this->L));
//             double ener = 0;
//             {
//                 for(int j = 0; j < this->L; j++){
//                     if( checkBit(state, j) )
//                         ener -= std::cos(two_pi * (j) / double(this->L));
//                 }
//             }
//             E(k) = ener;
//         }
//         std::cout << std::endl << std::endl;
//         E = arma::sort(E);
//         Esym.insert(Esym.end(), std::make_move_iterator(E.begin()), std::make_move_iterator(E.end()));
//         v_1d<std::string> temp_str = v_1d<std::string>(E.size(), "k=" + std::to_string(k) + ",p=" + to_string(p) + ",zx=" + to_string(zx) + ",zz=" + to_string(zz));
// 		symms.insert(symms.end(), std::make_move_iterator(temp_str.begin()), std::make_move_iterator(temp_str.end()));
//     };
//     loopSymmetrySectors(kernel);

//     u64 dim = ULLPOW(this->L);
//     arma::mat H(dim, dim, arma::fill::zeros);

//     arma::vec states(dim);
//     arma::vec E_dis = states;
// #pragma omp parallel for
//     for(long k = 0; k < dim; k++){
//         double ener = 0;
//         states(k) = k;
//         int fermion_num = __builtin_popcountll(k);
//         for(int j = 0; j < this->L; j++){
//             if( checkBit(k, j) )
//                 ener -= std::cos(two_pi * (j + 0.0) / double(this->L));
//         }
//         E_dis(k) = ener;
//     }
//     auto permut = sort_permutation(E_dis, [](const double a, const double b)
// 								   { return a < b; });
// 	apply_permutation(E_dis, permut);
// 	apply_permutation(states, permut);

//     permut = sort_permutation(Esym, [](const double a, const double b)
// 								   { return a < b; });
// 	apply_permutation(Esym, permut);
// 	apply_permutation(symms, permut);
// 	std::cout << std::endl << Esym.size() << std::endl << E_dis.size() << std::endl;
// 	printSeparated(std::cout, "\t", 20, true, "symmetry sector", "Energy sym", "Energy total", "difference");
// 	for (int k = 0; k < min((int)E_dis.size(), (int)Esym.size()); k++)
//         if(std::abs(Esym[k] - E_dis(k)) > 1e-14)
// 		    printSeparated(std::cout, "\t", 20, true, symms[k], Esym[k], E_dis(k), Esym[k] - E_dis(k));
    
//     return;


//      check_symmetry_generators();
//      compare_hamiltonian();
//        compare_energies(); return;

    // int p = this->syms.p_sym;
    // int zz = this->syms.zz_sym;
    // for( int p : {-1, 1}){
    //     for( int zz : {-1, 1}){
    //         for(int Lx = 3; Lx <= 18; Lx++){
    //             this->L = Lx;
    //             this->syms.k_sym = Lx % 2 == 0? Lx / 2 : 0;
    //             this->syms.p_sym = p;
    //             this->syms.zz_sym = zz;
            
    //             double dzeta = this->eta1;
    //             this->delta1 = (dzeta * dzeta - 1) / 2.;
    //             this->reset_model_pointer();
    //             this->diagonalize();
    //         }
    //     }
    // }
    
    // return;
    // int Lx = 15;

    // this->L = Lx;
    // this->syms.k_sym = Lx % 2 == 0? Lx / 2 : 0;
    // this->syms.zz_sym = 1;
    // for(double dzeta = 0.1; dzeta <= 1.0; dzeta += 0.01){
    //     this->eta1 = dzeta;
    //     this->delta1 = (dzeta * dzeta - 1) / 2.;
    //     this->reset_model_pointer();
    //     this->diagonalize();
    // }
    // return;
    // int k_sec = this->L % 2 == 0? this->L / 2 : 0;

    
                  
    // double dzeta = this->eta1;
    // double Jz = (dzeta * dzeta - 1) / 2.;
    
    // clk::time_point starter = std::chrono::system_clock::now();

    // auto model = std::make_unique<QHS::QHamSolver<XYZsym>>(this->boundary_conditions, this->L, this->J1, 0.0, Jz, this->delta2, this->eta1, this->eta2, 0, 0,
    //                                                              k_sec, this->syms.p_sym, this->syms.zx_sym, this->syms.zz_sym, this->add_edge_fields);
    // //auto model = std::make_unique<QHS::QHamSolver<XYZ>>(this->boundary_conditions, this->L, this->J1, 0.0, Jz, this->delta2, this->eta1, this->eta2, 0.0, 0.0, this->add_parity_breaking, this->add_edge_fields); 
    // model->diagonalization();
    // std::cout << " - - - - - - FINISHED DIAGONALIZATION IN : " << tim_s(starter) << " seconds - - - - - - " << std::endl; // simulation end
    // starter = std::chrono::system_clock::now();

    // arma::mat H = arma::real(model->get_dense_hamiltonian());
    // auto Urot = model->get_model_ref().get_hilbert_space().symmetry_rotation();
    // auto dim = model->get_hilbert_size();
    // arma::vec E1 = model->get_eigenvalues();
    // arma::mat V = (model->get_eigenvectors());

    // std::cout << dim << "\t\t" << E1.t() << std::endl << std::endl;

    // auto Q = this->create_supercharge(false);
    // std::cout << Q.n_cols << "," << Q.n_rows << std::endl << std::endl;
    // auto Q2 = this->create_supercharge(true);
    // std::cout << " - - - - - - FINISHED CREATING SUPERCHARGES IN : " << tim_s(starter) << " seconds - - - - - - " << std::endl; // simulation end
    // starter = std::chrono::system_clock::now();

    // // auto Qbar = make_supercharge(this->L, q, this->syms.p_sym, -this->syms.zz_sym, -this->syms.zz_sym);
    // // auto Qbar2 = make_supercharge(this->L - 1, q, this->L % 2 == 0? this->syms.p_sym : -this->syms.p_sym, -this->syms.zz_sym, -this->syms.zz_sym);
    // std::cout << Q2.n_cols << "," << Q2 .n_rows << std::endl << std::endl;
    // arma::mat H1 = arma::mat(Q.t() * Q);
    // arma::mat H2 = arma::mat(Q2 * Q2.t());
    // arma::mat H_q = (H1 + H2);
    
    // std::string m  = "ANIHILATED  ";
    // for(long k = 0; k < dim; k++){
    //     arma::vec state = arma::real(V.col(k));
    //     state.elem(arma::find( arma::abs(state) < 1e-11) ).zeros();
    //     printSeparated(std::cout, "\t", 16, true, E1(k), state.t());
    // }
    // std::vector<std::string> states_info(dim, "degenerate");
    // std::cout << "--------------------- Q|X> ------------------------------\n" << std::endl;
    // for(long k = 0; k < dim; k++){
    //     arma::vec state = arma::real(Q * V.col(k));
    //     state.elem(arma::find( arma::abs(state) < 1e-11) ).zeros();
    //     if(state.is_zero()) {
    //         states_info[k] = m + " Q ";
    //         printSeparated(std::cout, "\t", 16, true, E1(k), m * int(dim / 2));
    //     } else 
    //         printSeparated(std::cout, "\t", 16, true, E1(k), state.t());
    // }
    
    // std::cout << "--------------------- Q†|X> ------------------------------\n" << std::endl;
    // for(long k = 0; k < dim; k++){
    //     arma::vec state = arma::real(Q2.t() * V.col(k));
    //     state.elem(arma::find( arma::abs(state) < 1e-11) ).zeros();
    //     if(state.is_zero()) {
    //         states_info[k] = m + " Q+";
    //         printSeparated(std::cout, "\t", 16, true, E1(k), m * int(dim / 2));
    //     } else 
    //         printSeparated(std::cout, "\t", 16, true, E1(k), state.t());
    // }
    // std::cout << "----------------------------------------------------------\n" << std::endl;
    // for(long k = 0; k < dim; k++)
    //     printSeparated(std::cout, "\t", 16, true, E1(k), states_info[k]);
    
    // std::cout << " - - - - - - FINISHED ANIHILATION IN : " << tim_s(starter) << " seconds - - - - - - " << std::endl; // simulation end
    // starter = std::chrono::system_clock::now();

    // // states_info = std::vector<std::string>(dim, "degenerate");
    // // std::cout << "--------------------- Qbar|X> ------------------------------\n" << std::endl;
    // // for(long k = 0; k < dim; k++){
    // //     arma::vec state = arma::real(Qbar * V.col(k));
    // //     state.elem(arma::find( arma::abs(state) < 1e-13) ).zeros();
    // //     if(state.is_zero()) {
    // //         states_info[k] = m + " Q+";
    // //         printSeparated(std::cout, "\t", 16, true, E1(k), m * int(dim / 2));
    // //     } else 
    // //         printSeparated(std::cout, "\t", 16, true, E1(k), state.t());
    // // }
    // // std::cout << "--------------------- Qbar†|X> ------------------------------\n" << std::endl;
    // // for(long k = 0; k < dim; k++){
    // //     arma::vec state = arma::real(Qbar2.t() * V.col(k));
    // //     state.elem(arma::find( arma::abs(state) < 1e-13) ).zeros();
    // //     if(state.is_zero()) {
    // //         states_info[k] = m + " Q+";
    // //         printSeparated(std::cout, "\t", 16, true, E1(k), m * int(dim / 2));
    // //     } else 
    // //         printSeparated(std::cout, "\t", 16, true, E1(k), state.t());
    // // }
    // // std::cout << "----------------------------------------------------------\n" << std::endl;
    // // for(long k = 0; k < dim; k++)
    // //     printSeparated(std::cout, "\t", 16, true, E1(k), states_info[k]);

    // // states_info = std::vector<std::string>(dim, "degenerate");
    // // std::cout << "--------------------- C_L|X> ------------------------------\n" << std::endl;
    // // for(long k = 0; k < dim; k++){
    // //     arma::vec state = arma::real(Qbar.t() * Q * V.col(k));
    // //     state.elem(arma::find( arma::abs(state) < 1e-13) ).zeros();
    // //     if(state.is_zero()) {
    // //         states_info[k] = m + " Q+";
    // //         printSeparated(std::cout, "\t", 16, true, E1(k), m * int(dim / 2));
    // //     } else 
    // //         printSeparated(std::cout, "\t", 16, true, E1(k), state.t());
    // // }
    // // std::cout << "--------------------- C_L†|X> ------------------------------\n" << std::endl;
    // // for(long k = 0; k < dim; k++){
    // //     arma::vec state = arma::real((Qbar.t() * Q).t() * V.col(k));
    // //     state.elem(arma::find( arma::abs(state) < 1e-13) ).zeros();
    // //     if(state.is_zero()) {
    // //         states_info[k] = m + " Q+";
    // //         printSeparated(std::cout, "\t", 16, true, E1(k), m * int(dim / 2));
    // //     } else 
    // //         printSeparated(std::cout, "\t", 16, true, E1(k), state.t());
    // // }
    // // std::cout << "----------------------------------------------------------\n" << std::endl;
    // // for(long k = 0; k < dim; k++)
    // //     printSeparated(std::cout, "\t", 16, true, E1(k), states_info[k]);
        
    // arma::vec E2 = arma::eig_sym(H_q);
    // arma::vec E2_1 = arma::eig_sym(H1);
    // arma::vec E2_2 = arma::eig_sym(H2);

    // printSeparated(std::cout, "\t", 14, true, "H", "QQ† + Q†Q", "diff", "\t", "QQ†", "Q†Q");//, "\t---------", "\t", "H", "H_q", "diff", "\n");
    // for(long k = 0; k < dim; k++)
    //     printSeparated(std::cout, "\t", 14, true, E1(k), E2(k), std::abs(E1(k) - E2(k)), "\t", E2_1(k), E2_2(k));//, "\t---------", "\t", E1(k), E3(k), std::abs(E1(k) - E3(k)));
    
    



    // std::cout << " - - - - - - FINISHED DIAGONALIZATION OF SUPERHAMILTONIAN IN : " << tim_s(starter) << " seconds - - - - - - " << std::endl; // simulation end
    // return;
// //