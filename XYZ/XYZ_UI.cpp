#include "includes/XYZ_UI.hpp"


int outer_threads = 1;
int num_of_threads = 1;

std::string to_binary(u64 num, int size){
    std::string bin_num = "";
    while (num > 0)
    {
        int bin = num % 2;
        bin_num += std::to_string(bin);
        num /= 2;
    }
    std::reverse(bin_num.begin(), bin_num.end());
    if(bin_num.size() < size){
        bin_num = std::string(size - bin_num.size(), '0') + bin_num;
    }
    return bin_num;
}

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
    
    //  check_symmetry_generators();
    //  compare_hamiltonian();
    //  compare_energies(); return;


    int k_sec = this->L % 2 == 0? this->L / 2 : 0;

    
                  
    double dzeta = this->eta1;
    double Jz = (dzeta * dzeta - 1) / 2.;
    
    clk::time_point starter = std::chrono::system_clock::now();

    auto model = std::make_unique<QHamSolver<XYZsym>>(this->boundary_conditions, this->L, this->J1, 0.0, Jz, this->delta2, this->eta1, this->eta2, 0, 0,
                                                                 k_sec, this->syms.p_sym, this->syms.zx_sym, this->syms.zz_sym, this->add_edge_fields);
    //auto model = std::make_unique<QHamSolver<XYZ>>(this->boundary_conditions, this->L, this->J1, 0.0, Jz, this->delta2, this->eta1, this->eta2, 0.0, 0.0, this->add_parity_breaking, this->add_edge_fields); 
    model->diagonalization();
    std::cout << " - - - - - - FINISHED DIAGONALIZATION IN : " << tim_s(starter) << " seconds - - - - - - " << std::endl; // simulation end
    starter = std::chrono::system_clock::now();

    arma::mat H = arma::real(model->get_dense_hamiltonian());
    auto Urot = model->get_model_ref().get_hilbert_space().symmetry_rotation();
    auto dim = model->get_hilbert_size();
    arma::vec E1 = model->get_eigenvalues();
    arma::mat V = (model->get_eigenvectors());
    
    //arma::eig_sym(E1, V, H);

    int b = this->boundary_conditions;
    // E1 = -E1 + this->J1 * (this->L - b) * (2 + Jz) / 4. + b * this->J1 * (1 + 3 * dzeta * dzeta) / 4;
    // E1 = arma::sort(E1);

    std::cout << dim << "\t\t" << E1.t() << std::endl << std::endl;

    arma::vec up(2); up(0) = 1;
    arma::vec down(2); down(1) = 1;
    arma::mat e = arma::eye(2, 2);
    
    auto make_supercharge = [&](int size, arma::mat q, int p_sym, int zz_sym_in, int zz_sym_out) 
        -> arma::mat
    {
        
        auto check_spin = op::__builtins::get_digit(size);
        const u64 dim1 = ULLPOW(size);
        const u64 dim2 = ULLPOW(size + 1);

        arma::mat supercharge(dim2, dim1, arma::fill::zeros);
        for(int j = 0; j < size; j++){
            u64 dim_left = ULLPOW(j);
            u64 dim_right = ULLPOW(size - j - 1);
            arma::mat ham = arma::kron(arma::eye<arma::mat>(dim_right, dim_right), arma::kron(q, arma::eye<arma::mat>(dim_left, dim_left)));
            supercharge += (j % 2 == 0)? -ham : ham;
        }
        // for(long k = 0; k < dim1; k++){
        //     auto base_state = k;
        //     for(int j = 0; j <= size; j++){
        //         int site = j;
        //         int sign = j % 2 == 1? 1 : -1;
        //         if(j == size){
        //             site = size - 1;
        //             sign = (size % 2 == 0)? -1 : 1;
        //         }
        //         arma::vec spin = check_spin(base_state, site)? up : down;
        //         arma::vec state = q * spin;
                
        //         for(int ii = 0; ii < state.size(); ii++){
        //             if(std::abs(state(ii)) > 0)
        //             {
        //                 const u64 right = base_state % ULLPOW(size - site);
        //                 const u64 left = base_state - right;

        //                 const u64 _add = ii * ULLPOW(size - site - 1);

        //                 u64 new_idx = 2 * left + right + _add;
        //                 if(j == size){
        //                     new_idx = std::get<0>( T_op(new_idx) );
        //                 }

        //                 printSeparated(std::cout, "\t", 16, true, to_binary(base_state, size), site, left, right, _add, to_binary(new_idx, size + 1));
                        
        //                 supercharge(new_idx, k) += state(ii) * double(sign); // define in same basis as other charge
        //             }
        //         }
        //     }
        // }
        if(this->boundary_conditions){
            return supercharge;
        } else {
            int k_sec = size % 2 == 0? size / 2 : 0;
            arma::sp_mat T = arma::real(op::_translation_symmetry(size + 1, 0).to_matrix(dim2));

            std::vector<op::genOp> sym_gen;
            sym_gen.emplace_back(op::_parity_symmetry(size, p_sym));
            sym_gen.emplace_back(op::_spin_flip_z_symmetry(size, zz_sym_in));
            
            auto _hilbert_space_1 = point_symmetric(size, sym_gen, this->boundary_conditions, k_sec, 0);
            auto U1 = (_hilbert_space_1.symmetry_rotation());

            sym_gen = std::vector<op::genOp>();
            sym_gen.emplace_back(op::_parity_symmetry(size + 1, size % 2 == 0? -p_sym : p_sym));
            sym_gen.emplace_back(op::_spin_flip_z_symmetry(size + 1, zz_sym_out));
            
            k_sec = (size + 1) % 2 == 0? (size + 1) / 2 : 0;
            auto _hilbert_space_2 = point_symmetric(size + 1, sym_gen, this->boundary_conditions, k_sec, 0);
            auto U2 = (_hilbert_space_2.symmetry_rotation());
            
            u64 dim_rest = ULLPOW(size - 1);
            arma::mat Q0 = arma::kron(arma::eye<arma::mat>(dim_rest, dim_rest), q);
            supercharge += T * Q0 * ( (size % 2 == 0)? -1.0 : 1.0);

            return std::sqrt(size / (size + 1.0)) * U2.t() * supercharge * U1 / std::sqrt(2);
        }
    };

    arma::mat q(4, 2);   q(0, 1) = 1;            q(3, 1) = dzeta;
    arma::mat q2(4, 2);  q2(0, 0) = dzeta;       q2(3, 0) = 1;
    arma::mat qobc = q;  qobc(1, 0) = -dzeta;    qobc(2, 0) = -dzeta;    q(3, 1) = -dzeta;


    auto q_in = this->boundary_conditions? qobc : q2;
    auto Q = make_supercharge(this->L, q_in, this->syms.p_sym, this->syms.zz_sym, -this->syms.zz_sym);
    std::cout << Q.n_cols << "," << Q.n_rows << std::endl << std::endl;

    auto Q2 = make_supercharge(this->L - 1, q_in, this->L % 2 == 0? this->syms.p_sym : -this->syms.p_sym, -this->syms.zz_sym, this->syms.zz_sym);
    std::cout << " - - - - - - FINISHED CREATING SUPERCHARGES IN : " << tim_s(starter) << " seconds - - - - - - " << std::endl; // simulation end
    starter = std::chrono::system_clock::now();

    // auto Qbar = make_supercharge(this->L, q, this->syms.p_sym, -this->syms.zz_sym, -this->syms.zz_sym);
    // auto Qbar2 = make_supercharge(this->L - 1, q, this->L % 2 == 0? this->syms.p_sym : -this->syms.p_sym, -this->syms.zz_sym, -this->syms.zz_sym);
    std::cout << Q2.n_cols << "," << Q2 .n_rows << std::endl << std::endl;
    arma::mat H1 = Q.t() * Q;
    arma::mat H2 = Q2 * Q2.t();
    arma::mat H_q = (H1 + H2);
    
    std::string m  = "ANIHILATED  ";
    for(long k = 0; k < dim; k++){
        arma::vec state = arma::real(V.col(k));
        state.elem(arma::find( arma::abs(state) < 1e-13) ).zeros();
        printSeparated(std::cout, "\t", 16, true, E1(k), state.t());
    }
    std::vector<std::string> states_info(dim, "degenerate");
    std::cout << "--------------------- Q|X> ------------------------------\n" << std::endl;
    for(long k = 0; k < dim; k++){
        arma::vec state = arma::real(Q * V.col(k));
        state.elem(arma::find( arma::abs(state) < 1e-12) ).zeros();
        if(state.is_zero()) {
            states_info[k] = m + " Q ";
            printSeparated(std::cout, "\t", 16, true, E1(k), m * int(dim / 2));
        } else 
            printSeparated(std::cout, "\t", 16, true, E1(k), state.t());
    }
    
    std::cout << "--------------------- Q†|X> ------------------------------\n" << std::endl;
    for(long k = 0; k < dim; k++){
        arma::vec state = arma::real(Q2.t() * V.col(k));
        state.elem(arma::find( arma::abs(state) < 1e-12) ).zeros();
        if(state.is_zero()) {
            states_info[k] = m + " Q+";
            printSeparated(std::cout, "\t", 16, true, E1(k), m * int(dim / 2));
        } else 
            printSeparated(std::cout, "\t", 16, true, E1(k), state.t());
    }
    std::cout << "----------------------------------------------------------\n" << std::endl;
    for(long k = 0; k < dim; k++)
        printSeparated(std::cout, "\t", 16, true, E1(k), states_info[k]);
    
    std::cout << " - - - - - - FINISHED ANIHILATION IN : " << tim_s(starter) << " seconds - - - - - - " << std::endl; // simulation end
    starter = std::chrono::system_clock::now();

    // states_info = std::vector<std::string>(dim, "degenerate");
    // std::cout << "--------------------- Qbar|X> ------------------------------\n" << std::endl;
    // for(long k = 0; k < dim; k++){
    //     arma::vec state = arma::real(Qbar * V.col(k));
    //     state.elem(arma::find( arma::abs(state) < 1e-13) ).zeros();
    //     if(state.is_zero()) {
    //         states_info[k] = m + " Q+";
    //         printSeparated(std::cout, "\t", 16, true, E1(k), m * int(dim / 2));
    //     } else 
    //         printSeparated(std::cout, "\t", 16, true, E1(k), state.t());
    // }
    // std::cout << "--------------------- Qbar†|X> ------------------------------\n" << std::endl;
    // for(long k = 0; k < dim; k++){
    //     arma::vec state = arma::real(Qbar2.t() * V.col(k));
    //     state.elem(arma::find( arma::abs(state) < 1e-13) ).zeros();
    //     if(state.is_zero()) {
    //         states_info[k] = m + " Q+";
    //         printSeparated(std::cout, "\t", 16, true, E1(k), m * int(dim / 2));
    //     } else 
    //         printSeparated(std::cout, "\t", 16, true, E1(k), state.t());
    // }
    // std::cout << "----------------------------------------------------------\n" << std::endl;
    // for(long k = 0; k < dim; k++)
    //     printSeparated(std::cout, "\t", 16, true, E1(k), states_info[k]);

    // states_info = std::vector<std::string>(dim, "degenerate");
    // std::cout << "--------------------- C_L|X> ------------------------------\n" << std::endl;
    // for(long k = 0; k < dim; k++){
    //     arma::vec state = arma::real(Qbar.t() * Q * V.col(k));
    //     state.elem(arma::find( arma::abs(state) < 1e-13) ).zeros();
    //     if(state.is_zero()) {
    //         states_info[k] = m + " Q+";
    //         printSeparated(std::cout, "\t", 16, true, E1(k), m * int(dim / 2));
    //     } else 
    //         printSeparated(std::cout, "\t", 16, true, E1(k), state.t());
    // }
    // std::cout << "--------------------- C_L†|X> ------------------------------\n" << std::endl;
    // for(long k = 0; k < dim; k++){
    //     arma::vec state = arma::real((Qbar.t() * Q).t() * V.col(k));
    //     state.elem(arma::find( arma::abs(state) < 1e-13) ).zeros();
    //     if(state.is_zero()) {
    //         states_info[k] = m + " Q+";
    //         printSeparated(std::cout, "\t", 16, true, E1(k), m * int(dim / 2));
    //     } else 
    //         printSeparated(std::cout, "\t", 16, true, E1(k), state.t());
    // }
    // std::cout << "----------------------------------------------------------\n" << std::endl;
    // for(long k = 0; k < dim; k++)
    //     printSeparated(std::cout, "\t", 16, true, E1(k), states_info[k]);
        
    arma::vec E2 = arma::eig_sym(H_q);
    arma::vec E2_1 = arma::eig_sym(H1);
    arma::vec E2_2 = arma::eig_sym(H2);

    printSeparated(std::cout, "\t", 14, true, "H", "QQ† + Q†Q", "diff", "\t", "QQ†", "Q†Q");//, "\t---------", "\t", "H", "H_q", "diff", "\n");
    for(long k = 0; k < dim; k++)
        printSeparated(std::cout, "\t", 14, true, E1(k), E2(k), std::abs(E1(k) - E2(k)), "\t", E2_1(k), E2_2(k));//, "\t---------", "\t", E1(k), E3(k), std::abs(E1(k) - E3(k)));
    
    



    std::cout << " - - - - - - FINISHED DIAGONALIZATION OF SUPERHAMILTONIAN IN : " << tim_s(starter) << " seconds - - - - - - " << std::endl; // simulation end
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
    case 2:
        diagonal_matrix_elements();
        break;
	default:
		#define generate_scaling_array(name) arma::linspace(this->name, this->name + this->name##s * (this->name##n - 1), this->name##n);
		auto L_list = generate_scaling_array(L);
		auto J1_list = generate_scaling_array(J1);
		auto J2_list = generate_scaling_array(J2);
		auto delta1_list = generate_scaling_array(delta1);
		auto delta2_list = generate_scaling_array(delta2);
		auto eta1_list = generate_scaling_array(eta1);
		auto eta2_list = generate_scaling_array(eta2);
		auto hz_list = generate_scaling_array(hz);
        auto hx_list = generate_scaling_array(hz);
		auto w_list = generate_scaling_array(w);

		for (auto& system_size : L_list){
        for (auto& J1x : J1_list){
        for (auto& J2x : J2_list){
        for (auto& eta1x : eta1_list){
        for (auto& eta2x : eta2_list){
        for (auto& delta1x : delta1_list){
        for (auto& delta2x : delta2_list){
        for (auto& hxx : hx_list){
        for (auto& hzx : hz_list){
        for (auto& wx : w_list)
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
                                        //diagonal_matrix_elements();
                                        this->diagonalize();
                                        //this->eigenstate_entanglement();

                                    };
            loopSymmetrySectors(kernel); continue;


            this->reset_model_pointer();
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
        auto symmetric_model = std::make_unique<QHamSolver<XYZsym>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
                                                                        this->hx, this->hz, k, p, zz, zx, this->add_edge_fields);
        symmetric_model->diagonalization(false);
        arma::vec E = symmetric_model->get_eigenvalues();
        
        Esym.insert(Esym.end(), std::make_move_iterator(E.begin()), std::make_move_iterator(E.end()));
        v_1d<std::string> temp_str = v_1d<std::string>(E.size(), "k=" + std::to_string(k) + ",p=" + to_string(p) + ",zx=" + to_string(zx) + ",zz=" + to_string(zz));
		symms.insert(symms.end(), std::make_move_iterator(temp_str.begin()), std::make_move_iterator(temp_str.end()));
    };
    loopSymmetrySectors(kernel);

    auto full_model = std::make_unique<QHamSolver<XYZ>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
                                                            this->hx, this->hz, 0, this->add_edge_fields);
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
    auto full_model = std::make_unique<QHamSolver<XYZ>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
                                                            this->hx, this->hz, 0, this->add_edge_fields);
    arma::sp_mat Hfull = full_model->get_hamiltonian();
    const u64 dim = full_model->get_hilbert_size();
    arma::sp_cx_mat H(dim, dim);
    auto kernel = [&](int k, int p, int zx, int zz)
    {
        auto symmetric_model = std::make_unique<QHamSolver<XYZsym>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
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
            printSeparated(std::cout, "\t", 16, true, std::vector<bool>(this->L, idx), std::vector<bool>(this->L, state), val);
        }
        std::cout << std::endl;
    }
}


// -------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------- IMPLEMENTATION OF UI

/// @brief Create unique pointer to model with current parameters in class
typename ui::model_pointer ui::create_new_model_pointer(){
    #ifdef USE_SYMMETRIES
        return std::make_unique<QHamSolver<XYZsym>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
                                                    this->hx, this->hz, this->syms.k_sym, this->syms.p_sym, this->syms.zx_sym, this->syms.zz_sym, this->add_edge_fields);
    #else
        return std::make_unique<QHamSolver<XYZ>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
                                                    this->hx, this->hz, this->add_parity_breaking, this->add_edge_fields);// this->w, this->seed); 
    #endif
}

/// @brief Reset member unique pointer to model with current parameters in class
void ui::reset_model_pointer(){
    #ifdef USE_SYMMETRIES
        return this->ptr_to_model.reset(new QHamSolver<XYZsym>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
                                                    this->hx, this->hz, this->syms.k_sym, this->syms.p_sym, this->syms.zx_sym, this->syms.zz_sym, this->add_edge_fields)); 
    #else
        return this->ptr_to_model.reset(new QHamSolver<XYZ>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
                                                            this->hx, this->hz, this->add_parity_breaking, this->add_edge_fields));// this->w, this->seed)); 
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
          << "\u03B71s  = " << this->eta1 << std::endl
          << "\u03B71n  = " << this->eta1 << std::endl
		  << "\u03B72  = " << this->eta2 << std::endl
          << "\u03B72s  = " << this->eta1 << std::endl
          << "\u03B72n  = " << this->eta1 << std::endl

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
    #else
		 std::cout  << "seed  = " << this->seed << std::endl
		  << "realisations  = " << this->realisations << std::endl
		  << "honid  = " << this->jobid << std::endl
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
    //     auto symmetric_model = std::make_unique<QHamSolver<XYZsym>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2,
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





    // auto make_hamil = [&](int size, arma::mat q, int p_sym) 
    //     -> arma::sp_mat
    // {

    //     u64 dim = ULLPOW(size);
    //     arma::sp_mat H(dim, dim);
    //     arma::cx_mat e(2,2);   e(0,0) = 1.0;   e(1,1) = 1.0;
    //     arma::cx_mat ham;
    //     for(int j = 0; j < size - 1; j++){
    //         ham = -arma::kron(e, q.t()) * arma::kron(q, e) - arma::kron(q.t(), e) * arma::kron(e, q) 
    //                             + q*q.t() + 1. / 2. * (arma::kron(e, q.t() * q) + arma::kron(q.t() * q, e));
    //         u64 dim_left = ULLPOW(j);
    //         u64 dim_right = ULLPOW(size - j - 2);
    //         ham = arma::kron(arma::kron(arma::eye<arma::cx_mat>(dim_left, dim_left), ham), arma::eye<arma::cx_mat>(dim_right, dim_right));
    //         H += ham;
    //     }
    //     if(this->boundary_conditions){
    //         u64 dim_rest = ULLPOW(size - 1);
    //         ham = 1. / 2. * ( arma::kron(arma::eye<arma::cx_mat>(dim_rest, dim_rest), q.t() * q) + arma::kron(q.t() * q, arma::eye<arma::cx_mat>(dim_rest, dim_rest)) );
    //         return (H + arma::sp_cx_mat(ham)) / 2.0;
    //     } else {
    //         int k_sec = size % 2 == 0? size / 2 : 0;
    //         std::vector<op::genOp> sym_gen;
    //         sym_gen.emplace_back(op::_parity_symmetry(size, p_sym));
    //         sym_gen.emplace_back(op::_spin_flip_z_symmetry(size, this->syms.zz_sym));
            
    //         // if(this->hz == 0 && !this->add_edge_fields)
    //         //     if(size % 2 == 0 || this->hx != 0) // for odd system sizes enter only if previous symmetry not taken
    //         //         sym_gen.emplace_back(op::_spin_flip_x_symmetry(size, this->syms.zx_sym));

    //         auto _hilbert_space = point_symmetric(size, sym_gen, this->boundary_conditions, k_sec);
    //         auto U = _hilbert_space.symmetry_rotation();
    //         auto T = op::_translation_symmetry(size, k_sec, true).to_matrix(dim);
    //         H += T * ham * T.t();
    //         return U.t() * H * U / 2.0;// * std::sqrt(double(size) / double(size + 1.0));
    //     }
    // };

};