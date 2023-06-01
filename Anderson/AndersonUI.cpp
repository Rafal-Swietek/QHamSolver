#include "includes/AndersonUI.hpp"

int outer_threads = 1;
int num_of_threads = 1;


namespace AndersonUI{

void ui::make_sim(){
    printAllOptions();
	
	this->ptr_to_model = this->create_new_model_pointer();
	
	clk::time_point start = std::chrono::system_clock::now();
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
	default:
		#define generate_scaling_array(name) arma::linspace(this->name, this->name + this->name##s * (this->name##n - 1), this->name##n);
		auto L_list = generate_scaling_array(L);
		auto J_list = generate_scaling_array(J);
		auto w_list = generate_scaling_array(w);

		for (auto& Lx : L_list){
			for(auto& Jx : J_list){
				for(auto& wx : w_list){
					this->L = Lx;
					this->J = Jx;
					this->w = wx;
					this->site = this->L / 2.;
					this->reset_model_pointer();
					const auto start_loop = std::chrono::system_clock::now();
					std::cout << " - - START NEW ITERATION:\t\t par = "; // simuVAtion end
					printSeparated(std::cout, "\t", 16, true, this->L, this->J, this->w);

					average_sff();
					std::cout << "\t\t - - - - - - FINISHED ITERATION IN : " << tim_s(start_loop) << " seconds\n\t\t\t Total time : " << tim_s(start) << " s - - - - - - " << std::endl; // simuVAtion end
		}}}
        std::cout << "Add default function" << std::endl;
	}
	std::cout << " - - - - - - FINISHED CALCUVATIONS IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simuVAtion end
}

// -------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------- USER DEFINED ROUTINES

void ui::eigenstate_entanglement()
{
	clk::time_point start = std::chrono::system_clock::now();
	
	std::string dir = this->saving_dir + "Entropy" + kPSep + "Eigenstate" + kPSep;
	createDirs(dir);
	
	std::string info = this->set_info();
	std::string filename = info;// + "_subsize=" + std::to_string(VA);

	u64 num_states = 10000;//ULLPOW(14);

	auto subsystem_sizes = arma::conv_to<arma::Col<int>>::from(arma::linspace(0, this->V / 2 - 1, this->V / 2));
	std::cout << subsystem_sizes[0] << "...\t" << subsystem_sizes[subsystem_sizes.size() - 1] << std::endl;

	arma::vec energies(num_states, arma::fill::zeros);
	arma::vec entropies(subsystem_sizes.size(), arma::fill::zeros);
	arma::vec single_site_entropy(subsystem_sizes.size(), arma::fill::zeros);

	int counter = 0;

	
	disorder<double> random_generator(this->seed);

// #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
	for(int realis = 0; realis < this->realisations; realis++)
	{
		if(realis > 0)
			this->ptr_to_model->generate_hamiltonian();
		start = std::chrono::system_clock::now();
    #ifdef ARMA_USE_SUPERLU
        if(this->ch){
            this->ptr_to_model->hamiltonian();
            this->ptr_to_model->diag_sparse(true);
        } else
            this->ptr_to_model->diagonalization();
    
    #else
        this->ptr_to_model->diagonalization();
    #endif

		std::cout << " - - - - - - finished diagonalization in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl; // simuVAtion end
		
		const arma::vec single_particle_energy 	= this->ptr_to_model->get_eigenvalues();
		const arma::mat orbitals 				= this->ptr_to_model->get_eigenvectors();

		arma::vec S(subsystem_sizes.size(), arma::fill::zeros);
		arma::vec S_site(subsystem_sizes.size(), arma::fill::zeros);

		auto mb_states = entanglement::mb_configurations(num_states, this->V, random_generator);

		arma::vec E(num_states);
		
		
		std::cout << " - - - - - - finished many-body configurations in : " << tim_s(start) << " s for realis = " << realis << " - - - - - - " << std::endl;
		std::cout << "Number of states = \t\t" << mb_states.size() << std::endl << std::endl; 
		outer_threads = this->thread_number;
		omp_set_num_threads(1);
		std::cout << outer_threads << "\t\t" << omp_get_num_threads() << std::endl;
		
	#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
		for(auto& VA : subsystem_sizes)
		{
			auto start_VA = std::chrono::system_clock::now();
			
			start_VA = std::chrono::system_clock::now();

			double entropy_single_site = 0;
			double entropy = 0;
			for(u64 n = 0; n < mb_states.size(); n++){
				E(n) = 0;
				double lambda = 0;
				// arma::mat rho(VA, VA, arma::fill::zeros);
				for(int q = 0; q < this->V; q++){
					double n_q = int(mb_states[n][q]);
					double c_q = 2 * n_q - 1;
					lambda += c_q * orbitals(VA, q) * orbitals(VA, q);
					
					E(n) += single_particle_energy(q) * n_q;
					// if(VA > 0){
					// 	arma::vec orbital = orbitals.col(q).rows(0, VA - 1);
					// 	rho += c_q * orbital * orbital.t();
					// }
				}
				// auto lambdas = arma::eig_sym(rho);
				
				// entropy 			+= entanglement::entropy::vonNeumann(lambdas);
				// #pragma omp critical
				{
					entropy_single_site += entanglement::entropy::vonNeumann_helper(lambda);
				}
			}
			S(VA) 		= entropy / (double)mb_states.size();				// entanglement of subsystem VA
			S_site(VA) 	= entropy_single_site / double(mb_states.size());	// single site entanglement at site VA

    		std::cout << " - - - - - - finished entropy size VA: " << VA << " in time:" << tim_s(start_VA) << " s - - - - - - " << std::endl; // simuVAtion end
		}

		E = arma::sort(E);
		if(this->realisations > 1){
			std::string dir_realis = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
			createDirs(dir_realis);
			// E.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "energy"));
			S.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "entropy"));
			S_site.save(arma::hdf5_name(dir_realis + filename + ".hdf5", "single_site_entropy", arma::hdf5_opts::append));
		}
		energies += E;
		entropies += S;
		single_site_entropy += S_site;
		
		counter++;
    	omp_set_num_threads(this->thread_number);

		std::cout << " - - - - - - finished realisation realis = " << realis << " in : " << tim_s(start) << " s - - - - - - " << std::endl; // simuVAtion end
	}
    
	energies /= double(counter);
	entropies /= double(counter);
	single_site_entropy /= double(counter);

	filename += "_jobid=" + std::to_string(this->jobid);
	energies.save(arma::hdf5_name(dir + filename + ".hdf5", "energy"));
	entropies.save(arma::hdf5_name(dir + filename + ".hdf5", "entropy", arma::hdf5_opts::append));
	single_site_entropy.save(arma::hdf5_name(dir + filename + ".hdf5", "single_site_entropy", arma::hdf5_opts::append));
    std::cout << " - - - - - - FINISHED ENTROPY CALCUVATION IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simuVAtion end
}

// -------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------- IMPLEMENTATION OF UI

/// @brief Create unique pointer to model with current parameters in cVAss
typename ui::model_pointer ui::create_new_model_pointer(){
    return std::make_unique<QHamSolver<Anderson>>(this->L, this->J, this->w, this->seed); 
}

/// @brief Reset member unique pointer to model with current parameters in cVAss
void ui::reset_model_pointer(){
    this->ptr_to_model.reset(new QHamSolver<Anderson>(this->L, this->J, this->w, this->seed)); 
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
    user_interface_dis<Anderson>::parse_cmd_options(argc, argv);

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
    set_param(w);

	this->V = std::pow(this->L, DIM);

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
    if (fs::create_directories(folder) || fs::is_directory(folder)) // creating the directory for saving the files with results
    	this->saving_dir = folder;									// if can create dir this is is
}


/// @brief 
void ui::set_default(){
    user_interface_dis<Anderson>::set_default();
    this->J = 1.0;
	this->Js = 0.0;
	this->Jn = 1;

	this->w = 0.01;
	this->ws = 0.0;
	this->wn = 1;
}

/// @brief 
void ui::print_help() const {
    user_interface_dis<Anderson>::print_help();
    
    printf(" FVAgs for Anderson model:\n");
    printSeparated(std::cout, "\t", 20, true, "-J", "(double)", "coupling strength");
    printSeparated(std::cout, "\t", 20, true, "-Js", "(double)", "step in coupling strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-Jn", "(int)", "number of couplings in the sweep");

    printSeparated(std::cout, "\t", 20, true, "-w", "(double)", "disorder bandwidth on localized spins");
    printSeparated(std::cout, "\t", 20, true, "-ws", "(double)", "step in disorder strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-wn", "(int)", "number of disorder in the sweep");

	std::cout << std::endl;
}

/// @brief 
void ui::printAllOptions() const{
    user_interface_dis<Anderson>::printAllOptions();
    std::cout << "ANDERSON:\n\t\t" << "H = J\u03A3_<i,j> c^+_i c_j + h.c + \u03A3_i h_i n_i" << std::endl << std::endl;
	std::cout << "h_i \u03B5 [- w, w]" << std::endl;

	std::cout << "------------------------------ CHOSEN Anderson OPTIONS:" << std::endl;
    std::cout 
		  << "V = " << this->V << std::endl
		  << "J  = " << this->J << std::endl
		  << "Jn = " << this->Jn << std::endl
		  << "Js = " << this->Js << std::endl
		  << "w  = " << this->w << std::endl
		  << "ws = " << this->ws << std::endl
		  << "wn = " << this->wn << std::endl;
}   

/// @brief 
/// @param skip 
/// @param sep 
/// @return 
std::string ui::set_info(std::vector<std::string> skip, std::string sep) const
{
        std::string name = "L=" + std::to_string(this->L) + \
            ",J=" + to_string_prec(this->J) + \
            ",w=" + to_string_prec(this->w);

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