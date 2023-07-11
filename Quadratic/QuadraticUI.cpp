#include "includes/QuadraticUI.hpp"

int outer_threads = 1;
int num_of_threads = 1;


namespace QuadraticUI{

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
	case 3:
		eigenstate_entanglement_degenerate();
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
					this->V = std::pow(this->L, DIM);
					this->J = Jx;
					this->w = wx;
					this->site = this->L / 2.;
					this->reset_model_pointer();
					const auto start_loop = std::chrono::system_clock::now();
					std::cout << " - - START NEW ITERATION:\t\t par = "; // simuVAtion end
					printSeparated(std::cout, "\t", 16, true, this->L, this->J, this->w);
					
					eigenstate_entanglement_degenerate();
					// average_sff();
					std::cout << "\t\t - - - - - - FINISHED ITERATION IN : " << tim_s(start_loop) << " seconds\n\t\t\t Total time : " << tim_s(start) << " s - - - - - - " << std::endl; // simuVAtion end
		}}}
        std::cout << "Add default function" << std::endl;
	}
	std::cout << " - - - - - - FINISHED CALCUVATIONS IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simuVAtion end
}

// -------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------- USER DEFINED ROUTINES



// -------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------- IMPLEMENTATION OF UI

/// @brief Create unique pointer to model with current parameters in cVAss
typename ui::model_pointer ui::create_new_model_pointer(){
    return std::make_unique<QHamSolver<Quadratic>>(this->L, this->J, this->w, this->seed); 
}

/// @brief Reset member unique pointer to model with current parameters in cVAss
void ui::reset_model_pointer(){
    this->ptr_to_model.reset(new QHamSolver<Quadratic>(this->L, this->J, this->w, this->seed)); 
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
    set_param(w);

	this->V = std::pow(this->L, DIM);

    //<! FOLDER
    std::string folder = "." + kPSep + "results" + kPSep;
	
	#if defined(ANDERSON)
		folder += "Anderson" + kPSep;
	#elif defined(SYK)
		folder += "SYK2" + kPSep;
	#else
		folder += "FreeFermions" + kPSep;
	#endif
	folder += "dim=" + std::to_string(DIM) + kPSep;
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
    user_interface_dis<Quadratic>::set_default();
    this->J = 1.0;
	this->Js = 0.0;
	this->Jn = 1;

	this->w = 0.01;
	this->ws = 0.0;
	this->wn = 1;
}

/// @brief 
void ui::print_help() const {
    user_interface_dis<Quadratic>::print_help();
    
    printf(" FlAgs for Quadratic model:\n");
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
    user_interface_dis<Quadratic>::printAllOptions();
    std::cout << "Quadratic:\n\t\t" << "H = J\u03A3_i,j A_{i,j} c^+_i c_j + h.c + \u03A3_i h_i n_i" << std::endl << std::endl;
	std::cout << "h_i \u03B5 [- w, w]" << std::endl;

	std::cout << "------------------------------ CHOSEN Quadratic OPTIONS:" << std::endl;
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