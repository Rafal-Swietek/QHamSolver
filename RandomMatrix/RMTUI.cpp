#include "includes/RMTUI.hpp"

int outer_threads = 1;
int num_of_threads = 1;

bool normalize_grain = 1;

namespace QSunUI{

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
		multifractality();
		break;
	default:
		#define generate_scaling_array(name) arma::linspace(this->name, this->name + this->name##s * (this->name##n - 1), this->name##n);
		
        std::cout << "Add default function" << std::endl;
	}
	std::cout << " - - - - - - FINISHED CALCULATIONS IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
}






// -------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------- IMPLEMENTATION OF UI

/// @brief Create unique pointer to model with current parameters in class
typename ui::model_pointer ui::create_new_model_pointer(){
    return std::make_unique<QHamSolver<RandomMatrix>>(this->dim, this->b, this->alfa, this->gamma, this->seed);
}

/// @brief Reset member unique pointer to model with current parameters in class
void ui::reset_model_pointer(){
    this->ptr_to_model.reset(new QHamSolver<RandomMatrix>(this->dim, this->b, this->alfa, this->gamma, this->seed)); 
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
    user_interface_dis<RandomMatrix>::parse_cmd_options(argc, argv);

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
    
    set_param(b);
    set_param(gamma);
    _set_param_(alfa, true); // set always positive
	this->saving_dir = "." + kPSep + "results" + kPSep;
}


/// @brief 
void ui::set_default(){
    user_interface_dis<RandomMatrix>::set_default();
    this->b = 1.0;
	this->bs = 0.0;
	this->bn = 1;
	
	this->gamma = 1.0;
	this->gammas = 0.2;
	this->gamman = 1;

	this->alfa = 1.0;
	this->alfas = 0.02;
	this->alfan = 1;
}

/// @brief 
void ui::print_help() const {
    user_interface_dis<RandomMatrix>::print_help();
    
    printf(" Flags for Quantum Sun model:\n");
    printSeparated(std::cout, "\t", 20, true, "-dim", "(size_t)", "matrix size");
    printSeparated(std::cout, "\t", 20, true, "-b", "(double)", "coupling strength");
    printSeparated(std::cout, "\t", 20, true, "-bs", "(double)", "step in coupling strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-bn", "(int)", "number of couplings in the sweep");
    printSeparated(std::cout, "\t", 20, true, "-gamma", "(double)", "strength of ergodic bubble");

    printSeparated(std::cout, "\t", 20, true, "-alfa", "(double)", "decay control of coupling with distance");
    printSeparated(std::cout, "\t", 20, true, "-alfas", "(double)", "step in decay strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-alfan", "(int)", "number of values in the sweep");
	std::cout << std::endl;
}

/// @brief 
void ui::printAllOptions() const{
    user_interface_dis<RandomMatrix>::printAllOptions();
    std::cout << "QUANTUM SUN:\n\t\t" << "H = \u03B3R + J \u03A3_i \u03B1^{u_i} S^x_i S^x_i+1 + \u03A3_i h_i S^z_i" << std::endl << std::endl;
	std::cout << "u_i \u03B5 [j - \u03B6, j + \u03B6]"  << std::endl;
	std::cout << "h_i \u03B5 [h - w, h + w]" << std::endl;

	std::cout << "------------------------------ CHOSEN RandomMatrix OPTIONS:" << std::endl;
    std::cout 
		  << "Matrix Size = " << this->dim << std::endl
		  << "b  = " << this->b << std::endl
		  << "bn = " << this->bn << std::endl
		  << "bs = " << this->bs << std::endl
		  << "\u03B3 = " << this->gamma << std::endl
		  << "\u03B1  = " << this->alfa << std::endl
		  << "\u03B1s = " << this->alfas << std::endl
		  << "\u03B1n = " << this->alfan << std::endl;
}   

/// @brief 
/// @param skip 
/// @param sep 
/// @return 
std::string ui::set_info(std::vector<std::string> skip, std::string sep) const
{
        std::string name = "D=" + std::to_string(this->dim); + \
        //     ",N=" + std::to_string(this->grain_size) + \
        //     ",b=" + to_string_prec(this->b) + \
        //     ",g=" + to_string_prec(this->gamma);
        // if(this->alfa < 1.0) name += ",zeta=" + to_string_prec(this->zeta);
        
		// name += ",alfa=" + to_string_prec(this->alfa) + \
        //     ",h=" + to_string_prec(this->h) + \
        //     ",w=" + to_string_prec(this->w);
        // if(this->initiate_avalanche) name += ",ini_ave";

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