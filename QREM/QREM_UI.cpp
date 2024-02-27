#include "includes/QREM_UI.hpp"

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

namespace QREM_UI{



void ui::make_sim(){
    printAllOptions();

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
		spectral_form_factor();
		break;
    case 3:
        diagonal_matrix_elements();
        break;
	case 4:
		eigenstate_entanglement_degenerate();
		break;
	case 5:
		diagonal_matrix_elements();
		break;
	case 6:
		matrix_elements();
		break;
	default:
		auto L_list = arma::linspace(this->L, this->L + this->Ls * (this->Ln - 1), this->Ln);
        auto g_list = arma::linspace(this->g, this->g + this->gs * (this->gn - 1), this->gn);

        for(int system_size : L_list){ 
            for(double gx : g_list){   
                
        {
            this->L = system_size;
            this->g = gx;
            this->site = this->L / 2.;
            const auto start_loop = std::chrono::system_clock::now();
            std::cout << " - - START NEW ITERATION:\t\t par = "; // simulation end
            printSeparated(std::cout, "\t", 16, true, this->L, this->g);

            this->reset_model_pointer();
            
            this->matrix_elements(); continue;
            
            this->eigenstate_entanglement(); continue;

            diagonal_matrix_elements();
            std::cout << "\t\t - - - - - - FINISHED ITERATION IN : " << tim_s(start_loop) << " seconds\n\t\t\t Total time : " << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
        }
        }}
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
            ",g=" + to_string_prec(this->g);

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

// -------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------- IMPLEMENTATION OF UI

/// @brief Create unique pointer to model with current parameters in class
typename ui::model_pointer ui::create_new_model_pointer(){
    return std::make_unique<QHS::QHamSolver<QREM>>(this->L, this->g, this->seed); 
}

/// @brief Reset member unique pointer to model with current parameters in class
void ui::reset_model_pointer(){
    this->ptr_to_model.reset(new QHS::QHamSolver<QREM>(this->L, this->g, this->seed));
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
    user_interface_dis<QREM>::parse_cmd_options(argc, argv);

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
    set_param(g);

    //<! FOLDER
    std::string folder = this->dir_prefix + "results" + kPSep;
    
    if (fs::create_directories(folder) || fs::is_directory(folder)) // creating the directory for saving the files with results
    	this->saving_dir = folder;									// if can create dir this is is
}


/// @brief 
void ui::set_default(){
    user_interface_dis<QREM>::set_default();
    this->g = 1.0;
	this->gs = 0.0;
	this->gn = 1;
}

/// @brief 
void ui::print_help() const {
    user_interface_dis<QREM>::print_help();

    printf(" Flags for QREM model:\n");
    printSeparated(std::cout, "\t", 20, true, "-g", "(double)", "perturbation strength");
    printSeparated(std::cout, "\t", 20, true, "-gs", "(double)", "step in perturbation strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-gn", "(int)", "number of perturbations in the sweep");
    
	std::cout << std::endl;
}

/// @brief 
void ui::printAllOptions() const{
    user_interface_dis<QREM>::printAllOptions();
    printSeparated(std::cout, "\t", 16, true, "Hamiltonian:", "H = \u03A3_k=1^D E_n|k><k| + gV");
    printSeparated(std::cout, "\t", 16, true, "Perturbation:", "V = 1/sqrt{L ln(L)}\u03A3_i S^x_i");
    printSeparated(std::cout, "\t", 16, true, "random energies", "E_n is gaussian with sigma^2=L/2");

	std::cout << "------------------------------ CHOSEN QREM OPTIONS:" << std::endl;
    std::cout 
		  << "g  = " << this->g << std::endl
		  << "gn = " << this->gn << std::endl
		  << "gs = " << this->gs << std::endl
          << "seed  = " << this->seed << std::endl
		  << "realisations  = " << this->realisations << std::endl
		  << "jobid  = " << this->jobid << std::endl;
          std::cout << std::endl;
    printSeparated(std::cout, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
}   




};
