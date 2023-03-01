#include "includes/QSunUI.hpp"

int outer_threads = 1;
int num_of_threads = 1;

namespace QSunUI{

void ui::make_sim(){
    printAllOptions();
    bool normalize_grain = 1;

    this->ptr_to_model = std::make_shared<QHamSolver<QuantumSun>>(  this->L,                    //<! system size
                                                                    this->J,                    //<! coupling of grain to spins
                                                                    this->alfa,                 //<! coupling decay parameter
                                                                    this->gamma,                //<! strength of ergodic bubble
                                                                    this->w,                    //<! disorder on spins (bandwidth control)
                                                                    this->h,                    //<! uniform field on spins
                                                                    this->seed,                 //<! random seed
                                                                    this->grain_size,           //<! size of ergodic grain
                                                                    this->zeta,                 //<! randomness on positions for decaying coupling
                                                                    this->initiate_avalanche,   //<!  initiate avalanche with first coupling=1
																	normalize_grain				//<!  keep grain with unit HS norm
                                                                ); 
	
	clk::time_point start = std::chrono::system_clock::now();
    switch (this->fun)
	{
	case 0: 
		diagonalize(); 
		break;
	case 1:
		spectral_form_factor();
		break;
	default:
		#define generate_scaling_array(name) arma::linspace(this->name, this->name + this->name##s * (this->name##n - 1), this->name##n);
		auto L_list = generate_scaling_array(L);
		auto J_list = generate_scaling_array(J);
		auto alfa_list = generate_scaling_array(alfa);
		auto h_list = generate_scaling_array(h);
		auto w_list = generate_scaling_array(w);

		for (auto& system_size : L_list){
			this->ptr_to_model = std::make_shared<QHamSolver<QuantumSun>>(  this->L,            //<! system size
                                                                    this->J,                    //<! coupling of grain to spins
                                                                    this->alfa,                 //<! coupling decay parameter
                                                                    this->w,                    //<! disorder on spins (bandwidth control)
                                                                    this->h,                    //<! uniform field on spins
                                                                    this->seed,                 //<! random seed
                                                                    this->grain_size,           //<! size of ergodic grain
                                                                    this->zeta,                 //<! randomness on positions for decaying coupling
                                                                    this->initiate_avalanche    //<!  initiate avalanche with first coupling=1
                                                                );
			for (auto& alfax : alfa_list){
				for (auto& hx : h_list){
					for(auto& Jx : J_list){
						for(auto& wx : w_list){
							this->L = system_size;
							this->alfa = alfax;
							this->h = hx;
							this->J = Jx;
							this->w = wx;
							this->site = this->L / 2.;

							const auto start_loop = std::chrono::system_clock::now();
							std::cout << " - - START NEW ITERATION:\t\t par = "; // simulation end
							printSeparated(std::cout, "\t", 16, true, this->L, this->J, this->alfa, this->h, this->w);

							average_sff();
							std::cout << "\t\t - - - - - - FINISHED ITERATION IN : " << tim_s(start_loop) << " seconds\n\t\t\t Total time : " << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
						}}}}}
        std::cout << "Add default function" << std::endl;
	}
	std::cout << " - - - - - - FINISHED CALCULATIONS IN : " << tim_s(start) << " seconds - - - - - - " << std::endl; // simulation end
}





// -------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------- IMPLEMENTATION OF UI

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
    user_interface<QuantumSun>::parse_cmd_options(argc, argv);

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
    set_param(alfa);
    set_param(h);
    set_param(w);

    choosen_option = "-gamma";
    this->set_option(this->gamma, argv, choosen_option);

    choosen_option = "-zeta";
    this->set_option(this->zeta, argv, choosen_option);
    
    choosen_option = "-ini_ave";
    this->set_option(this->initiate_avalanche, argv, choosen_option);

    choosen_option = "-M";
    this->set_option(this->grain_size, argv, choosen_option);

    this->saving_dir = "." + kPSep + "results" + kPSep;
}


/// @brief 
void ui::set_default(){
    user_interface<QuantumSun>::set_default();
    this->J = 1.0;
	this->Js = 0.0;
	this->Jn = 1;

	this->zeta = 0.2;
	this->gamma = 1.0;

	this->h = 0.0;
	this->hs = 0.1;
	this->hn = 1;

	this->w = 0.01;
	this->ws = 0.0;
	this->wn = 1;

	this->alfa = 1.0;
	this->alfas = 0.02;
	this->alfan = 1;

	this->grain_size = 1;
    this->initiate_avalanche = 0;
}

/// @brief 
void ui::print_help() const {
    user_interface<QuantumSun>::print_help();
    
    printf(" Flags for Quantum Sun model:\n");
    printSeparated(std::cout, "\t", 20, true, "-J", "(double)", "coupling strength");
    printSeparated(std::cout, "\t", 20, true, "-Js", "(double)", "step in coupling strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-Jn", "(int)", "number of couplings in the sweep");
    printSeparated(std::cout, "\t", 20, true, "-gamma", "(double)", "strength of ergodic bubble");

    printSeparated(std::cout, "\t", 20, true, "-alfa", "(double)", "decay control of coupling with distance");
    printSeparated(std::cout, "\t", 20, true, "-alfas", "(double)", "step in decay strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-alfan", "(int)", "number of values in the sweep");

    printSeparated(std::cout, "\t", 20, true, "-h", "(double)", "uniform field on spins");
    printSeparated(std::cout, "\t", 20, true, "-hs", "(double)", "step in field strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-hn", "(int)", "number of field values in the sweep");

    printSeparated(std::cout, "\t", 20, true, "-w", "(double)", "disorder bandwidth on localized spins");
    printSeparated(std::cout, "\t", 20, true, "-ws", "(double)", "step in disorder strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-wn", "(int)", "number of disorder in the sweep");

    printSeparated(std::cout, "\t", 20, true, "-zeta", "(double)", "randomness in position for coupling to grain");
    printSeparated(std::cout, "\t", 20, true, "-M", "(int)", "size of random grain (number of spins inside grain)");
    printSeparated(std::cout, "\t", 20, true, "-ini_ave", "(boolean)", "initiate avalanche by hand");
	std::cout << std::endl;
}

/// @brief 
void ui::printAllOptions() const{
    user_interface<QuantumSun>::printAllOptions();
    std::cout << "QUANTUM SUN:\n\t\t" << "H = \u03B3R + J \u03A3_i \u03B1^{u_i} S^x_i S^x_i+1 + \u03A3_i h_i S^z_i" << std::endl << std::endl;
	std::cout << "u_i \u03B5 [j - \u03B6, j + \u03B6]"  << std::endl;
	std::cout << "h_i \u03B5 [h - w, h + w]" << std::endl;

	std::cout << "------------------------------ CHOSEN QuantumSun OPTIONS:" << std::endl;
    std::cout 
		  << "grain size = " << this->grain_size << std::endl
		  << "J  = " << this->J << std::endl
		  << "Jn = " << this->Jn << std::endl
		  << "Js = " << this->Js << std::endl
		  << "\u03B3 = " << this->gamma << std::endl
		  << "h  = " << this->h << std::endl
		  << "hs = " << this->hs << std::endl
		  << "hn = " << this->hn << std::endl
		  << "w  = " << this->w << std::endl
		  << "ws = " << this->ws << std::endl
		  << "wn = " << this->wn << std::endl
		  << "\u03B1  = " << this->alfa << std::endl
		  << "\u03B1s = " << this->alfas << std::endl
		  << "\u03B1n = " << this->alfan << std::endl
		  << "\u03B6 = " << this->zeta << std::endl
		  << "initialize avelanche = " << this->initiate_avalanche << std::endl;
}   

/// @brief 
/// @param skip 
/// @param sep 
/// @return 
std::string ui::set_info(std::vector<std::string> skip, std::string sep) const
{
        std::string name = "L=" + std::to_string(this->L) + \
            ",M=" + std::to_string(this->grain_size) + \
            ",J=" + to_string_prec(this->J) + \
            ",g=" + to_string_prec(this->gamma) + \
            ",zeta=" + to_string_prec(this->zeta) + \
            ",alfa=" + to_string_prec(this->alfa) + \
            ",h=" + to_string_prec(this->h) + \
            ",w=" + to_string_prec(this->w) + \
            ",ini_ave=" + std::to_string((int)this->initiate_avalanche);

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