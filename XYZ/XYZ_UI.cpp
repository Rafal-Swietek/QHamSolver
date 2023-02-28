#include "includes/XYZ_UI.hpp"

int outer_threads = 1;
int num_of_threads = 1;

namespace XYZ_UI{

void ui::make_sim(){
    printAllOptions();
    
    std::vector<std::string> res, sym_gen = {"P", "Zx", "Zy", "Zz"};
    const int NUM_OF_GENERATORS = sym_gen.size();
	for (int k = 0; k < NUM_OF_GENERATORS; k++) { // loop over all symmetries (no combinations)
		auto sym = sym_gen[k];
		for (int i = 0; i < NUM_OF_GENERATORS - k; i++) { // length of product
			auto sym_temp = sym;
			for (int j = k + 1; j < k + 1 + i; j++) // product off all possible lengths: 1,12,123,1234,... with no repetition starting at k
				sym_temp = sym_temp + " " + sym_gen[j];
			std::cout << sym_temp << std::endl;
		}
	}

    return;
    this->ptr_to_model = std::make_shared<QHamSolver<XYZ>>( this->boundary_conditions,  //<! boundary condition
                                                            this->L,                    //<! system size
                                                            this->J1,                   //<! coupling of grain to spins
                                                            this->J2,                   //<! coupling decay parameter
                                                            this->delta1,               //<! disorder on spins (bandwidth control)
                                                            this->delta2,               //<! uniform field on spins
                                                            this->eta1,                 //<! random seed
                                                            this->eta2,                 //<! size of ergodic grain
                                                            this->hx,                   //<! randomness on positions for decaying coupling
                                                            this->hz,                   //<! randomness on positions for decaying coupling
                                                            this->add_parity_breaking,  //<! randomness on positions for decaying coupling
                                                            this->w,                    //<! randomness on positions for decaying coupling
                                                            this->seed                  //<! randomness on positions for decaying coupling
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
		auto J1_list = generate_scaling_array(J1);
		auto J2_list = generate_scaling_array(J2);
		auto delta1_list = generate_scaling_array(delta1);
		auto delta2_list = generate_scaling_array(delta2);
		auto hz_list = generate_scaling_array(hz);
        auto hx_list = generate_scaling_array(hz);
		auto w_list = generate_scaling_array(w);

		for (auto& system_size : L_list){
			
            this->ptr_to_model = std::make_shared<QHamSolver<XYZ>>(this->boundary_conditions, this->L, this->J1, this->J2, this->delta1, this->delta2, this->eta1, this->eta2, 
                                                                    this->hx, this->hz, this->add_parity_breaking, this->w, this->seed); 
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

                                        std::cout << "\t\t - - - - - - FINISHED ITERATION IN : " << tim_s(start_loop) << " seconds\n\t\t\t Total time : " << tim_s(start) << " s - - - - - - " << std::endl; // simulation end
						}}}}}}}
            }
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
    user_interface<XYZ>::parse_cmd_options(argc, argv);

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
    user_interface<XYZ>::set_default();
    this->J1 = 1.0;
	this->J1s = 0.0;
	this->J1n = 1;
    this->J2 = 1.0;
	this->J2s = 0.0;
	this->J2n = 1;

    this->delta1 = 1.0;
	this->delta1s = 0.0;
	this->delta1n = 1;
    this->delta2 = 1.0;
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
	this->eta2 = 0.5;

    this->add_parity_breaking = 1;
}

/// @brief 
void ui::print_help() const {
    user_interface<XYZ>::print_help();

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

    printSeparated(std::cout, "\t", 20, true, "-w", "(double)", "disorder strength from uniform distribution");
    printSeparated(std::cout, "\t", 20, true, "-ws", "(double)", "step in disorder strength sweep");
    printSeparated(std::cout, "\t", 20, true, "-wn", "(int)", "number of disorder in the sweep");

    printSeparated(std::cout, "\t", 20, true, "-seed", "(u64)", "randomness in position for coupling to grain");
    printSeparated(std::cout, "\t", 20, true, "-pb", "(boolean)", "add parity breaking term on edge (when no disorder, i.e. w=0)");
	std::cout << std::endl;
}

/// @brief 
void ui::printAllOptions() const{
    user_interface<XYZ>::printAllOptions();
    std::cout << "H = \u03A3_r J_r\u03A3_i [ (1-\u03B7_r) S^x_i S^x_i+1 + (1+\u03B7_r) S^y_i S^y_i+1) + \u0394_r S^z_iS^z_i+1] + \u03A3_i h^z_i S^z_i + h^x\u03A3_i S^x_i" << std::endl << std::endl;
	std::cout << "h_i \u03B5 [hz - w, hz + w]" << std::endl;

	std::cout << "------------------------------ CHOSEN QuantumSun OPTIONS:" << std::endl;
    std::cout 
		  << "J1  = " << this->J1 << std::endl
		  << "J1n = " << this->J1n << std::endl
		  << "J1s = " << this->J1s << std::endl

		  << "J2  = " << this->J2 << std::endl
		  << "J2n = " << this->J2n << std::endl
		  << "J2s = " << this->J2s << std::endl

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
		  << "hzs = " << this->hzs << std::endl
		  << "w  = " << this->w << std::endl
		  << "ws = " << this->ws << std::endl
		  << "wn = " << this->wn << std::endl
		  << "add parity breaking term = " << this->add_parity_breaking << std::endl;
    printSeparated(std::cout, "\t", 16, true, "----------------------------------------------------------------------------------------------------");
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
            ",e1=" + to_string_prec(this->eta1) + \
            ",e2=" + to_string_prec(this->eta2) + \
            ",hx=" + to_string_prec(this->hx) + \
            ",hz=" + to_string_prec(this->hz) + \
            ",w=" + to_string_prec(this->w) + \
            ",pb=" + std::to_string((int)this->add_parity_breaking);

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