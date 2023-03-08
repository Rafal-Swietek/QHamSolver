#pragma once

#include "QHamSolver.h"

#include "statistics.hpp"
#include "statistics_dist.hpp"
#include "spectrals.hpp"
#include "thermodynamics.hpp"
#include "entanglement.hpp"
#include "adiabatic_gauges.hpp"

extern int outer_threads;

inline
std::vector<std::string> 
change_input_to_vec_of_str(int argc, char** argv)
{
	// -1 because first is the name of the file
	std::vector<std::string> tmp(argc - 1, "");
	for (int i = 0; i < argc - 1; i++)
		tmp[i] = argv[i + 1];
	return tmp;
}

//<! -------------------------------------------------------------------------------------------
//<! ----------------------------------------------------------------------------- BASE UI CLASS
template <class Hamiltonian>
class user_interface {
	static_check((std::is_base_of_v<_hamiltonian, Hamiltonian>), 
                    "\n" BAD_INHERITANCE "\n\t base class is: hamiltonian_base<element_type, hilbert_space>");
protected:
	unsigned int thread_number = 1;									// number of threads
	int boundary_conditions = 1;									// boundary conditions - 0 - PBC, 1 - OBC, 2 - ABC,...
	std::string saving_dir = "";									// directory for files to be saved onto

	// ----------------------------------- FUNCTIONS FOR READING THE INPUT

	std::string getCmdOption(const v_1d<std::string>& vec, std::string option) const;					 	// get the option from cmd input

	// ----------------------------------- TEMPLATES
	typedef std::unique_ptr<QHamSolver<Hamiltonian>> model_pointer;
    typedef typename Hamiltonian::element_type element_type;
	
	template <typename _ty>
	void set_option(_ty& value, const v_1d<std::string>& argv, 
                        std::string choosen_option, bool geq_0 = false);					        // set an option

	//template <typename _ty>
	//void set_default_msg(_ty& value, string option, string message, 
    //                                const unordered_map <string, string>& map) const;	    // setting value to default and sending a message

    unsigned int L, Ls, Ln;								// lattice params
    bool ch;											// boolean values
    int realisations;									// number of realisations to average on for disordered case - symmetries got 1
    size_t seed;										// radnom seed for random generator
    int jobid;											// unique _id given to current job

    double q_ipr;										// q for participation ratio calculation
    double beta;										// inverse temperature
    
	int mu;												// small bucket for the operator fluctuations to be averaged onto
	int num_of_points;									// number of points for time evolution/spectral functions/SFF/etc..
    
	int site;											// site for operator averages
    int op;												// choose operator
    int fun;											// choose function to start calculations

	model_pointer ptr_to_model;
public:
	virtual ~user_interface() = default;

	// ----------------------------------- HELPING FUNCIONS
	virtual void set_default();			
	virtual void print_help() const;													// set default parameters

	/// @brief Exits the program and prints description of all UI parameters.
	virtual void exit_with_help() const
		{ this->print_help(); std::exit(1); }
	
	virtual void printAllOptions() const;
	virtual std::string set_info(std::vector<std::string> skip = {}, 
										std::string sep = "_") const = 0;
	
	// ------------------------------------------------- REAL PARSING
	virtual void parse_cmd_options(int argc, std::vector<std::string> argv);			// the function to parse the command line

	// ------------------------------------------------- NON-VIRTUALS

	std::vector<std::string> parse_input_file(string filename) const;					// if the input is taken from file we need to make it look the same way as the command line does

	// ------------------------------------------------- SIMULATIONS
	virtual model_pointer create_new_model_pointer() = 0;
	virtual void reset_model_pointer() = 0;

	virtual void make_sim() = 0;			//<! main simulation funciton

	// ------------------------------------------------- MAIN ROUTINES
	template <
		typename callable,	//<! callable lambda function
		typename... _types	//<! argument-types passed to lambda
		> 
	void average_over_realisations(
		callable& lambda, 			//!< callable function
		_types... args				//!< arguments passed to callable interface lambda
	){
	#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
		for(int r = 0; r < this->realisations; r++){
			auto dummy_lambda = [&lambda](int real, auto... args){
				lambda(real, args...);
			};
			dummy_lambda(r, args...);
		}
    };

	auto get_eigenvalues(std::string _suffix = "");
	virtual void diagonalize();
	virtual void spectral_form_factor();
	virtual void average_sff();
	virtual void eigenstate_entanglement() = 0;
};

// MOVE TO USER_INTERFACE_IMPL.HPP

// -------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------- IMPLEMENTATION OF UI

/// @brief Get value of option in UI from command line
/// @tparam Hamiltonian template parameter for current used model
/// @param vec all values in cmd line
/// @param option chosen option to get
/// @return value of given option as string
template <class Hamiltonian>
std::string user_interface<Hamiltonian>::getCmdOption(const v_1d<std::string> &vec, std::string option) const
{
	if (auto itr = std::find(vec.begin(), vec.end(), option); itr != vec.end() && ++itr != vec.end())
		return *itr;
	return std::string();
}

/// @brief Set parameter (member of UI class) from input value
/// @tparam Hamiltonian template parameter for current used model
/// @tparam _ty type of value
/// @param value reference to parameter
/// @param argv input array from comand line
/// @param choosen_option option to be set as std::string
/// @param geq_0 controls if value should be positive
template <class Hamiltonian>
template <typename _ty>
void user_interface<Hamiltonian>::set_option(_ty &value, const v_1d<string> &argv, std::string choosen_option, bool geq_0)
{
	if (std::string option = this->getCmdOption(argv, choosen_option); option != "")
		value = static_cast<_ty>(std::stod(option)); // set value to an option

	if (geq_0 && value < 0){					  // if the variable shall be bigger equal 0
		//this->set_default_msg(value, choosen_option.substr(1),
		//					  choosen_option + " cannot be negative\n", isingUI::table);
		std::cout << "\t--->" + choosen_option + " cannot be negative:\t Taking absolute value!" << std::endl;
		if (std::string option = this->getCmdOption(argv, choosen_option); option != "")
			value = -static_cast<_ty>( std::stod(option) ); // set value to an option
	}	
}

/// @brief Read all values form input file to vector
/// @tparam Hamiltonian template parameter for current used model
/// @param filename name of input file (with directory relative to current dir)
/// @return vector of all flags from input file
template <class Hamiltonian>
std::vector<std::string> user_interface<Hamiltonian>::parse_input_file(std::string filename) const
{
	std::vector<std::string> commands(1, "");
	std::ifstream inputFile(filename);
	std::string line = "";
	if (!inputFile.is_open())
		std::cout << "Cannot open a file " + filename + " that I could parse. All parameters are default. Sorry :c \n";
	else
	{
		if (std::getline(inputFile, line))
		{
			commands = split_str(line, " "); // saving lines to out vector if it can be done, then the parser shall treat them normally
		}
	}
	return std::vector<std::string>(commands.begin(), commands.end());
}

/// @brief Prints all general UI option decriptions
/// @tparam Hamiltonian Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface<Hamiltonian>::print_help() const {
	printf(
		" Usage: name of executable [options] outputDirectory \n"
		" The input can be both introduced with [options] described below or with giving the input directory(which also is the flag in the options)\n"
		" options:\n"
		"-f input file for all of the options : (default none)\n"
		"-mu bucket size for ergodic coefficients (default 5)\n"
		"-L system size length minimum: bigger than 0 (default 8)\n"
		"-Ls system size length step: bigger equal than 0 (default 0)\n"
		"-Ln system size length number: bigger than 0 (default 1)\n"
		"-b boundary conditions : bigger than 0 (default 0 - PBC)\n"
		"	0 -- PBC\n"
		"	1 -- OBC\n"
		"	2 -- ABC -- none so far implemented\n"
		"-s site to act with local operators (default 0)\n"
		"-op flag to choose operator: \n"
		"	0 -- Sz_i-local\n"
		"	1 -- Sx_i-local\n"
		"	2 -- Hi\n"
		"	3 -- Sz_q\n"
		"	4 -- Sx_q\n"
		"	5 -- Hq\n"
		"	6 -- I+_n - TFIM LIOMs ordered by locality\n"
		"	7 -- I-_n - TFIM LIOMs ordered by locality\n"
		"	8 -- J    - spin current\n"
		"	9 -- Sx    - global X-magnetization\n"
		"	10 -- Sz   - global Z-magnetization\n"
		"	11 -- sum_i Sx_i Sx_i+1   - nearest neighbour X\n"
		"	12 -- sum_i Sz_i Sz_i+1   - nearest neighbour Z\n"
		"	13 -- sum_i Sx_i Sx_i+2   - next nearest neighbour X\n"
		"	14 -- sum_i Sz_i Sz_i+2   - next nearest neighbour Z\n"
		"	  -- to get sum of local Sz or Sx take Sz_q or Sx_q with -s=0\n"
		"	  -- i or q are set to site (flag -s); (default 0)\n"
		""
		"-fun choose function to start calculations: check user_interface.cpp -> make_sim() to find functions\n May be different in other models"
		"-th number of threads to be used for CPU parallelization : depends on the machine specifics, default(1)\n"
		"-ch general boolean flag used in different context (default: 0)\n"
		"-scale choose scale for data: either linear-0, log-1 or custom-2 (default: linear)\n"
		"-h quit with help\n"
	);
	std::cout << std::endl;
}

/// @brief Sets all UI parameters to default values
/// @tparam Hamiltonian Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface<Hamiltonian>::set_default(){
	this->saving_dir = "." + std::string(kPathSeparator) + "results" + std::string(kPathSeparator); // directory for the result files to be saved into
	this->L = 4;
	this->Ls = 1;
	this->Ln = 1;

	this->realisations = 100;
	this->site = 0;
	this->op = 0;
	this->fun = INT_MAX;
	this->mu = 5;
	
	this->q_ipr = 1.0;
	this->beta = 0.0;

	this->boundary_conditions = 0;
	this->thread_number = 1;

	this->ch = false;

	this->seed = static_cast<long unsigned int>(87178291199L);
	this->jobid = 0;
	this->num_of_points = 5000;
}

/// @brief Prints all general UI option values
/// @tparam Hamiltonian Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface<Hamiltonian>::printAllOptions() const {

	std::cout << "Chosen spin species is S = " << _Spin << std::endl << std::endl;
	std::cout << "------------------------------CHOSEN OPTIONS:" << std::endl;
	std::string opName = "";//std::get<0>(IsingModel_disorder::opName(this->op, this->site));
	std::cout << "DIR = " << this->saving_dir << std::endl
		  << "L  = " << this->L << std::endl
		  << "Ls = " << this->Ls << std::endl
		  << "Ln = " << this->Ln << std::endl
		  << "thread_num = " << this->thread_number << std::endl
		  << "site = " << this->site << std::endl
		  << "operator = " << opName << std::endl
		  << "bucket size = " << this->mu << std::endl
		  << "boolean value = " << this->ch << std::endl
		  << "realisations = " << this->realisations << std::endl
		  << "jobid = " << this->jobid << std::endl
		  << "seed = " << this->seed << std::endl
		  << "q_ipr = " << this->q_ipr << std::endl
		  << "\u03B2 = " << this->beta << std::endl;

	std::cout << "---------------------------------------------------------------------------------\n\n";
	#ifdef PRINT_HELP
		user_interface<Hamiltonian>::print_help();
		std::cout << "---------------------------------------------------------------------------------\n\n";
	#endif
	std::cout << "------------------------------CHOSEN MODEL:" << std::endl;
}

/// @brief Sets model parameters from values in command line
/// @tparam Hamiltonian Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface<Hamiltonian>::parse_cmd_options(int argc, std::vector<std::string> argv)
{
	// SET DEFAULT VALUES
	this->set_default(); // setting default at the very beginning

	std::string choosen_option = "";																// current choosen option

	//---------- SIMULATION PARAMETERS
	// system size
	choosen_option = "-L";
	this->set_option(this->L, argv, choosen_option, true);
	choosen_option = "-Ls";
	this->set_option(this->Ls, argv, choosen_option);
	choosen_option = "-Ln";
	this->set_option(this->Ln, argv, choosen_option, true);

	// boundary condition
	choosen_option = "-b";
	this->set_option(this->boundary_conditions, argv, choosen_option, true);
	if (this->boundary_conditions > 2)
		_assert_(false, "max boundary condition is 2");

	// choose site
	choosen_option = "-s";
	this->set_option(this->site, argv, choosen_option);

	// choose operator
	choosen_option = "-op";
	this->set_option(this->op, argv, choosen_option, true);
	
	// q for participation ration calculation
	choosen_option = "-q_ipr";
	this->set_option(this->q_ipr, argv, choosen_option);

	// boolean value
	choosen_option = "-ch";
	this->set_option(this->ch, argv, choosen_option, true);
	
	// disorder
	choosen_option = "-jobid";
	this->set_option(this->jobid, argv, choosen_option, true);
	choosen_option = "-seed";
	this->set_option(this->seed, argv, choosen_option, true);
	choosen_option = "-r";
	this->set_option(this->realisations, argv, choosen_option, true);

	//choose dimensionality
	choosen_option = "-beta";
	this->set_option(this->beta, argv, choosen_option);

	// choose function
	choosen_option = "-fun";
	this->set_option(this->fun, argv, choosen_option, true);

	// buckets
	choosen_option = "-mu";
	this->set_option(this->mu, argv, choosen_option, true);
	choosen_option = "-num_of_points";
	this->set_option(this->num_of_points, argv, choosen_option, true);

	// thread number
	choosen_option = "-th";
	this->set_option(this->thread_number, argv, choosen_option, true);
	this->thread_number /= outer_threads;
	if (this->thread_number > std::thread::hardware_concurrency())
		_assert_(false, "Wrong number of threads. Exceeding hardware\n");

	if(this->thread_number <= 0) this->thread_number = 1;
	omp_set_num_threads(this->thread_number);
	num_of_threads = this->thread_number;

	// get help
	choosen_option = "-help";
	if (std::string option = this->getCmdOption(argv, choosen_option); option != "")
		exit_with_help();

	std::cout << " - - - - - - MAKING ISING INTERFACE AND USING OUTER THREADS : "
			  << outer_threads << " - - - - - - " << endl; // setting the number of threads to be used with omp

	std::cout << " - - - - - - MAKING ISING INTERFACE AND USING INNER THREADS : "
			  << thread_number << " - - - - - - " << endl; // setting the number of threads to be used with omp

}

// -------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------- MAIN ROUTINES

/// @brief Diagonalize model hamiltonian and save spectrum to .hdf5 file
/// @tparam Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface<Hamiltonian>::diagonalize(){
	clk::time_point start = std::chrono::system_clock::now();
	std::string dir = this->saving_dir + "DIAGONALIZATION" + kPSep;
	createDirs(dir);

#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
	for(int realis = 0; realis < this->realisations; realis++)	
	{
		int real = realis + this->jobid;
		std::string _suffix = "_real=" + std::to_string(real);
		#ifdef USE_SYMMETRIES
			//<! no suffix for symmetric model
			_suffix = "";
		#endif
		std::string info = this->set_info({});
		std::cout << "\n\t\t--> finished creating model for " << info + _suffix << " - in time : " << tim_s(start) << "s" << std::endl;
		
		if(realis > 0)
			ptr_to_model->generate_hamiltonian();
		ptr_to_model->diagonalization();
		arma::vec eigenvalues = ptr_to_model->get_eigenvalues();
		
		std::cout << "\t\t	--> finished diagonalizing for " << info + _suffix << " - in time : " << tim_s(start) << "s" << std::endl;
		
		//std::cout << eigenvalues.t() << std::endl;

		std::string name = dir + info + _suffix + ".hdf5";
		eigenvalues.save(arma::hdf5_name(name, "eigenvalues", arma::hdf5_opts::append));
		std::cout << "\t\t	--> finished saving eigenvalues for " << info + _suffix << " - in time : " << tim_s(start) << "s" << std::endl;
		if(this->ch){
			auto H = ptr_to_model->get_dense_hamiltonian();
			H.save(arma::hdf5_name(name, "hamiltonian", arma::hdf5_opts::append));
			std::cout << "\t\t	--> finished saving Hamiltonian for " << info << " - in time : " << tim_s(start) << "s" << std::endl;

			auto V = ptr_to_model->get_eigenvectors();
			V.save(arma::hdf5_name(name, "eigenvectors", arma::hdf5_opts::append));
			std::cout << "\t\t	--> finished saving eigenvectors for " << info << " - in time : " << tim_s(start) << "s" << std::endl;
		}
	};
	
}


/// @brief Get eigenvalues from .hdf5 file if present else diagonalize model
/// @tparam Hamiltonian template parameter for current used model
/// @param _suffix string suffix used distinct different realisations
/// @return returns arma::vec of eigenvalues
template <class Hamiltonian>
auto user_interface<Hamiltonian>::get_eigenvalues(std::string _suffix) 
{
	arma::vec eigenvalues;
	std::string dir = this->saving_dir + "DIAGONALIZATION" + kPSep;
	createDirs(dir);
	std::string name = dir + this->set_info({});
	bool loaded;
	#pragma omp critical
	{
		loaded = eigenvalues.load(arma::hdf5_name(name + _suffix + ".hdf5", "eigenvalues/dataset"));
		if(!loaded)
			loaded = eigenvalues.load(arma::hdf5_name(name + ".hdf5", "eigenvalues/" + _suffix));
		if(!loaded){
			ptr_to_model->diagonalization(false);
			eigenvalues = ptr_to_model->get_eigenvalues();	
		}
	}
	#ifndef MY_MAC
		// save eigenvalues (yet unsaved)
		if(!loaded)
			eigenvalues.save(arma::hdf5_name(name + _suffix + ".hdf5", "eigenvalues/"));
	#endif
	return eigenvalues;
}


/// @brief Calculate spectral form factor
/// @tparam Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface<Hamiltonian>::spectral_form_factor(){
	clk::time_point start = std::chrono::system_clock::now();
	
	std::string dir = this->saving_dir + "SpectralFormFactor" + kPSep;
	if(this->beta > 0){
		dir += "beta=" + to_string_prec(this->beta) + kPSep;
	}
	createDirs(dir);
	//------- PREAMBLE
	std::string info = this->set_info();

	const double chi = 0.341345;
	u64 dim = ptr_to_model->get_hilbert_size();

	const double wH = sqrt(this->L) / (chi * dim);
	double tH = 1. / wH;
	double r1 = 0.0, r2 = 0.0;
	int time_end = (int)std::ceil(std::log10(5 * tH));
	time_end = (time_end / std::log10(tH) < 1.5) ? time_end + 1 : time_end;

	arma::vec times = arma::logspace(log10(1.0 / (two_pi * dim)), 1, this->num_of_points);
	arma::vec times_fold = arma::logspace(-2, time_end, this->num_of_points);

	arma::vec sff(this->num_of_points, arma::fill::zeros);
	arma::vec sff_fold(this->num_of_points, arma::fill::zeros);
	double Z = 0.0, Z_fold = 0.0;
	double wH_mean = 0.0;
	double wH_typ  = 0.0;
	
#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
	for(int realis = 0; realis < this->realisations; realis++)
	{
		std::string suffix = "_real=" + std::to_string(realis + this->jobid);
		if(realis > 0)
			ptr_to_model->generate_hamiltonian();
		arma::vec eigenvalues = this->get_eigenvalues(suffix);
		
		if(this->fun == 1) std::cout << "\t\t	--> finished loading eigenvalues for " << info + suffix << " - in time : " << tim_s(start) << "s" << std::endl;
		if(eigenvalues.empty()) continue;
		dim = eigenvalues.size();

		u64 E_av_idx = spectrals::get_mean_energy_index(eigenvalues);
		const u64 num = this->L <= 9? 0.25 * dim : 0.5 * dim;
		const u64 num2 = this->L <= 12? 50 : 500;

		// ------------------------------------- calculate level statistics
			double r1_tmp = 0, r2_tmp = 0, wH_mean_r = 0, wH_typ_r = 0;
			int count = 0;
			for(int i = (E_av_idx - num / 2); i < (E_av_idx + num / 2); i++){
				const double gap1 = eigenvalues(i) - eigenvalues(i - 1);
				const double gap2 = eigenvalues(i + 1) - eigenvalues(i);
				const double min = std::min(gap1, gap2);
				const double max = std::max(gap1, gap2);
				wH_mean_r += gap2;
				wH_typ_r += std::log(gap2);
        		if (abs(gap1) <= 1e-15 || abs(gap2) <= 1e-15){ 
        		    std::cout << "Index: " << i << std::endl;
        		    _assert_(false, "Found degeneracy, while doing r-statistics!\n");
        		}
				r1_tmp += min / max;
				if(i >= (E_av_idx - num2 / 2) && i < (E_av_idx + num2 / 2))
					r2_tmp += min / max;
				count++;
			}
			wH_mean_r /= double(count);
			wH_typ_r = std::exp(wH_typ_r / double(count));
			r1_tmp /= double(count);
			r2_tmp /= double(num2);
			if(this->fun == 1) std::cout << "\t\t	--> finished unfolding for " << info + suffix << " - in time : " << tim_s(start) << "s" << std::endl;
		
		// ------------------------------------- calculate sff
			auto [sff_r_folded, Z_r_folded] = statistics::spectral_form_factor(eigenvalues, times_fold, this->beta, 0.5);
			eigenvalues = statistics::unfolding(eigenvalues);

			auto [sff_r, Z_r] = statistics::spectral_form_factor(eigenvalues, times,this->beta, 0.5);
			#pragma omp critical
			{
				r1 += r1_tmp;
				r2 += r2_tmp;

				sff += sff_r;
				Z += Z_r;
				sff_fold += sff_r_folded;
				Z_fold += Z_r_folded;
				
				wH_mean += wH_mean_r;
				wH_typ  += wH_typ_r;
			}
		if(this->fun == 1) std::cout << "\t\t	--> finished realisation for " << info + suffix << " - in time : " << tim_s(start) << "s" << std::endl;
		
		//--------- SAVE REALISATION TO FILE
		#if !defined(MY_MAC)
			std::string dir_re  = dir + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
			createDirs(dir_re);
			save_to_file(dir_re + info + ".dat", 			times, 		sff_r, 		  Z_r, 		  r1_tmp, r2_tmp, wH_mean_r, wH_typ_r);
			save_to_file(dir_re + "folded" + info + ".dat", times_fold, sff_r_folded, Z_r_folded, r1_tmp, r2_tmp, wH_mean_r, wH_typ_r);
		#else
			std::cout << this->jobid + realis << std::endl;
		#endif
	}

	// --------------------------------------------------------------- AVERAGE CURRENT REALISATIONS
	if(sff.is_empty()) return;
	if(sff.is_zero()) return;
	if(this->jobid > 0) return;

	double norm = this->realisations;
	r1 /= norm;
	r2 /= norm;
	sff = sff / Z;
	sff_fold = sff_fold / Z_fold;
	wH_mean /= norm;
	wH_typ /= norm;

	// ---------- find Thouless time
	double eps = 8e-2;
	auto K_GOE = [](double t){
		return t < 1? 2 * t - t * log(1+2*t) : 2 - t * log( (2*t+1) / (2*t-1) );
	};
	double thouless_time = 0;
	double t_max = 2.5;
	double delta_min = 1e6;
	for(int i = 0; i < sff.size(); i++){
		double t = times(i);
		double delta = abs(log10( sff(i) / K_GOE(t) )) - eps;
		delta *= delta;
		if(delta < delta_min){
			delta_min = delta;
			thouless_time = times(i); 
		}
		if(times(i) >= t_max) break;
	}
	save_to_file(dir + info + ".dat", 			 times, 	 sff, 	   1.0 / wH_mean, thouless_time, 		   r1, r2, dim, 1.0 / wH_typ);
	save_to_file(dir + "folded" + info + ".dat", times_fold, sff_fold, 1.0 / wH_mean, thouless_time / wH_mean, r1, r2, dim, 1.0 / wH_typ);
}

/// @brief Average sff over realisations
/// @tparam Hamiltonian template parameter for current used model
template <class Hamiltonian>
void user_interface<Hamiltonian>::average_sff(){

	std::string dir = this->saving_dir + "SpectralFormFactor" + kPSep;
	std::string info = this->set_info();
	arma::vec times, times_fold; // are always the same
	arma::vec sff(this->num_of_points, arma::fill::zeros);
	arma::vec sff_fold(this->num_of_points, arma::fill::zeros);
	double Z = 0.0;
	double Z_folded = 0.0;
	double r1 = 0.0;
	double r2 = 0.0;
	double wH = 0.;
	double wH_typ = 0.;
	size_t dim = ptr_to_model->get_hilbert_size();
	int counter_realis = 0;
	
	outer_threads = this->thread_number;
	omp_set_num_threads(1);
	std::cout << "THREAD COUNT:\t\t" << outer_threads << "\t\t" << omp_get_num_threads() << std::endl;
#pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
	for(int realis = 0; realis < this->realisations; realis++)
	{
		std::string dir_re  = this->saving_dir + "SpectralFormFactor" + kPSep + "realisation=" + std::to_string(this->jobid + realis) + kPSep;
		std::ifstream file;
		
		auto data = readFromFile(file, dir_re + info + ".dat");
		if(data.empty()) continue;
		if(data[0].size() != sff.size()) {
			std::cout << "Incompatible data dimensions" << std::endl;
			continue;
		}
		file.close();
		#pragma omp critical
		{
			times = data[0];
			sff += data[1];
			Z += data[2](0);
			r1 += data[3](0);
			r2 += data[4](0);
			wH += data[5](0);
			wH_typ += data[6](0);
			counter_realis++;
		}

		data = readFromFile(file, dir_re + "folded" + info + ".dat");
		if(data.empty()) continue;
		if(data[0].size() != sff_fold.size()) {
			std::cout << "Incompatible data dimensions" << std::endl;
			continue;
		}
		file.close();
		#pragma omp critical
		{
			times_fold = data[0];
			sff_fold += data[1];
			Z_folded += data[2](0);
		}
	}

	if(sff.is_empty()) return;
	if(sff.is_zero()) return;

	double norm = counter_realis;
	r1 /= norm;
	r2 /= norm;
	sff = sff / Z;
	sff_fold = sff_fold / Z_folded;
	wH /= norm;
	wH_typ /= norm;

	// ---------- find Thouless time
	double eps = 5e-2;
	auto K_GOE = [](double t){
		return t < 1? 2 * t - t * log(1+2*t) : 2 - t * log( (2*t+1) / (2*t-1) );
	};

	double thouless_time = 0;
	double delta_min = 1e6;
	for(int i = 0; i < sff.size(); i++){
		double delta = abs(log10( sff(i) / K_GOE(times(i)) )) - eps;
		delta *= delta;
		if(delta < delta_min){
			delta_min = delta;
			thouless_time = times(i); 
		}
		if(times(i) >= 2.5) break;
	}

	double thouless_time_fold = 0;
	delta_min = 1e6;
	for(int i = 0; i < sff_fold.size(); i++){
		double delta = abs(log10( sff_fold(i) / K_GOE(times_fold(i)) )) - eps;
		delta *= delta;
		if(delta < delta_min){
			delta_min = delta;
			thouless_time_fold = times_fold(i); 
		}
		if(times_fold(i) >= 2.5 / wH) break;
	}
	save_to_file(dir + info + ".dat", 			 times, 	 sff, 	   wH, thouless_time, 	   r1, r2, dim, wH_typ);
	save_to_file(dir + "folded" + info + ".dat", times_fold, sff_fold, wH, thouless_time_fold, r1, r2, dim, wH_typ);
}