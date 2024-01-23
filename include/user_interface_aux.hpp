#pragma once


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

	if (geq_0 && value < _ty(0)){					  // if the variable shall be bigger equal 0
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
		"Lanczos parameters:"
		"-l_maxiter Maximal number of lanczos iterations for either regular or Block Lanczos"
		"-l_steps Number of lanczos iterations fo either regular or Block Lanczos"
		"-l_realis NUmber of realizations for FTLM methods"
		"-l_bundle Number of initial states for Block Lanczos method"
		"-mem_over_perf Use Hamiltonian-vector product on-the-fly (no Hamiltonian in memory)"
		"-reortho Use full reorthogonalization in Lanczos methods"
		""
		"-tol tolerance for iterative methods (i.e. Lanczos)"
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
	this->dir_prefix = "";
	
	this->L = 4;
	this->Ls = 1;
	this->Ln = 1;

	this->site = 0;
	this->op = 0;
	this->fun = INT_MAX;
	this->mu = 5;
	this->tol = 1e-14;
	
	this->q_ipr = 1.0;
	this->beta = 0.0;

	this->boundary_conditions = 0;
	this->thread_number = 1;

	this->ch = false;
	this->num_of_points = 5000;
	this->seed = std::random_device{}();


	this->l_maxiter = 1000;
	this->l_steps = 100;
	this->l_realis = 1;
	this->l_bundle = 5;
	this->mem_ver_perf = false;
	this->reorthogonalize = true;

	this->dt = 0.01;
	this->tend = 1000;
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
		  << "q_ipr = " << this->q_ipr << std::endl
		  << "\u03B2 = " << this->beta << std::endl
		  << "seed = " << this->seed << std::endl
		  << "tol = " << this->tol << std::endl
		  << "l_maxiter = " << this->l_maxiter << std::endl
		  << "l_steps = " << this->l_steps << std::endl
		  << "l_realis = " << this->l_realis << std::endl
		  << "l_bundle = " << this->l_bundle << std::endl
		  << "mem_over_perf = " << this->mem_ver_perf << std::endl
		  << "reortho = " << this->reorthogonalize << std::endl
		  << "dt = " << this->dt << std::endl
		  << "tend = " << this->tend << std::endl;

	#ifdef PRINT_HELP
		std::cout << "---------------------------------------------------------------------------------\n\n";
		this->print_help();
		std::cout << "---------------------------------------------------------------------------------\n\n";
	#endif
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
	if(this->site < 0)
		this->site = this->L / 2;

	// choose operator
	choosen_option = "-op";
	this->set_option(this->op, argv, choosen_option, true);
	
	// q for participation ration calculation
	choosen_option = "-q_ipr";
	this->set_option(this->q_ipr, argv, choosen_option);

	// boolean value
	choosen_option = "-ch";
	this->set_option(this->ch, argv, choosen_option, true);
	
	// choose temperature
	choosen_option = "-beta";
	this->set_option(this->beta, argv, choosen_option);

	// choose function
	choosen_option = "-fun";
	this->set_option(this->fun, argv, choosen_option, true);

	// choose tolerance
	choosen_option = "-tol";
	this->set_option(this->tol, argv, choosen_option);

	// Lanczos parameters
	choosen_option = "-l_maxiter";
	this->set_option(this->l_maxiter, argv, choosen_option, false);
	choosen_option = "-l_steps";
	this->set_option(this->l_steps, argv, choosen_option, true);
	choosen_option = "-l_realis";
	this->set_option(this->l_realis, argv, choosen_option, true);
	choosen_option = "-l_bundle";
	this->set_option(this->l_bundle, argv, choosen_option, true);
	choosen_option = "-mem_over_perf";
	this->set_option(this->mem_ver_perf, argv, choosen_option, true);
	choosen_option = "-reortho";
	this->set_option(this->reorthogonalize, argv, choosen_option, true);

	// dynamics
	choosen_option = "-dt";
	this->set_option(this->dt, argv, choosen_option, false);
	choosen_option = "-tend";
	this->set_option(this->tend, argv, choosen_option, false);

	// buckets
	choosen_option = "-mu";
	this->set_option(this->mu, argv, choosen_option, true);
	choosen_option = "-num_of_points";
	this->set_option(this->num_of_points, argv, choosen_option, true);

	// random seed
	choosen_option = "-seed";
	this->set_option(this->seed, argv, choosen_option, true);

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

	// random seed
	choosen_option = "-dir";
	if (std::string option = this->getCmdOption(argv, choosen_option); option != "")
		this->dir_prefix = option;

	std::cout << " - - - - - - MAKING ISING INTERFACE AND USING OUTER THREADS : "
			  << outer_threads << " - - - - - - " << endl; // setting the number of threads to be used with omp

	std::cout << " - - - - - - MAKING ISING INTERFACE AND USING INNER THREADS : "
			  << thread_number << " - - - - - - " << endl; // setting the number of threads to be used with omp

}

// -------------------------------------------------------------------------------------------------------------------------------

// -------------------------------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------------------- MAIN ROUTINES

/// @brief Get eigenvalues from .hdf5 file if present else diagonalize model
/// @tparam Hamiltonian template parameter for current used model
/// @param _suffix string suffix used distinct different realisations
/// @return returns arma::vec of eigenvalues
template <class Hamiltonian>
arma::vec user_interface<Hamiltonian>::get_eigenvalues(std::string prefix, bool diag_if_empty) 
{
	arma::vec eigenvalues;
	std::string dir = this->saving_dir + "DIAGONALIZATION" + kPSep;
	createDirs(dir);
	std::string name = dir + prefix + this->set_info({});
	bool loaded;
	#pragma omp critical
	{
		loaded = eigenvalues.load(arma::hdf5_name(name + ".hdf5", "eigenvalues/dataset"));
		if(!loaded)
			loaded = eigenvalues.load(arma::hdf5_name(name + ".hdf5", "eigenvalues/"));
		name = this->saving_dir + "Entropy/Eigenstate" + kPSep + prefix + this->set_info({});
		if(!loaded)
			loaded = eigenvalues.load(arma::hdf5_name(name + ".hdf5", "energies"));
		if(!loaded){
			std::cout << "Not found:\t" << name << std::endl;
			if(diag_if_empty){
				ptr_to_model->diagonalization(false);
				eigenvalues = ptr_to_model->get_eigenvalues();	
			}
		}
	}
	// save eigenvalues (yet unsaved)
	// if(!loaded && diag_if_empty)
	// 	eigenvalues.save(arma::hdf5_name(name + _suffix + ".hdf5", "eigenvalues"));

	return eigenvalues;
}


