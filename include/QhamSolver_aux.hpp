#pragma once

namespace QHS{

    template <class Hamiltonian>
    class QHamSolver{
        static_check((std::is_base_of_v<_hamiltonian, Hamiltonian>), 
                        "\n" BAD_INHERITANCE "\n\t base class is: hamiltonian_base<element_type, hilbert_space>");
        
        protected:
            typedef typename Hamiltonian::element_type _ty;

            Hamiltonian H;
            arma::Mat<_ty> eigenvectors;
            arma::vec eigenvalues;
            u64 dim;
        public:
            u64 E_av_idx;
            
            //<! ----------------------------------------------- CONSTRUCTORS / DESTRUCTORS
            ~QHamSolver() { DESTRUCTOR_CALL; }

            template <typename... _param_types>
            QHamSolver(_param_types... args);

            //<! ----------------------------------------------- GETTERS
            auto get_hilbert_size()	                 const { return this->dim; }					        // get the Hilbert space size
            auto get_mapping()		                 const { return this->H.get_mapping(); }		        // constant reference to the mapping
            auto& get_eigenvectors()                 const { return this->eigenvectors; }			    // get the const reference to the eigenvectors
            auto get_eigenState(u64 idx)             const { return this->eigenvectors.col(idx); }	    // get the eigenvector at index idx
            
            auto get_eigenStateCoeff(u64 idx, u64 k) const { return this->eigenvectors.col(idx)(k); }	    // get the coefficient of eigenstate idx at position k
            
            auto get_eigenValue(u64 idx)             const { return this->eigenvalues(idx); }            // get eigenenergy at position idx
            auto get_eigenvalues()	                 const { return this->eigenvalues; }			        // get the const reference to eigenvalues
            auto get_hamiltonian()	                 const { return this->H.get_hamiltonian(); }	        // get the const reference to a Hamiltonian
            auto get_dense_hamiltonian()             const { return this->H.get_dense_hamiltonian(); }	// get the const reference to a Hamiltonian

            auto& get_model_ref()                    const { return this->H; }

            //<! ----------------------------------------------- ROUTINES
            void generate_hamiltonian();
            void diagonalization(bool get_eigenvectors = true, const char* method = "dc");
            double diag_sparse(int Nev, int s, double tol = 1e-14, int seed = std::random_device{}());
    };


    //<! ---------------------------------------------------------------------------------------------------------------------------------------
    //<! ------------------------------------------------------------------------------------------------------------------------ IMPLEMENTATION


    //<! ------------------------------------------------------------------------------ CONSTRUCTORS

    /// @brief 
    /// @tparam Hamiltonian
    /// @tparam _param_types 
    /// @param ...args 
    template <class Hamiltonian>
    template <typename... _param_types>
    QHamSolver<Hamiltonian>::QHamSolver(_param_types... args)
    {
        CONSTRUCTOR_CALL;
        //<! Initlialize model
        H = Hamiltonian(args...);
        this->dim = H.get_hilbert_space_size();

        //<! what else?
        _extra_debug(
            std::cout << FUN_SIGNATURE << "::\n\t QHamSolver initialized with: "
                << var_name_value(this->dim, 0) << "\n MODEL:\n" << H << std::endl;
        )
    }


    //<! ------------------------------------------------------------------------------ ROUTINES

    /// @brief 
    /// @tparam Hamiltonian 
    template <class Hamiltonian>
    void QHamSolver<Hamiltonian>::generate_hamiltonian()
        { this->H.create_hamiltonian(); }

    /// @brief Method do diagonalize the Hamiltonian
    /// @tparam Hamiltonian template typename for the Hamiltonian class
    /// @param get_eigenvectors choose to get or not eigenstates
    /// @param method choose methods ("dc" or "std") of diagonalization
    template <class Hamiltonian>
    void QHamSolver<Hamiltonian>::diagonalization(bool get_eigenvectors, const char* method) 
    {
        //out << real(H) << endl;
        arma::Mat<_ty> H_temp;
        try {
            H_temp = this->H.get_dense_hamiltonian();
            if (get_eigenvectors){
                if(this->dim > 2e5)
                    method = "std"; // Change method due to smaller memory consumption (higher CPU time though -- benchmark)
                arma::eig_sym(this->eigenvalues, this->eigenvectors, H_temp, method);
            }
            else
                arma::eig_sym(this->eigenvalues, H_temp);
            #ifndef NODEBUG
                std::cout << "\t HAMILTONIAN TYPE: " + type_name<_ty>() + "\n\tsparse - dim(H) = " + matrix_size(this->H.n_nonzero * sizeof(this->H(0, 0)))
                    + "\n\tdense - dim(H) = " + matrix_size(H_temp.n_cols * H_temp.n_rows * sizeof(H_temp(0, 0)))
                    + "\n\tspectrum size: " + std::to_string(this->dim) << std::endl << std::endl;
            #endif
        }
        catch (...) {
            handle_exception(std::current_exception(), 
                "sparse - dim(H) = " + matrix_size(this->H.n_nonzero * sizeof(this->H(0, 0)))
                + "\ndense - dim(H) = " + matrix_size(H_temp.n_cols * H_temp.n_rows * sizeof(H_temp(0, 0)))
                + "\n spectrum size: " + std::to_string(this->dim)
            );
        }
        double E_av = arma::mean(this->eigenvalues);
        auto i = min_element(begin(this->eigenvalues), end(this->eigenvalues), [=](double x, double y) {
            return abs(x - E_av) < abs(y - E_av);
            });
        this->E_av_idx = i - begin(this->eigenvalues);
        _extra_debug(
            printSeparated(std::cout, "\t", 16, true, "guessed index", "mean energy", "energies close to this value (-1,0,+1) around found index");
            printSeparated(std::cout, "\t", 16, true, this->E_av_idx, E_av, this->eigenvalues(this->E_av_idx - 1), this->eigenvalues(this->E_av_idx),  this->eigenvalues(this->E_av_idx + 1));
        )

        //<! Force release of memory for dense matrix
        H_temp.reset();
    }

    /// @brief Method to diagonalize large sparse matrices: get states in the centrum of the spectrum
    /// @tparam Hamiltonian template typename for the Hamiltonian class
    /// @param Nev number of requested eigenstates
    /// @param s block size
    /// @param tol tolerance for algorithm
    /// @param seed input seed got random computation
    template <class Hamiltonian>
    double QHamSolver<Hamiltonian>::diag_sparse(int Nev, int s, double tol, int seed)
    {
        auto Hamil = this->H.get_hamiltonian();
        auto polfed = polfed::POLFED<QHamSolver::_ty>(Hamil, Nev, s, -1, tol, 0.2, seed, true);
        auto [E, V] = polfed.eig();
        auto indices = arma::sort_index(E);
        this->eigenvalues = E( indices );
        this->eigenvectors = V.cols( indices );
        // #ifdef EXTRA_DEBUG
            std::cout << "-------------------------------------- TEST POLFED SPECTRUM --------------------------------------" << std::endl;
            double error = -100;
            for(int n = 0; n < this->eigenvalues.size(); n++)
            {
                auto value = arma::cdot(this->eigenvectors.col(n), Hamil * this->eigenvectors.col(n));
                double error_n = std::abs(value - this->eigenvalues(n));
                if( error_n > error )   error = error_n;
                printSeparated(std::cout, "\t", 20, true, n, this->eigenvalues(n), value, error_n);
            }
            std::cout << "-------------------------------------- TEST POLFED SPECTRUM --------------------------------------" << std::endl;
        // #endif
        return error;
    }
}