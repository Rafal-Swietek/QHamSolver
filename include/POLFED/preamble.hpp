#pragma once

namespace polfed{

    /// @brief Find exterior eigenenergies for class Hamiltonina to rescale
	/// @tparam _ty type of input Hamiltonian (enforces type on output state)
	/// @tparam converge_type enum type for convergence criterion (energies or states)
	template <typename _ty, converge converge_type>
	inline
    std::pair<double, double>
    POLFED<_ty, converge_type>::get_energy_bounds()
    {
        auto lancz = lanczos::Lanczos<_ty, converge::energies>(this->H, 1, 1000, 1e-15, this->seed, this->use_krylov);        
        lancz.diagonalization();
        auto E = lancz.get_eigenvalues();
        double Emin = E(0);
        double Emax = E(E.size() - 1);

        return std::make_pair(Emin, Emax);
	}

    /// @brief Find exterior eigenenergies for class Hamiltonina to rescale
    /// @param steps reference to set number of lanczos iterations used
	/// @tparam _ty type of input Hamiltonian (enforces type on output state)
	/// @tparam converge_type enum type for convergence criterion (energies or states)
	template <typename _ty, converge converge_type>
	inline
    std::pair<double, double>
    POLFED<_ty, converge_type>::get_energy_bounds(int& steps)
    {
        auto lancz = lanczos::Lanczos<_ty, converge::energies>(this->H, 1, 1000, 1e-15, this->seed, this->use_krylov);        
        lancz.diagonalization();
        auto E = lancz.get_eigenvalues();
        double Emin = E(0);
        double Emax = E(E.size() - 1);
        steps = lancz.get_lanczossteps();
        return std::make_pair(Emin, Emax);
	}

    /// @brief Find order of polynomial
    /// @tparam _ty type of input Hamiltonian (enforces type on Krylov basis)
	/// @tparam converge_type enum type for convergence criterion (energies or states)
    template <typename _ty, converge converge_type>
	inline 
    void POLFED<_ty, converge_type>::set_poly_order(double Emin, double Emax) 
    {
        //<! set first and second moment
        _ty var     = arma::trace(this->H * this->H) / double(this->N) - this->sigma * this->sigma;
        _ty DOS     = this->N * (Emax - Emin) / 2.0 / ( std::sqrt(2.0 * two_pi * var) );
        _ty delta   = this->num_of_eigval / (2.0 * DOS);
        
        this->K = 15;
        //<! loop until only desired states left in energy window
        while(true){
            this->coeff = arma::Col<_ty>(this->K + 1, arma::fill::zeros);
            this->coeff(0) = 1.0;
            for(int n = 1; n < K+1; n++)
                this->coeff(n) = 2.0 * std::cos(n * std::acos(this->sigma));
            _ty poly = clenshaw::chebyshev(this->K, this->coeff, this->sigma + delta) / clenshaw::chebyshev(this->K, this->coeff, this->sigma);
            if( std::abs(poly) < this->cutoff){
                this->K -= 2;
                break;
            } else
                this->K += 2;
        }
    }

    /// @brief Transform Hamiltonian to polynomial series
	/// @tparam _ty type of input Hamiltonian (enforces type on Krylov basis)
	/// @tparam converge_type enum type for convergence criterion (energies or states)
	template <typename _ty, converge converge_type>
	inline 
    void POLFED<_ty, converge_type>::transform_matrix() {

        //<! get energy boundaries (to rescale E \in [-1,1])
        _debug_start( clk::time_point start = std::chrono::system_clock::now(); )
        this->P_H = this->H;
        int steps = 0;
        auto [Emin, Emax] = this->get_energy_bounds(steps);
        this->P_H = (2.0 * this->P_H - (Emax + Emin) * arma::eye<arma::SpMat<_ty>>(this->N, this->N)) / (Emax - Emin);
        _debug_end( std::cout << "\t\tFound energy boundaries:\t" <<  Emin << " < E < " << Emax << "\tusing " << steps << " iterations\tin " << tim_s(start) << " seconds" << std::endl; )
    
        // set polynomial order
        _debug_start( start = std::chrono::system_clock::now(); )
        this->sigma = arma::trace(this->P_H) / double(this->N);
        this->set_poly_order(Emin, Emax);
        _debug_end( std::cout << "\t\tSet polynomial order:\tK=" <<  this->K << "\tin " << tim_s(start) << " seconds" << std::endl; )

        // set coefficients
        _debug_start( start = std::chrono::system_clock::now(); )
        this->coeff = arma::Col<_ty>(this->K + 1, arma::fill::zeros);
        this->coeff(0) = 1.0;
        for(int n = 1; n < K+1; n++)
            this->coeff(n) = 2.0 * std::cos(n * std::acos(this->sigma));

        _debug_end( std::cout << "\t\tSet coefficients for polynomial:\t" 
                                #ifdef EXTRA_DEBUG
                                    <<  this->coeff.t() 
                                #endif
                                << "\tin " << tim_s(start) << " seconds" << std::endl; )


        // final transform
        _debug_start( start = std::chrono::system_clock::now(); )
        double D = clenshaw::chebyshev(this->K, this->coeff, this->sigma);
        auto kernel = [this, D](const arma::Mat<_ty>& bundle) -> arma::Mat<_ty>
            { return -1.0 * clenshaw::chebyshev(this->K, this->coeff, this->P_H, bundle) / D; };
        this->PH_multiply = hamiltonian_func_ptr<arma::Mat<_ty>>(kernel);
        // double D = clenshaw::chebyshev(K, this->coeff, sigma);
        // this->P_H = clenshaw::chebyshev(K, this->coeff, this->P_H) / D;
        _debug_end( std::cout << "\t\tSet tranform (on-the-fly) of Hamiltonian to polynomial series:\tin " << tim_s(start) << " seconds" << std::endl; )
        
        // auto Efull = arma::eig_sym(arma::Mat<_ty>(this->H));
        // auto Escaled = arma::eig_sym(arma::Mat<_ty>(this->P_H));
        // auto [E, V] = this->eig();

        // std::string name = "./_D=" + std::to_string(this->N);
        // Efull.save(arma::hdf5_name(name + ".hdf5", "full"));
        // Escaled.save(arma::hdf5_name(name + ".hdf5", "scaled", arma::hdf5_opts::append));
        // E.save(arma::hdf5_name(name + ".hdf5", "polfed", arma::hdf5_opts::append));
        // this->coeff.save(arma::hdf5_name(name + ".hdf5", "coeff", arma::hdf5_opts::append));
        // arma::vec params(3);
        // params(0) = this->num_of_eigval;
        // params(1) = this->K;
        // params(2) = this->sigma;
        // params.save(arma::hdf5_name(name + ".hdf5", "params", arma::hdf5_opts::append));
        
	}
}