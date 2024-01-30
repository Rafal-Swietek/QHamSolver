#pragma once

namespace polfed{

    /// @brief Calculate eigenvalues and eigenstates using Block-Lanczos of transformed hamiltonian
    /// @tparam _ty type of input Hamiltonian (enforces type on Krylov basis)
	/// @tparam converge_type enum type for convergence criterion (energies or states)
    /// @return true eigenvalues and eigenvectors as std::pair
    template <typename _ty, converge converge_type>
inline
    std::pair<arma::vec, arma::Mat<_ty>>
    POLFED<_ty, converge_type>::eig()
    {
        //<! START BLOCK-LANCZOS ITERATION
        _debug_start( clk::time_point start = std::chrono::system_clock::now(); )
        auto lancz_block = lanczos::BlockLanczos<_ty, converge::states>(this->PH_multiply, this->N, 
                                this->num_of_eigval, this->bundle_size, 3 * this->num_of_eigval, this->tolerance, this->seed, this->use_krylov);
        lancz_block.diagonalization();
        this->conv_steps = lancz_block.get_lanczossteps();
        _debug_end( std::cout << "\t\tFinished Block-Lanczos Iteration in \t" <<  lancz_block.get_lanczossteps() << "\tin " << tim_s(start) << " seconds" << std::endl; )
        
        //<! GET EIGNESTATES
        _debug_start( start = std::chrono::system_clock::now(); )
        auto V = lancz_block.get_eigenstates();
        _debug_end( std::cout << "\t\tTranformed desired states to Hilbert basis in " << tim_s(start) << " seconds" << std::endl; )
        
        //<! GET EIGENVALUES
        _debug_start( start = std::chrono::system_clock::now(); )
        arma::vec E(this->num_of_eigval);
    #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
        for(long k = 0; k < this->num_of_eigval; k++)
            E(k) = arma::cdot(V.col(k), this->H * V.col(k));
        _debug_end( std::cout << "\t\tCalculated eigenenergies from original Hamiltonian with <k|H|k> in " << tim_s(start) << " seconds" << std::endl; )
        
    //     //<! SORT EIGENPAIRS
    //     _debug_start( start = std::chrono::system_clock::now(); )
    //     arma::vec E(this->num_of_eigval);
    // #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
    //     for(long k = 0; k < this->num_of_eigval; k++)
    //         E(k) = arma::cdot(V.col(k), this->H * V.col(k));
    //     _debug_end( std::cout << "\t\tSorted eigenstates according to eigenvalues in " << tim_s(start) << " seconds" << std::endl; )
        return std::make_pair(E, V);
    }

	//! ------------------------------------------------------------------------------------------------------------ PASS REFERENCE TO CONTAINERS
    /// @brief Calculate eigenvalues and eigenstates using Block-Lanczos of transformed hamiltonian
    /// @tparam _ty type of input Hamiltonian (enforces type on Krylov basis)
	/// @tparam converge_type enum type for convergence criterion (energies or states)
    /// @return true eigenvalues and eigenvectors as std::pair
    template <typename _ty, converge converge_type>
    inline
    void
    POLFED<_ty, converge_type>::eig(arma::vec& E, arma::Mat<_ty>& V)
    {
        //<! START BLOCK-LANCZOS ITERATION
        _debug_start( clk::time_point start = std::chrono::system_clock::now(); )
        auto lancz_block = lanczos::BlockLanczos<_ty, converge::states>(this->PH_multiply, this->N, 
                                this->num_of_eigval, this->bundle_size, 3 * this->num_of_eigval, this->tolerance, this->seed, this->use_krylov);
        lancz_block.diagonalization();
        this->conv_steps = lancz_block.get_lanczossteps();
        _debug_end( std::cout << "\t\tFinished Block-Lanczos Iteration in \t" <<  lancz_block.get_lanczossteps() << "\tin " << tim_s(start) << " seconds" << std::endl; )

        //<! GET EIGNESTATES
        _debug_start( start = std::chrono::system_clock::now(); )
        V = lancz_block.get_eigenstates();
        _debug_end( std::cout << "\t\tTranformed desired states to Hilbert basis in " << tim_s(start) << " seconds" << std::endl; )
        
        //<! GET EIGENVALUES
        _debug_start( start = std::chrono::system_clock::now(); )
        E.resize(this->N);
    #pragma omp parallel for num_threads(outer_threads) schedule(dynamic)
        for(long k = 0; k < this->N; k++)
            E(k) = arma::cdot(V.col(k), this->H * V.col(k));
        _debug_end( std::cout << "\t\tCalculated eigenenergies from original Hamiltonian with <k|H|k> in " << tim_s(start) << " seconds" << std::endl; )
    }
    
    /// @brief Calculate only eigenstates using Block-Lanczos of transformed hamiltonian
    /// @tparam _ty type of input Hamiltonian (enforces type on Krylov basis)
	/// @tparam converge_type enum type for convergence criterion (energies or states)
    /// @param V reference to eigenvectors to be overwritten with true eigenstates from polfed
    template <typename _ty, converge converge_type>
    inline
    void
    POLFED<_ty, converge_type>::eig(arma::Mat<_ty>& V)
    {
        //<! START BLOCK-LANCZOS ITERATION
        _debug_start( clk::time_point start = std::chrono::system_clock::now(); )
        auto lancz_block = lanczos::BlockLanczos<_ty, converge::states>(this->PH_multiply, this->N, 
                                this->num_of_eigval, this->bundle_size, 3 * this->num_of_eigval, this->tolerance, this->seed, this->use_krylov);
        lancz_block.diagonalization();
        this->conv_steps = lancz_block.get_lanczossteps();
        _debug_end( std::cout << "\t\tFinished Block-Lanczos Iteration in \t" <<  lancz_block.get_lanczossteps() << "\tin " << tim_s(start) << " seconds" << std::endl; )

        //<! GET EIGNESTATES
        _debug_start( start = std::chrono::system_clock::now(); )
        V = lancz_block.get_eigenstates();
        _debug_end( std::cout << "\t\tTranformed desired states to Hilbert basis in " << tim_s(start) << " seconds" << std::endl; )
    }

    /// @brief Calculate eigenvalues using Block-Lanczos of transformed hamiltonian
    /// @tparam _ty type of input Hamiltonian (enforces type on Krylov basis)
	/// @tparam converge_type enum type for convergence criterion (energies or states)
    /// @param E reference to energies to be overwritten with true eigenenergies from polfed
    template <typename _ty, converge converge_type>
    inline
    void
    POLFED<_ty, converge_type>::eig(arma::vec& E)
    {
        //<! START BLOCK-LANCZOS ITERATION
        _debug_start( clk::time_point start = std::chrono::system_clock::now(); )
        auto lancz_block = lanczos::BlockLanczos<_ty, converge::states>(this->PH_multiply, this->N, 
                                this->num_of_eigval, this->bundle_size, 3 * this->num_of_eigval, this->tolerance, this->seed, this->use_krylov);
        lancz_block.diagonalization();
        this->conv_steps = lancz_block.get_lanczossteps();
        _debug_end( std::cout << "\t\tFinished Block-Lanczos Iteration in \t" <<  lancz_block.get_lanczossteps() << "\tin " << tim_s(start) << " seconds" << std::endl; )

        //<! GET EIGENVALUES
        _debug_start( start = std::chrono::system_clock::now(); )
        E = lancz_block.get_eigenvalues();
        
        _assert_(E.size() == lancz_block.get_lanczossteps(), "Not implemented inverse transformation to get eigenvalues without states.");
        
        //TODO: find method to inverse polynomial
        _debug_end( std::cout << "\t\tCalculated eigenenergies from original Hamiltonian with <k|H|k> in " << tim_s(start) << " seconds" << std::endl; )
    }
}