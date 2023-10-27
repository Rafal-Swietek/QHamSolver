#pragma once
#include "many_body_configurations.hpp"

namespace single_particle{

    namespace correlators{

        /// @brief Calculate one-body correlation matrix for eigenstate state with single particle states in orbitals. The size of matrix is set by VA
        /// @param orbitals single particle wavefunctions
        /// @param state many-body eigenstate (configuration -- product state as boost::dynamic_bitset)
        /// @param VA subsystem size (can be whole system)
        /// @param J_m reference to one-body correlation matrix to add new entries
        /// @param lambda reference to single-site one-body correlation
        /// @param prefactor prefactor for correlation matrix (by default = 1.0)
        inline
        void one_body(const arma::cx_mat& orbitals, const boost::dynamic_bitset<>& state, int VA, arma::cx_mat& J_m, cpx& lambda, double prefactor = 1.0)
        {
            arma::uvec col_idx(long(state.count()));
            int idx = 0;
            for(int q = 0; q < orbitals.n_cols; q++){
                double n_q = int(state[q]);
                lambda += prefactor * n_q * std::abs(orbitals(q, VA) * std::conj(orbitals(q, VA)));
                if(n_q == 1)
                    col_idx(idx++) = q;
            }
            if(VA > 0){
                arma::uvec row_idx = arma::regspace<arma::uvec>(0, VA-1);
                auto W = orbitals.submat(row_idx, col_idx);
                J_m += prefactor * W * W.t();
            }
            // for(int q = 0; q < orbitals.n_cols; q++){
            //     double n_q = int(state[q]);
            //     lambda += prefactor * n_q * std::abs(orbitals(q, VA) * std::conj(orbitals(q, VA)));
            //     if(VA > 0){
            //         auto orbital = orbitals.col(q).rows(0, VA - 1);
            //         J_m += prefactor * n_q * orbital * orbital.t();
            //     }
            // }
        }

        /// @brief Calculate one-body correlation matrix for eigenstate state with single particle states in orbitals. The size of matrix is set by VA
        /// @param orbitals single particle wavefunctions
        /// @param state many-body eigenstate (configuration -- product state as boost::dynamic_bitset)
        /// @param VA subsystem size (can be whole system)
        inline
        std::pair<arma::cx_mat, cpx> one_body(const arma::cx_mat& orbitals, const boost::dynamic_bitset<>& state, int VA)
        {
            arma::cx_mat J_m(VA, VA, arma::fill::zeros);
			cpx lambda = 0.0;
            arma::uvec col_idx(long(state.count()));
            int idx = 0;
            for(int q = 0; q < orbitals.n_cols; q++){
                double n_q = int(state[q]);
                lambda += n_q * std::abs(orbitals(q, VA) * std::conj(orbitals(q, VA)));
                if(n_q == 1)
                    col_idx(idx++) = q;
            }
            if(VA > 0){
                arma::uvec row_idx = arma::regspace<arma::uvec>(0, VA-1);
                auto W = orbitals.submat(row_idx, col_idx);
                J_m += W * W.t();
            }
            std::make_pair(J_m, lambda);    
        }
    }


    namespace entanglement{

        /// @brief Calcute term in  sum for single lambda (also used when size of one-body correlation matrix is 1)
        /// @param lambda input eigenvalue of one-body correlation matrix
        /// @return entanglement entropy given by the eigenvalue of correlation matrix
        inline
        double vonNeumann_helper(double lambda){
            if(std::abs(lambda) > 1.0 - 1e-12)
                return 0;
            else{
                double lam1 = (1 + lambda) / 2.;
                double lam2 = (1 - lambda) / 2.;
                return -lam1 * std::log(lam1) - lam2 * std::log(lam2); 
            }
        }


        /// @brief Calculate von Neumann entanglement entropy from eigenvalues of one-body correlation matrix
        /// @param corr_coeff eigenvalues of one-body correlation matrix
        /// @return entanglement entropy
        inline
        double vonNeumann(arma::vec corr_coeff){
            double S = 0;
            for(auto& lam : corr_coeff)
                S += vonNeumann_helper(lam);
            return S;
        }

    }
};