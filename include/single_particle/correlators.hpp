#pragma once
#include "many_body_configurations.hpp"

namespace QHS{
    namespace single_particle{

        namespace tools{
            
            /// @brief Extract the matrix of the input Gaussian state using input single-particle states
            /// @tparam _ty typename of the input orbitals
            /// @param orbitals single particle wavefunctions
            /// @param state many-body eigenstate (configuration -- product state as boost::dynamic_bitset)
            /// @return Slater determinant for given input state
            template <typename _ty>
            inline
            const arma::Mat<_ty> get_matrix_state(const arma::Mat<_ty>& orbitals, const boost::dynamic_bitset<>& state)
            {
                arma::uvec col_idx(long(state.count()));
                int idx = 0;
                int V = orbitals.n_cols;
                for(int q = 0; q < V; q++){
                    double n_q = int(state[q]);
                    if(n_q == 1)
                        col_idx(idx++) = q;
                }
                arma::uvec row_idx = arma::regspace<arma::uvec>(0, V - 1);
                return orbitals.submat(row_idx, col_idx);
            }

        }

        namespace correlators{

            /// @brief Calculate one-body correlation matrix for eigenstate state with single particle states in orbitals. The size of matrix is set by VA
            /// @tparam _ty typename of the input orbitals
            /// @param orbitals single particle wavefunctions
            /// @param state many-body eigenstate (configuration -- product state as boost::dynamic_bitset)
            /// @param VA subsystem size (can be whole system)
            /// @param J_m reference to one-body correlation matrix to add new entries
            /// @param lambda reference to single-site one-body correlation
            /// @param prefactor prefactor for correlation matrix (by default = 1.0)
            template <typename _ty>
            inline
            void one_body(const arma::Mat<_ty>& orbitals, const boost::dynamic_bitset<>& state, int VA, arma::Mat<_ty>& J_m, _ty& lambda, double prefactor = 1.0)
            {
                arma::uvec col_idx(long(state.count()));
                int idx = 0;
                int V = orbitals.n_cols;
                for(int q = 0; q < V; q++){
                    double n_q = int(state[q]);
                    if(VA < V)
                        lambda += prefactor * n_q * std::abs(orbitals(q, VA) * std::conj(orbitals(q, VA)));
                    if(n_q == 1)
                        col_idx(idx++) = q;
                }
            
                if(VA > 0){
                    arma::uvec row_idx = arma::regspace<arma::uvec>(V - VA, V - 1);
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
            /// @tparam _ty typename of the input orbitals
            /// @param orbitals single particle wavefunctions
            /// @param state many-body eigenstate (configuration -- product state as boost::dynamic_bitset)
            /// @param VA subsystem size (can be whole system)
            template <typename _ty>
            inline
            std::pair<arma::Mat<_ty>, cpx> one_body(const arma::Mat<_ty>& orbitals, const boost::dynamic_bitset<>& state, int VA)
            {
                arma::Mat<_ty> J_m(VA, VA, arma::fill::zeros);
                _ty lambda = 0.0;
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
                return std::make_pair(J_m, lambda);    
            }
        }

    };
}