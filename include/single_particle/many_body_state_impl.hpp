#pragma once

namespace QHS{

    namespace single_particle{

        namespace slater{

            /// @brief Initialize the ManyBodyState class to convert from gaussian to many-body states
            /// @tparam _ty type of input orbitals
            /// @tparam use_U1_decomp Create ManyBody state in U(1) subspace
            template <typename _ty, bool use_U1_decomp>
            inline 
            void ManyBodyState<_ty, use_U1_decomp>::initialize() 
            {
                CONSTRUCTOR_CALL;
                this->check_spin = QOps::__builtins::get_digit(this->volume);
            }
            
            /// @brief Initialize the ManyBodyState class to convert from gaussian to many-body states
            /// @tparam _ty type of input orbitals
            /// @tparam use_U1_decomp Create ManyBody state in U(1) subspace
            /// @param gaussian_state Input gaussian state (bitset) as quasiparticle product state
            template <typename _ty, bool use_U1_decomp>
            inline
            arma::uvec ManyBodyState<_ty, use_U1_decomp>::set_indices(const boost::dynamic_bitset<>& gaussian_state, int N)
            {
                arma::uvec set_idx(N, arma::fill::zeros);
                int count = 0;
                for(int id = 0; id < gaussian_state.size(); id++){
                    if( (bool)gaussian_state[id] ){
                        set_idx(count) = id;
                        count++;
                    }
                }
                
                return set_idx;
            }

            /// @brief Initialize the ManyBodyState class to convert from gaussian to many-body states
            /// @tparam _ty type of input orbitals
            /// @tparam use_U1_decomp Create ManyBody state in U(1) subspace
            /// @param gaussian_state Input gaussian state (bitset) as quasiparticle product state
            template <typename _ty, bool use_U1_decomp>
            inline
            arma::uvec ManyBodyState<_ty, use_U1_decomp>::_set_indices(const boost::dynamic_bitset<>& gaussian_state)
            {
                return set_indices(gaussian_state, this->num_particles);
            }

            /// @brief Initialize the ManyBodyState class to convert from gaussian to many-body states
            /// @tparam _ty type of input orbitals
            /// @tparam use_U1_decomp Create ManyBody state in U(1) subspace
            /// @param state_idx Input gaussian state (bitset) as quasiparticle product state
            template <typename _ty, bool use_U1_decomp>
            inline
            arma::uvec ManyBodyState<_ty, use_U1_decomp>::_set_ell_indices(u64 state_idx)
            {
                arma::uvec set_ell(this->num_particles, arma::fill::zeros);
                int count = 0;
                for(int id = 0; id < this->volume; id++){
                    if( this->check_spin(state_idx, id) ){
                        set_ell(count) = id;
                        count++;
                    }
                }

                return set_ell;
            }

            /// @brief Calculate Slater determinant of gaussian state (given by set indices set_q) with product state (given by set indices set_l)
            /// @param set_l set indices of current product state
            /// @param set_q sert indices of gaussian state of interest
            /// @tparam _ty type of input orbitals
            /// @tparam use_U1_decomp Create ManyBody state in U(1) subspace
            /// @return slater determinant for given indices
            template <typename _ty, bool use_U1_decomp>
            inline
            _ty ManyBodyState<_ty, use_U1_decomp>::determinant(const arma::uvec& set_l, const arma::uvec& set_q)
            {
                auto W = this->_orbitals.submat(set_l, set_q);
                
                arma::cx_vec eigs = arma::eig_gen(W);
                arma::cx_double d2 = arma::prod(eigs);
                
                // arma::cx_double d1 = arma::det(W);
                // if (std::abs(std::abs(d1) - std::abs(d2)) > 1e-10) {
                //     std::cout << "PROBLEM\n";
                //     std::cout << "det      = " << d1 << "\n";
                //     std::cout << "prod eig = " << d2 << "\n";
                //     std::cout << "ratio    = " << d1/d2 << "\n";
                // }
                return d2;
                // return arma::det(W);
                // auto eigs = arma::eig_gen(W);
                // return arma::prod(eigs);
            }
            // template <>
            // inline
            // double ManyBodyState<double, true>::determinant(const arma::uvec& set_l, const arma::uvec& set_q)
            // {
            //     auto W = this->_orbitals.submat(set_l, set_q);

            //     auto eigs = arma::eig_gen(W);

            //     _ty d_eig = arma::prod(eigs);
            //     _ty d_det = arma::det(W);

            //     if (std::abs(d_eig - d_det) > 1e-10) {
            //         std::cout << "W =\n" << W << std::endl;
            //         std::cout << "det      = " << d_det << std::endl;
            //         std::cout << "prod eig = " << d_eig << std::endl;
            //     }
            //     return d_eig;
            //     // return arma::det(W);
            //     // auto eigs = arma::eig_gen(W);
            //     // return std::real(arma::prod(eigs));
            // }
            // template <>
            // inline
            // double ManyBodyState<double, false>::determinant(const arma::uvec& set_l, const arma::uvec& set_q)
            // {
            //     auto W = this->_orbitals.submat(set_l, set_q);

            //     auto eigs = arma::eig_gen(W);

            //     _ty d_eig = arma::prod(eigs);
            //     _ty d_det = arma::det(W);

            //     if (std::abs(d_eig - d_det) > 1e-10) {
            //         std::cout << "W =\n" << W << std::endl;
            //         std::cout << "det      = " << d_det << std::endl;
            //         std::cout << "prod eig = " << d_eig << std::endl;
            //     }
            //     return d_eig;
            //     // return arma::det(W);
            //     // auto eigs = arma::eig_gen(W);
            //     // return std::real(arma::prod(eigs));
            // }
        }
    }
}