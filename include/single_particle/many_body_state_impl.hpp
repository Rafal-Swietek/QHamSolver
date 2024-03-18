#pragma once

namespace QHS{

    namespace single_particle{

        namespace slater{

            /// @brief Initialize the ManyBodyState class to convert from gaussian to many-body states
            /// @tparam _ty type of input orbitals
            template <typename _ty>
            inline 
            void ManyBodyState<_ty>::initialize() 
            {
                CONSTRUCTOR_CALL;
                this->check_spin = QOps::__builtins::get_digit(this->volume);
            }
            
            /// @brief Initialize the ManyBodyState class to convert from gaussian to many-body states
            /// @tparam _ty type of input orbitals
            /// @param gaussian_state Input gaussian state (bitset) as quasiparticle product state
            template <typename _ty>
            inline
            arma::uvec ManyBodyState<_ty>::set_indices(const boost::dynamic_bitset<>& gaussian_state, int N)
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
            /// @param gaussian_state Input gaussian state (bitset) as quasiparticle product state
            template <typename _ty>
            inline
            arma::uvec ManyBodyState<_ty>::_set_indices(const boost::dynamic_bitset<>& gaussian_state)
            {
                return set_indices(gaussian_state, this->num_particles);
            }

            /// @brief Initialize the ManyBodyState class to convert from gaussian to many-body states
            /// @tparam _ty type of input orbitals
            /// @param state_idx Input gaussian state (bitset) as quasiparticle product state
            template <typename _ty>
            inline
            arma::uvec ManyBodyState<_ty>::_set_ell_indices(u64 state_idx)
            {
                arma::uvec set_ell(this->num_particles, arma::fill::zeros);
                int count = 0;
                for(int id = 0; id < this->volume; id++){
                    if( this->check_spin(state_idx, this->volume - id - 1) ){
                        set_ell(count) = id;
                        count++;
                    }
                }

                return set_ell;
            }

            /// @brief Calculate Slater determinant of gaussian state (given by set indices set_q) with product state (given by set indices set_l)
            /// @param set_l set indices of current product state
            /// @param set_q sert indices of gaussian state of interest
            /// @return slater determinant for given indices
            template <typename _ty>
            inline
            _ty ManyBodyState<_ty>::determinant(const arma::uvec& set_l, const arma::uvec& set_q)
            {
                auto W = this->_orbitals.submat(set_l, set_q);
                auto eigs = arma::eig_gen(W);
                return arma::prod(eigs);
            }
        }
    }
}