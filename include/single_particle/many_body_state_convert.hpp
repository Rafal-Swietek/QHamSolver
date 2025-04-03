#pragma once

namespace QHS{
    namespace single_particle{

        namespace slater{

            /// @brief Convert gaussian state (input) to many-body state (reference)
            /// @tparam _ty type of input orbitals
            /// @tparam use_U1_decomp Create ManyBody state in U(1) subspace
            /// @param many_body_state reference to vector denoting the many-body state
            /// @param gaussian_state gaussian state to transform to many-body space (as dynamic_bitset)
            template <typename _ty, bool use_U1_decomp>
            inline
            void ManyBodyState<_ty, use_U1_decomp>::convert(arma::Col<_ty>& many_body_state, const boost::dynamic_bitset<>& gaussian_state)
            {
                _assert_(gaussian_state.size() == this->volume && gaussian_state.count() == this->num_particles, 
                            INCOMPATIBLE_DIMENSION "Input gaussian state does not match class' system size");
                // _assert_(many_body_state.size() == ULLPOW(this->volume),
                //             INCOMPATIBLE_DIMENSION "Input many body state does not match");
                
                arma::uvec set_q = this->_set_indices(gaussian_state);

            #pragma omp parallel for// num_threads(outer_threads) schedule(dynamic)
                for(long k = 0; k < this->_hilbert_space.get_hilbert_space_size(); k++){
                    u64 state_idx = this->_hilbert_space(k);
                
                    arma::uvec set_l = this->_set_ell_indices(state_idx);
                    
                    u64 idx_in_state = state_idx;
                    if constexpr (use_U1_decomp == true)
                        idx_in_state = k;
                    
                    many_body_state(idx_in_state) += this->determinant(set_l, set_q);
                }
            }
            
            /// @brief Convert gaussian state (input) to many-body state (reference) with additional prefactor. Used when summing over different gaussian states
            /// @tparam _ty type of input orbitals
            /// @tparam use_U1_decomp Create ManyBody state in U(1) subspace
            /// @param many_body_state reference to vector denoting the many-body state
            /// @param gaussian_state gaussian state to transform to many-body space (as dynamic_bitset)
            /// @param prefactor complex prefactor to state
            template <typename _ty, bool use_U1_decomp>
            inline
            void ManyBodyState<_ty, use_U1_decomp>::convert(arma::Col<_ty>& many_body_state, const boost::dynamic_bitset<>& gaussian_state, _ty prefactor)
            {
                _assert_(gaussian_state.size() == this->volume && gaussian_state.count() == this->num_particles, 
                            INCOMPATIBLE_DIMENSION "Input gaussian state does not match class' system size");
                // _assert_(many_body_state.size() == ULLPOW(this->volume),
                //             INCOMPATIBLE_DIMENSION "Input many body state does not match class' Hilbert space");
                
                arma::uvec set_q = this->_set_indices(gaussian_state);

            #pragma omp parallel for// num_threads(outer_threads) schedule(dynamic)
                for(long k = 0; k < this->_hilbert_space.get_hilbert_space_size(); k++){
                    u64 state_idx = this->_hilbert_space(k);
                
                    arma::uvec set_l = this->_set_ell_indices(state_idx);
                    
                    u64 idx_in_state = state_idx;
                    if constexpr (use_U1_decomp == true)
                        idx_in_state = k;
                    
                    many_body_state(idx_in_state) += prefactor * this->determinant(set_l, set_q);
                }
            }

            /// @brief Convert gaussian state (input) to many-body state (reference) with additional prefactor and fill participation ratios. Used when summing over different gaussian states
            /// @tparam _ty type of input orbitals
            /// @tparam use_U1_decomp Create ManyBody state in U(1) subspace
            /// @param many_body_state reference to vector denoting the many-body state
            /// @param gaussian_state gaussian state to transform to many-body space (as dynamic_bitset)
            /// @param prefactor complex prefactor to state
            template <typename _ty, bool use_U1_decomp>
            inline
            void ManyBodyState<_ty, use_U1_decomp>::convert(arma::Col<_ty>& many_body_state, const boost::dynamic_bitset<>& gaussian_state, _ty prefactor, arma::vec qs, arma::vec& prs)
            {
                _assert_(gaussian_state.size() == this->volume && gaussian_state.count() == this->num_particles, 
                            INCOMPATIBLE_DIMENSION "Input gaussian state does not match class' system size");
                // _assert_(many_body_state.size() == ULLPOW(this->volume),
                //             INCOMPATIBLE_DIMENSION "Input many body state does not match class' Hilbert space");
                
                arma::uvec set_q = this->_set_indices(gaussian_state);

            #pragma omp parallel for// num_threads(outer_threads) schedule(dynamic)
                for(long k = 0; k < this->_hilbert_space.get_hilbert_space_size(); k++){
                    u64 state_idx = this->_hilbert_space(k);
                
                    arma::uvec set_l = this->_set_ell_indices(state_idx);

                    _ty det = this->determinant(set_l, set_q);
                    
                    u64 idx_in_state = state_idx;
                    if constexpr (use_U1_decomp == true)
                        idx_in_state = k;
                    
                    many_body_state(idx_in_state) += prefactor * det;

                    // for(int ii = 0; ii < qs.size(); ii++){
                    //     double q = qs(ii);
                    //     prs(ii) += std::pow(std::abs(prefactor * det), 2 * q);
                    // }
                }
            }

            /// @brief Convert gaussian state (input) to many-body state
            /// @tparam _ty type of input orbitals
            /// @tparam use_U1_decomp Create ManyBody state in U(1) subspace
            /// @param gaussian_state gaussian state to transform to many-body space (as dynamic_bitset)
            /// @return vector denoting the many-body state
            template <typename _ty, bool use_U1_decomp>
            inline
            arma::Col<_ty> ManyBodyState<_ty, use_U1_decomp>::convert(const boost::dynamic_bitset<>& gaussian_state)
            {
                _assert_(gaussian_state.size() == this->volume && gaussian_state.count() == this->num_particles, 
                            INCOMPATIBLE_DIMENSION "Input gaussian state does not match class' system size");
                arma::Col<_ty> many_body_state;
                try_alloc_vector(many_body_state, ULLPOW(this->volume) );

                arma::uvec set_q = this->_set_indices(gaussian_state);
                std::cout << set_q.t();
            #pragma omp parallel for// num_threads(outer_threads) schedule(dynamic)
                for(long k = 0; k < this->_hilbert_space.get_hilbert_space_size(); k++){
                    u64 state_idx = this->_hilbert_space(k);
                    
                    arma::uvec set_l = this->_set_ell_indices(state_idx);
                    
                    u64 idx_in_state = state_idx;
                    if constexpr (use_U1_decomp == true)
                        idx_in_state = k;
                    
                    many_body_state(idx_in_state) = this->determinant(set_l, set_q);
                }

                return many_body_state;
            }

        }
    }

}