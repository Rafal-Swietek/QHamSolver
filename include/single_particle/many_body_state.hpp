#pragma once
#ifndef _MANYBODY_STATE
#define _MANYBODY_STATE

namespace QHS{
    namespace single_particle{

        namespace slater{

            template <typename _ty>
            class ManyBodyState
            {
                const arma::Mat<_ty>& _orbitals;
                U1_hilbert_space<U1::charge, true> _hilbert_space;
                QOps::_ifun check_spin;

                int volume;
                int num_particles;

                void initialize();

                arma::uvec _set_indices(const boost::dynamic_bitset<>& gaussian_state);
                arma::uvec _set_ell_indices(u64 state_idx);
                
                _ty determinant(const arma::uvec& set_l, const arma::uvec& set_q);
            public:

                static inline arma::uvec set_indices(const boost::dynamic_bitset<>& gaussian_state, int N);

                //------------------------------------------------------------------------------------------------ CONSTRUCTOS
                ~ManyBodyState() { DESTRUCTOR_CALL; };
                ManyBodyState() = delete;

                /// @brief Constructor of Many Body Gaussian state creator
                /// @param orbitals_in single particle orbitals
                /// @param V system volume 
                /// @param N number of particles
                explicit ManyBodyState(const arma::Mat<_ty>& orbitals_in, int V, int N)
                    : _orbitals(orbitals_in), volume(V), num_particles(N)
                    {
                        this->_hilbert_space = U1_hilbert_space<U1::charge, true>(this->system_size, this->num_particles);
                        initialize(); 
                    }

                /// @brief Constructor of Many Body Gaussian state creator
                /// @param orbitals_in single particle orbitals
                /// @param _hilbert_in Input U(1) hilbert space
                explicit ManyBodyState(const arma::Mat<_ty>& orbitals_in, const U1_hilbert_space<U1::charge, true>& _hilbert_in)
                    : _orbitals(orbitals_in), _hilbert_space(_hilbert_in)
                    {
                        auto [V, N] = this->_hilbert_space.get_U1_params();
                        this->volume        = V;
                        this->num_particles = N;
                        initialize(); 
                    }
                
                //------------------------------------------------------------------------------------------------ CAST STATES TO FULL HILBERT SPACE:

                void convert(arma::Col<_ty>& many_body_state,   const boost::dynamic_bitset<>& gaussian_state);
                void convert(arma::cx_vec& many_body_state,     const boost::dynamic_bitset<>& gaussian_state, cpx prefactor);

                arma::Col<_ty> convert(const boost::dynamic_bitset<>& gaussian_state);
                
                //------------------------------------------------------------------------------------------------ GENERATE CONFIGURATIONS

            };


        }
    }
}
#include "many_body_state_impl.hpp"
#include "many_body_state_convert.hpp"


#endif