#pragma once

namespace operators{
    namespace fermions{
        namespace spinless{ 
            /// @brief SigmaZ operator on input site
            /// @tparam _ty template for return type
            /// @param base_vec Input state to act SigmaZ on
            /// @param L system size
            /// @param site site to act operator
            /// @return pair of return value and resulting state
            template <typename _ty>
            inline
            std::pair<_ty, u64> 
            density(u64 base_vec, unsigned int L, int site) {
                return std::make_pair(
                    checkBit(base_vec, L - 1 - site) ? 1 : 0,
                    base_vec
                    );
            };

            /// @brief Sigma+ operator on input site
            /// @tparam _ty template for return type
            /// @param base_vec Input state to act Sigma+ on
            /// @param L system size
            /// @param site site to act operator
            /// @return pair of return value and resulting state
            template <typename _ty>
            inline
            std::pair<_ty, u64> 
            create(u64 base_vec, unsigned int L, int site) {
                u64 mask = reverseBits( ULLPOW(site)-1, L );
                double sign = (__builtin_popcountll(base_vec & mask) % 2)? -1 : +1;
                return std::make_pair(
                    _ty( checkBit(base_vec, L - 1 - site)? 0.0 : sign ),
                    flip(base_vec, BinaryPowers[L - 1 - site], L - 1 - site)
                    );
            };

            /// @brief Sigma- operator on input site
            /// @tparam _ty template for return type
            /// @param base_vec Input state to act Sigma- on
            /// @param L system size
            /// @param site site to act operator
            /// @return pair of return value and resulting state
            template <typename _ty>
            inline
            std::pair<_ty, u64> 
            anihilate(u64 base_vec, unsigned int L, int site) {
                u64 mask = reverseBits( ULLPOW(site)-1, L );
                double sign = (__builtin_popcountll(base_vec & mask) % 2)? -1 : +1;
                return std::make_pair(
                    _ty( checkBit(base_vec, L - 1 - site)? sign : 0.0 ),
                    flip(base_vec, BinaryPowers[L - 1 - site], L - 1 - site)
                    );
            };
        }
    }
    namespace spin_half{

    }
}
