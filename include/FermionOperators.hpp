#pragma once

namespace operators{
    namespace fermions{
        namespace spinless{

            inline
            double
            calculate_sign(u64 base_vec, unsigned int L, int site){
                u64 mask = reverseBits( ULLPOW(site)-1, L );
                // u64 mask = ULLPOW(site)-1;
                return (__builtin_popcountll(base_vec & mask) % 2)? -1 : +1;
            };

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
                double sign = calculate_sign(base_vec, L, site);
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
                double sign = calculate_sign(base_vec, L, site);
                return std::make_pair(
                    _ty( checkBit(base_vec, L - 1 - site)? sign : 0.0 ),
                    flip(base_vec, BinaryPowers[L - 1 - site], L - 1 - site)
                    );
            };
        }
        namespace spin_half{

                inline
                double
                calculate_sign(u64 base_vec, unsigned int L, int site){
                    u64 mask = reverseBits( ULLPOW(site)-1, 2*L );
                    // u64 mask = ULLPOW(site)-1;
                    // std::cout << "\n" << site << "\t" << boost::dynamic_bitset<>(2*L, base_vec) << "\t" << boost::dynamic_bitset<>(2*L, mask) << std::endl;
                    return (__builtin_popcountll(base_vec & mask) % 2)? -1 : +1;
                };

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
                    int pos = 2*L - 1 - 2*site;
                    return std::make_pair(
                        (_ty)checkBit(base_vec, pos) + (_ty)checkBit(base_vec, pos-1),
                        base_vec
                        );
                };

                /// @brief Anihilate spin-up particle in input site on site site
                /// @tparam _ty template for return type
                /// @param base_vec Input state to act operator on
                /// @param L system size
                /// @param site site to act operator
                /// @return pair of return value and resulting state
                template <typename _ty>
                inline
                std::pair<_ty, u64> 
                create_up(u64 base_vec, unsigned int L, int site) {
                    double sign = calculate_sign(base_vec, L, 2*site);
                    int pos = 2*L - 1 - 2*site;
                    return std::make_pair(
                        _ty( checkBit(base_vec, pos)? 0.0 : sign ),
                        flip(base_vec, BinaryPowers[pos], pos)
                        );
                };

                /// @brief Anihilate spin-up particle in input site on site site
                /// @tparam _ty template for return type
                /// @param base_vec Input state to act operator on
                /// @param L system size
                /// @param site site to act operator
                /// @return pair of return value and resulting state
                template <typename _ty>
                inline
                std::pair<_ty, u64> 
                anihilate_up(u64 base_vec, unsigned int L, int site) {
                    double sign = calculate_sign(base_vec, L, 2*site);
                    int pos = 2*L - 1 - 2*site;
                    return std::make_pair(
                        _ty( checkBit(base_vec, pos)? sign : 0.0 ),
                        flip(base_vec, BinaryPowers[pos], pos)
                        );
                };

                /// @brief Create spin-down particle in input site on site site
                /// @tparam _ty template for return type
                /// @param base_vec Input state to act operator on
                /// @param L system size
                /// @param site site to act operator
                /// @return pair of return value and resulting state
                template <typename _ty>
                inline
                std::pair<_ty, u64> 
                create_down(u64 base_vec, unsigned int L, int site) {
                    double sign = calculate_sign(base_vec, L, 2*site+1);
                    int pos = 2*L - 2*site - 2;
                    return std::make_pair(
                        _ty( checkBit(base_vec, pos)? 0.0 : sign ),
                        flip(base_vec, BinaryPowers[pos], pos)
                        );
                };

                /// @brief Anihilate spin-down particle in input site on site site
                /// @tparam _ty template for return type
                /// @param base_vec Input state to act operator on
                /// @param L system size
                /// @param site site to act operator
                /// @return pair of return value and resulting state
                template <typename _ty>
                inline
                std::pair<_ty, u64> 
                anihilate_down(u64 base_vec, unsigned int L, int site) {
                    double sign = calculate_sign(base_vec, L, 2*site+1);
                    int pos = 2*L - 2*site - 2;
                    return std::make_pair(
                        _ty( checkBit(base_vec, pos)? sign : 0.0 ),
                        flip(base_vec, BinaryPowers[pos], pos)
                        );
                };
        }
    }
}
