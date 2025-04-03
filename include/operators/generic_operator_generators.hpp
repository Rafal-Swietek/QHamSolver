#pragma once

namespace QOps{

    /// @brief Creates permutation generator for permutation p
    /// @param L system size
    /// @param p permutation vector (stores how the lattice sites are permuted)
    /// @return permutation generator
    template <typename _ty>
    inline
    auto _permutation_generator(unsigned int L, std::vector<int> p)
    {
        auto _kernel = __builtins::permutation<_ty>(L, p);
        return generic_operator<_ty>(L, std::move(_kernel), _ty(1.0));
    }




};