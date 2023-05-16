#pragma once

namespace op{

    /// @brief Creates permutation generator for permutation p
    /// @param L system size
    /// @param p permutation vector (stores how the lattice sites are permuted)
    /// @return permutation generator
    inline
    auto _permutation_generator(unsigned int L, std::vector<int> p)
    {
        auto _kernel = __builtins::permutation(L, p);
        return generic_operator<>(L, _kernel, 1.0);
    }




};