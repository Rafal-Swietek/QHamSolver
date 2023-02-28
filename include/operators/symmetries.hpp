#pragma once

//#define NOP
namespace op{
    
    /// @brief Create generic_operator<> for symmetry generator
    /// @param L system size (defines hilbert space)
    /// @param gen choose among fixed __builtin generators
    /// @param sym_eig_val symmetry eigenvalue (defined through sector)
    /// @return symmetry generator
    inline
    generic_operator<>
    symmetry(int L, __builtin_operators gen, cpx sym_eig_val)
    { 
        auto _kernel = choose_symmetry(gen, L); 
        return generic_operator<>(L, _kernel, sym_eig_val);
    }

    /// @brief Creates translation geerator for given sector and hilbert space
    /// @param L system size (defines hilbert space)
    /// @param sector symmetry sector (int) --> quasimomentum sector is calculated within
    /// @return translation geerator
    inline
    auto _translation_symmetry(int L, int sector)
    { 
        _assert_((sector >= 0 && sector < L), NOT_ALLOWED_SYM_SECTOR);
        const double ksym = two_pi * (double)sector / L;
        return symmetry(L, __builtin_operators::T, std::exp(-im * ksym));
    }

    /// @brief Creates parity generator for given sector and hilbert space
    /// @param L system size (defines hilbert space)
    /// @param sector symmetry sector (int)
    /// @return parity generator
    inline
    auto _parity_symmetry(int L, int sector)
        { _assert_((sector == -1 || sector == 1), NOT_ALLOWED_SYM_SECTOR);
          return symmetry(L, __builtin_operators::P, (double)sector); }
    
    /// @brief Creates spin flip in X generator for given sector and hilbert space
    /// @param L system size (defines hilbert space)
    /// @param sector symmetry sector (int)
    /// @return spin flip in X generator
    inline
    auto _spin_flip_x_symmetry(int L, int sector)
        { _assert_((sector == -1 || sector == 1), NOT_ALLOWED_SYM_SECTOR);
          return symmetry(L, __builtin_operators::Zx, (double)sector); }
    
    /// @brief Creates spin flip in Y generator for given sector and hilbert space
    /// @param L system size (defines hilbert space)
    /// @param sector symmetry sector (int)
    /// @return spin flip in Y generator
    inline
    auto _spin_flip_y_symmetry(int L, int sector)
        { _assert_((sector == -1 || sector == 1), NOT_ALLOWED_SYM_SECTOR);
          return symmetry(L, __builtin_operators::Zy, (double)sector); }
    
    /// @brief Creates spin flip in Z generator for given sector and hilbert space
    /// @param L system size (defines hilbert space)
    /// @param sector symmetry sector (int)
    /// @return spin flip in Z generator
    inline
    auto _spin_flip_z_symmetry(int L, int sector)
        { _assert_((sector == -1 || sector == 1), NOT_ALLOWED_SYM_SECTOR);
          return symmetry(L, __builtin_operators::Zz, (double)sector); }


};