#pragma once

//#define NOP
namespace op{
    
    /// @brief Create generic_operator<> for symmetry generator
    /// @param L system size (defines hilbert space)
    /// @param gen choose among fixed __builtin generators
    /// @param sym_eig_val symmetry eigenvalue (defined through sector)
    /// @param arg additional parameter for symmetry generator (is not necessary), ie translation has shift size
    /// @return symmetry generator
    inline
    generic_operator<>
    symmetry(unsigned int L, __builtin_operators gen, cpx sym_eig_val, int arg = 1)
    { 
        auto _kernel = choose_symmetry(gen, L, arg); 
        return generic_operator<>(L, _kernel, sym_eig_val);
    }

    /// @brief Creates translation geerator for given sector and hilbert space
    /// @param L system size (defines hilbert space)
    /// @param sector symmetry sector (int) --> quasimomentum sector is calculated within
    /// @return translation geerator
    inline
    auto _translation_symmetry(unsigned int L, int sector, bool inverse = false, int shift = 1)
    { 
        _assert_((sector >= 0 && sector < L), NOT_ALLOWED_SYM_SECTOR);
        const double ksym = two_pi * (double)sector / L * double(shift);    // eigenvalue of T^l is e^(-ikl) with l = shift
        if(inverse) return symmetry(L, __builtin_operators::Tinv, std::exp(im * ksym), arg);
        else        return symmetry(L, __builtin_operators::T,    std::exp(-im * ksym), arg);
    }

    /// @brief Creates parity generator for given sector and hilbert space
    /// @param L system size (defines hilbert space)
    /// @param sector symmetry sector (int)
    /// @return parity generator
    inline
    auto _parity_symmetry(unsigned int L, int sector)
        { _assert_((sector == -1 || sector == 1), NOT_ALLOWED_SYM_SECTOR);
          return symmetry(L, __builtin_operators::P, (double)sector); }
    
    /// @brief Creates spin flip in X generator for given sector and hilbert space
    /// @param L system size (defines hilbert space)
    /// @param sector symmetry sector (int)
    /// @return spin flip in X generator
    inline
    auto _spin_flip_x_symmetry(unsigned int L, int sector)
        { _assert_((sector == -1 || sector == 1), NOT_ALLOWED_SYM_SECTOR);
          return symmetry(L, __builtin_operators::Zx, (double)sector); }
    
    /// @brief Creates spin flip in Y generator for given sector and hilbert space
    /// @param L system size (defines hilbert space)
    /// @param sector symmetry sector (int)
    /// @return spin flip in Y generator
    inline
    auto _spin_flip_y_symmetry(unsigned int L, int sector)
        { _assert_((sector == -1 || sector == 1), NOT_ALLOWED_SYM_SECTOR);
          return symmetry(L, __builtin_operators::Zy, (double)sector); }
    
    /// @brief Creates spin flip in Z generator for given sector and hilbert space
    /// @param L system size (defines hilbert space)
    /// @param sector symmetry sector (int)
    /// @return spin flip in Z generator
    inline
    auto _spin_flip_z_symmetry(unsigned int L, int sector)
        { _assert_((sector == -1 || sector == 1), NOT_ALLOWED_SYM_SECTOR);
          return symmetry(L, __builtin_operators::Zz, (double)sector); }


};