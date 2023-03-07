#pragma once

#include "../QHamSolver.h"
//TODO: So far only for spin chains easy implementation, easy to change

enum class avail_symmetries {
	T,		// translation
	P,		        // parity
	Zx,		// spin-flip X
	Zy,		// spin-flip Y
	Zz		// spin-flip Z
};

/// <summary>
/// Check the k'th bit
/// </summary>
/// <param name="n">Number on which the bit shall be checked</param>
/// <param name="k">number of bit (from 0 to 63)</param>
/// <returns>Bool on if the bit is set or not</returns>
inline bool checkBit(u64 n, int k) {
	return n & ULLPOW(k);
}

namespace _builtins{
    
    /// @brief 
    /// @param n 
    /// @param L 
    /// @return 
    inline u64 translation(u64 n, unsigned int L) 
        {
            u64 maxPower = BinaryPowers[L - int32_t(1)];
	        return (n >= maxPower) ? (((int64_t)n - (int64_t)maxPower) * 2 + 1) : n * 2;    
        }

    /// @brief 
    /// @param n 
    /// @param L 
    /// @return 
    inline u64 spin_flip_x(u64 n, unsigned int L) 
        { return BinaryPowers[L] - n - 1; }

    /// @brief 
    /// @param n 
    /// @param L 
    /// @return 
    inline u64 unit(u64 n, unsigned int L) 
        { return n; }


    /// @brief 
    /// @param n 
    /// @param L 
    /// @return 
    inline u64 parity(u64 n, unsigned int L) 
        { 
            u64 rev = (lookup[n & 0xffULL] << 56) |					// consider the first 8 bits
            (lookup[(n >> 8) & 0xffULL] << 48) |				// consider the next 8 bits
            (lookup[(n >> 16) & 0xffULL] << 40) |				// consider the next 8 bits
            (lookup[(n >> 24) & 0xffULL] << 32) |				// consider the next 8 bits
            (lookup[(n >> 32) & 0xffULL] << 24) |				// consider the next 8 bits
            (lookup[(n >> 40) & 0xffULL] << 16) |				// consider the next 8 bits
            (lookup[(n >> 48) & 0xffULL] << 8) |				// consider the next 8 bits
            (lookup[(n >> 54) & 0xffULL]);						// consider last 8 bits
            
            return (rev >> (64 - L));								// get back to the original maximal number
        }

};

