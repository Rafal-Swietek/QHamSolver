#pragma once

//! --------------------------------------------------------------------------------
//! --------------------------------------------------------BUILT-IN OPERATORS CLASS
namespace op {

	enum class __builtin_operators {
		e,		  // neutral element
		P,		  // parity (reverse)
		Zx,		  // global spin flip in X
		Zy,		  // global spin flip in Y
		Zz,		  // global spin flip in Z
		T,		  // translation
		digit,	  // check digit at specified position
		Z_i,	  // flip spin locally
	};	// built-in operators

	// namespace holding builtin function generators
	// with L (system size) and config (system configuration) as input
	// the function return a lambda with specified operation, capturing this input
	namespace __builtins {

		//! pre-calculate powers at operator setting level
		inline
		auto calculate_powers() -> std::vector<u64>
		{
			std::vector<u64> powers;
			for (int i = 0; i < 64 / block_size; i++)	// fill powers with config^0, config^1,... up til 2^63 (maximum of uint64_t)
				powers.push_back(ULLPOW(block_size * i));
			return powers;
		};


		//! ---------------------------------------------- SYMMETRY GENERATORS
		//! 
		//! parity generator: inverse order of binary/octal/.. string, i.e->binary-> P * |010111> = |111010>
		inline
		auto parity(unsigned int L) -> _global_fun
		{
			return [L](u64 n)
			{
				u64 m = 0;
				int dummy = L;
				while (dummy > 0) {
					m = (m << block_size) | (n & u64(config - 1ULL));
					n = (n >> block_size);
					--dummy;
				}
				return std::make_pair(m, 1.0);
			};
		};
		//!
		//! spin-flip generator: flip all digits of binary/octal/.. string, i.e->binary-> Z * |010111> = |101000>
		inline
		auto spin_flip_x(unsigned int L) -> _global_fun
			{ return [L](u64 n) { return std::make_pair(powers[L] - 1 - n, 1.0); }; };
		
		inline
		auto spin_flip_y(unsigned int L) -> _global_fun
			{ 
				static_check((config == 2), ONLY_SPIN_HALF_OEPRATOR);
				return [L](u64 n) { 
							const int _num_of_down_spins = L - __builtin_popcountll(n);
							double sign = -2.0 * (_num_of_down_spins % 2) + 1.0;
							return std::make_pair(powers[L] - 1 - n, pow_im(L) * sign); 
						}; 
			};
		
		inline
		auto spin_flip_z(unsigned int L) -> _global_fun
			{ 
				static_check((config == 2), ONLY_SPIN_HALF_OEPRATOR);
				return [L](u64 n) { 
							const int _num_of_down_spins = L - __builtin_popcountll(n);
							double sign = -2.0 * (_num_of_down_spins % 2) + 1.0;
							return std::make_pair(n, sign); 
						}; 
			};
		

		//! translation generator: shift order of binary/octal/.. string to the left, i.e->binary-> T * |010111> = |101110>
		inline
		auto translation(unsigned int L) -> _global_fun
		{
			return [L](u64 n)
					{
						u64 rotate = block_size * (L - 1ULL);		// shift generator of all blocks of bits except the first one
						u64 first_digit = n &
							(u64(config - 1ULL)
								<< rotate);						// conjuction of 11.. and the first digit in binary (11.. is shifted to the position of the first digit)
						u64 other_digit = n - first_digit;		// remaining digits
						u64 final_state = (other_digit << block_size)
							| (first_digit >> rotate);			// first part rotates the remaining digits (or block of bits) by left_shift by 'blocks' positions\
																							(equivalent to one position in octal code), while the latter shifts the first digit to the end
						return std::make_pair(final_state, 1.0);
					};
		};


		//! ---------------------------------------------- OTHER BIT OEPRATIONS
		//! checks the digit at the current position
		inline
		auto get_digit(unsigned int L) -> _ifun
		{
			return [L](u64 n, int bit_pos)
					{
						short int pos = L - bit_pos - 1;
						return (n & (u64(config - 1ULL) << block_size * pos) ) / powers[pos];
					};
		};

		//! flips a bit at the specified position
		inline
		auto flip(unsigned int L) -> _local_fun
		{
			return [L](u64 n, int bit_pos)
				{
					u64 result = (u64(config - 1ULL)
						<< (block_size * (L - bit_pos - 1)) ) ^ n;				// XOR the digit at bit_pos 
					return std::make_pair(result, 1.0);
				};
		};

	};


	inline
	auto
	choose_symmetry(__builtin_operators sym, unsigned int L)
	{
		switch(sym){
			case __builtin_operators::T: 		return __builtins::translation(L);
			case __builtin_operators::P: 		return __builtins::parity(L);
			case __builtin_operators::Zx: 		return __builtins::spin_flip_x(L);
			case __builtin_operators::Zy: 		return __builtins::spin_flip_y(L);
			case __builtin_operators::Zz: 		return __builtins::spin_flip_z(L);
			default:
				std::cout << "No other operator implemented. Using translation as default" << std::endl;
				return __builtins::translation(L);
		}
	}
};
