#pragma once

#ifndef CONFIG
	#define CONFIG 2
#endif

/// @brief calculate block size in bit represenation of sttae
/// @param of input nuber (local hilbert space dimension)
/// @return block size
constexpr int _block(int of)
	{ return (int) gcem::ceil(gcem::log2(of)); }

constexpr int config = CONFIG;				// base configuration (i.e. 2->binary, 8->octal code,...)
constexpr int block_size = _block(config);	// number of spins per site (2->1, 3->2, 4->2, 8->3, ...)


const v_1d<u64> powers =  []{ 	v_1d<u64> a(64 / block_size);
								for (int i = 0; i < 64 / block_size; ++i) 
									{ a[i] = ULLPOW(block_size * i); }
								return a;
							}(); 					// vector containing powers of config from config^0 to config^(L-1) (filled with lambda)


//! --------------------------------- ERROR MESSAGES FOR OPERATOR CLASSES
#define INCOMPATIBLE_DIMENSION	"ERROR OP-1: Incompatible operator dimensions. Different Hilbert space sizes"
#define NOT_ALLOWED_INPUT_OP	"ERROR OP-2: input not amongst allowed operators! See ALLOWED_OPERATORS macro"
#define ONLY_EMPTY_TEMPLATE		"ERROR OP-3: operator/function only allowed for non-templated instance"
#define NOT_ALLOWED_OPERATION	"ERROR OP-4: not permitted operation! See definition"
#define ONLY_SPIN_HALF_OEPRATOR	"ERROR OP-5: operator only valid for spin-half systems (config == 2) (to be extended)"
#define NOT_ALLOWED_SYM_SECTOR	"ERROR OP-6: (symmetry) not allowed symmetry sector, check available"
//! --------------------------------- HELPER FUNCTIONS
//! --- compare hilbert spaces
#define CHOOSE_MACRO(_1,_2,NAME,...) NAME
#define check_hilbert_space_this(op_rhs)			(( this->L == op_rhs.L))
#define check_hilbert_space_input(op_lhs, op_rhs)	((op_lhs.L == op_rhs.L))

#define check_hilbert_space(...) CHOOSE_MACRO(__VA_ARGS__, check_hilbert_space_input, check_hilbert_space_this)(__VA_ARGS__)

//! --- sth



//! --------------------------------- SETUP RUN-TIME CHECKS
#if defined(DISABLE_DEBUG)
	#define assert_hilbert_space(...) __PRAGMA message( "Debug disabled. No bounds check on operator Hilbert spaces. Check dimensionality yourself!")
#else
	#define assert_hilbert_space(...) assert(check_hilbert_space(__VA_ARGS__) && INCOMPATIBLE_DIMENSION)
#endif


//! --------------------------------- TODOS
#if CONFIG > 2
    #pragma message("---> Generalize U(1) hilbert space to arbitrary model, i.e. hubbard, spin-1, Kondo-Heisenberg...")
#endif