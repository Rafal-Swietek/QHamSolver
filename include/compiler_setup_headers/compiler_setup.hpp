#pragma once


#if !defined(FUN_SIGNATURE)
	#if defined (__GNUG__)
		#define FUN_SIGNATURE  __PRETTY_FUNCTION__
	#elif defined (_MSC_VER)
		#define FUN_SIGNATURE  __FUNCSIG__ 
	#elif defined(__INTEL_COMPILER)
		#define FUN_SIGNATURE  __FUNCTION__
	#else 
		#define FUN_SIGNATURE  __func__
	#endif
#endif


//! --------------------------------------------------------- STATIC CONDITION CHECK
#define TO_STRING_IMPL(x) #x
// convert to string literals
#define TO_STRING(x) TO_STRING_IMPL(x)

/// convert file line to string output
#define LINE_STR TO_STRING(__LINE__)

/// compiler-time check with assertion and pretty printing
#define static_check(condition, str_lit) static_assert(condition, __FILE__"(line=" LINE_STR "): " str_lit)

//! --------------------------------- ERROR MESSAGES
#define _INCOMPATIBLE_DIMENSION	"Incompatible dimensions. Different Hilbert space sizes"

#define NOT_CONSTRUCTIBLE	"ERROR 1: input is not constructible and cannot be stored in std::vector"
#define NOT_ALLOWED_INPUT	"ERROR 2: not allowed input to function"
#define NOT_CONVERTIBLE		"ERROR 3: not convertible to generic type"
#define BAD_INHERITANCE		"ERROR 4: given class is not inheriting from the appropriate base or links to class with 'final' keyword"
#define INCOMPATIBLE_SIZE	"ERROR 5: size of input arrays does not match"
#define INCOMPATIBLE_DIMENSION	"ERROR 6: " _INCOMPATIBLE_DIMENSION

//! --------------------------------------------------------- EXTRA DEBUG SETUP
#ifndef NODEBUG
	#define DEBUG
#endif

//<! pre-definition
void _current_profiling_info();

#if defined(EXTRA_DEBUG)
	#undef NODEBUG
	
	#define DESTRUCTOR_CALL std::cout << FUN_SIGNATURE << "->\tdestructor called" << std::endl << std::endl;
	#define CONSTRUCTOR_CALL std::cout << FUN_SIGNATURE << "->\tconstructor called" << std::endl << std::endl;
	#define _debug_start(expr)	expr; (void)_profile_memory_(); (void)_profile_cpu_();
	#define _debug_end(expr)	_current_profiling_info(); std::cout << "--------------->"; expr;

	//<! extra debug macro to hide if not code for runtime
	#define _extra_debug_(expr)	expr;
	#define _extra_debug(expr)	std::cout << "--------------->"; _extra_debug_(expr)
#else 
	#define DESTRUCTOR_CALL 
	#define CONSTRUCTOR_CALL
	#if defined(DEBUG)
		#define _debug_end(expr)	std::cout << "------>"; expr
	#else 
		#define _debug_end(expr)
	#endif
	#define _debug_start(expr) _debug_end(expr)

	//<! extra debug macro to hide if not code for runtime
	#define _extra_debug_(expr)
	#define _extra_debug(expr)
#endif

//! --------------------------------------------------------- DEBUG SETUP
#define stringize(x) #x
#ifndef _assert_
	#include <iostream>
	//<! DEFINE USER ASSERT
	#ifndef NODEBUG
		#define _assert_(Expr, Msg) \
			__M_Assert(stringize(Expr), Expr, __FILE__, __LINE__, FUN_SIGNATURE, Msg)
	#else
		#define _assert_(Expr, Msg) ;
	#endif

	inline void __M_Assert(const char* expr_str, bool expr, const char* file, int line, const char* _func_sign, const char* msg)
	{
		if (!expr)
		{
			std::cerr << "Assert failed:\t" << msg << "\n"
				<< "Expected:\t" << expr_str << "\n"
				<< "Source:\t\t" << file << ", line " << line << "\n"
				<< "Function:\t\t" << _func_sign << "\n";
			abort();
		}
	}
	inline void __M_Assert(const char* expr_str, bool expr, const char* file, int line, const char* _func_sign, std::string msg)
	{
		if (!expr)
		{
			std::cerr << "Assert failed:\t" << msg << "\n"
				<< "Expected:\t" << expr_str << "\n"
				<< "Source:\t\t" << file << ", line " << line << "\n"
				<< "Function:\t\t" << _func_sign << "\n";
			abort();
		}
	}
#endif


//! -------------------------------------------------------------------------------------------------------------- compiler
//#undef HAS_CXX17
//#undef HAS_CXX20

//! check compiler version, only C++17 or newer currently valid for this library
//! older versions are not suppeortd
#if !defined(_MSVC_LANG)
	#if (__cplusplus >= 202002L)
		#define HAS_CXX20
	#elif (__cplusplus >= 201703L)
		#define HAS_CXX17
	#else
		#error "--> at least C++17 compiler required; older versions are not suppeortd"
	#endif
	
#else 
	// MS has weird names for compiler version
	#if (_MSVC_LANG >= 202002L)
		#define HAS_CXX20 true
	#elif (_MSVC_LANG >= 201703L)
		#define HAS_CXX17 true
	#else
		#error "--> C++17 compiler required; older versions are not supported"
	#endif
	
#endif

#ifdef HAS_CXX20
	#pragma message ("--> Compiling with c++20 compiler. Metaprogramming fully unlocked.")
#elif defined HAS_CXX17
	#pragma message ("--> Compiling with c++17 compiler. Failed to use type_traits and concepts in metaprogramming")
#else
	#pragma message ("--> Ensuring correct types in metaprogramming disabled.\n Check user types against usage in code to omit program failure.")
#endif


//! ---------------------------------------------------- define builtin functions
#ifdef _MSC_VER
#include <intrin.h>
#include <nmmintrin.h>
#define __builtin_popcount __popcnt
#define __builtin_popcountll _mm_popcnt_u64
#endif

// #ifdef HAS_CXX20 && !defined( __clang__ )
// 	#define __builtin_popcountll std::popcount
// #else
// 	#define __builtin_popcountll __builtin_popcountll
// #endif

//<! for generic lambda input: has to be a callable
#if defined(HAS_CXX20)
	#define callable_type typename//std::invocable
#else
	#define callable_type typename
#endif