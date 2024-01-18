#ifndef __FUNALGEBRA
#define __FUNALGEBRA

#include <complex>
#include <functional>				// std::function
#include "typedef_functions.h"
//--------------------------------------------------------------------------------
//------------------------------------------------MULTIPLICATION OF STD::FUNCTIONS
namespace QOps {
	//! function behavior: f*g == f(g(n,...),...)
	//! f*g |n> = f*g_nm |m> = f_mk * g_nm |k>
	// multiplication for generic input functions
	template <typename... _Type1, typename... _Type2 >
	inline auto operator*(
		_func< return_type >::input< _Type1... > f,
		_func< return_type >::input< _Type2... > g
		)
	{
		return [f, g](u64 num,
			_Type1... args1,
			_Type2... args2) -> return_type
		{
			auto [state, val] = g(num, args2...);
			auto [state_final, ret_final] = f(state, args1...);
			return std::make_pair(state_final, val * ret_final);
		};
	};

	// without template expansion
	template <typename... _Type>
	inline auto operator%(
		_func< return_type >::input< _Type... > f,
		_func< return_type >::input< _Type... > g
		)
	{
		return [f, g](u64 num,
			_Type... args) -> return_type
		{
			auto [state, val] = g(num, args...);
			auto [state_final, ret_final] = f(state, args...);
			return std::make_pair(state_final, val * ret_final);
		};
	};

	//------------------------------------------------------------- ADDITIONAL ALGEBRA ON VECTORS
#if defined(HAS_CXX20)
	template <
		has_multiplication A,
		has_multiplication B		// ensures classes have overloaded operator*			
	>
#else
#pragma message ("The product of multiplication of instances of input classes has to be copy constructible (because using std::any as result)")
	template <class A, class B>
#endif
	inline auto operator*(
		const std::vector<A>& _left,
		const std::vector<B>& _right
		)
		-> std::vector<decltype(_left[0] * _right[0])> // returns vector of different classes
	{	
		// ensures that A * B is allowed to store in std::any (has to be copy constructible)
		static_check((traits::is_copy_constructible_v<A, B>), NOT_CONSTRUCTIBLE "::\t operators need to be copy constructible"); 
		
		std::vector<decltype(_left[0] * _right[0])> return_vec;
		return_vec.reserve(_left.size() * _right.size());
		int counter = 0;
		for (int k = 0; k < _left.size(); k++)
			for (int j = 0; j < _right.size(); j++)
				return_vec.emplace_back(_left[k] * _right[j]);
		return return_vec;
	}
};

#endif