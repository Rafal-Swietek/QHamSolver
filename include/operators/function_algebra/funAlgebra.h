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
	template <typename return_ty, typename... _Type1, typename... _Type2 >
	inline auto operator*(
		typename _func< return_ty >::template input< _Type1... > f,
		typename _func< return_ty >::template input< _Type2... > g
		)
	{
		return [f, g](u64 num,
			_Type1... args1,
			_Type2... args2) -> return_ty
		{
			auto [state, val] = g(num, args2...);
			auto [state_final, ret_final] = f(state, args1...);
			return std::make_pair(state_final, val * ret_final);
		};
	};

	// without template expansion
	template <typename return_ty, typename... _Type>
	inline auto operator%(
		typename _func< return_ty >::template input< _Type... > f,
		typename _func< return_ty >::template input< _Type... > g
		)
	{
		return [f, g](u64 num,
			_Type... args) -> return_ty
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

template <typename ReturnTy, typename state_ty, typename... Args>
struct Kernel {
    std::function<ReturnTy(state_ty, Args...)> fn;

    Kernel() = default;
    Kernel(std::function<ReturnTy(state_ty, Args...)> f) : fn(std::move(f)) {}

    ReturnTy operator()(state_ty n, Args... args) const {
        return fn(n, args...);
    }

    // Now define the operator% for this custom type
    friend Kernel operator%(const Kernel& f, const Kernel& g) {
        return Kernel{
            [f, g](state_ty n, Args... args) -> ReturnTy {
                auto [state, val] = g(n, args...);
                auto [state_final, ret_final] = f(state, args...);
                return { state_final, val * ret_final };
            }
        };
    }

	template <typename..._ty>
	friend Kernel operator*(const Kernel<ReturnTy, state_ty, Args...>& f, const Kernel<ReturnTy, state_ty, _ty...>& g) {
        return Kernel{
            [f, g](state_ty num,
				Args... args1,
				_ty... args2) -> ReturnTy
			{
				auto [state, val] = g(num, args2...);
				auto [state_final, ret_final] = f(state, args1...);
				return std::make_pair(state_final, val * ret_final);
			}
        };
    }
};

#endif