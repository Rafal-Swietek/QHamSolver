#pragma once
//! Header file to define algebra of operators<_ty...> class. Overloaded multiplication operators, addition and substraction.
//!  - uses the general funAlgebra.h to mimic algebra on std::function
//!  - 
namespace op {

	//! ------------------------------------------------------------------------------------------------ MULTIPLICATION
	//! ----------------------------------------------------------- with another class instance
	//! --	 X = operators<...> * operators<...> implementation
	template <typename... _ty> 
	template <typename... _ty2>
	inline 
	auto generic_operator<_ty...>::
		operator*(const generic_operator<_ty2...>& _operator)
		const
		-> generic_operator<_ty..., _ty2...>
	{
		assert_hilbert_space(_operator);
		auto new_kernel = this->_kernel * _operator._kernel;	// new operator kernel
		return generic_operator<_ty..., _ty2...>
			(
				this->L,
				std::move(new_kernel),
				this->opVal * _operator.opVal
			);
	}

	//! ----------------------------------------------------------- with another function 
	//! --	 X = operators<...> * fun<...> implementation
	template <typename... _ty>
	template <typename... _ty2>
	inline 
	auto generic_operator<_ty...>::
		operator*(const std::function<return_type(u64, _ty2...)>& opFun) 
		const
		-> generic_operator<_ty..., _ty2...>
	{
		return generic_operator<_ty..., _ty2...>(
				this->L, std::move(this->_kernel * opFun), this->opVal);
	}

	//! --	 X = fun<...> * operators<...> implementation
	// in operator_class.h
	
	//! ------------------------------------------------------------------------------------------------ MULTIPLICATION WITNO NO TEMPLATE EXPANSION
	//! ----------------------------------------------------------- with another class instance
	//! --	 X = operators<...> * operators<...> implementation
	template <typename... _ty>
	inline 
	auto generic_operator<_ty...>::
		operator%(const generic_operator<_ty...>& _operator)
		const
		-> generic_operator<_ty...>
	{
		assert_hilbert_space(_operator);
		auto new_kernel = this->_kernel % _operator._kernel;	// new operator kernel
		return generic_operator<_ty...>
			(
				this->L,
				std::move(new_kernel),
				this->opVal * _operator.opVal
				);
	}

	//! --	 X *= operators<...> implementation
	template <typename... _ty>
	inline 
	auto generic_operator<_ty...>::
		operator%=(const generic_operator<_ty...>& _operator)
	{
		assert_hilbert_space(_operator);
		this->_kernel = this->_kernel % _operator._kernel;
		this->opVal = this->opVal * _operator.opVal;
	}

	//! ----------------------------------------------------------- with another function 
	//! --	 X = operators<...> * fun<...> implementation
	template <typename... _ty>
	inline 
	auto generic_operator<_ty...>::
		operator%(const std::function<return_type(u64, _ty...)>& opFun)
		const
		-> generic_operator<_ty...>
	{
		return generic_operator<_ty...>(
			this->L, std::move(this->_kernel % opFun), this->opVal);
	}

	//! --	 X = fun<...> % operators<...> implementation
	// in operator_class.h
	

};