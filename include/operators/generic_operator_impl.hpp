#pragma once

namespace QOps {


	// For changin to arbitrary output value
	// decltype(std::declval<out_ty&>() * std::declval<out_ty2&>())
	
	//! generic class for operators (single operator or operator products)
	template <typename... _ty>
	class generic_operator{

	protected:
		typedef _func<return_type>::input<_ty...> kernel_type;	// type of callable operator kernel

		static
		inline
		const
		kernel_type unit_kernel =
			[](u64 num, _ty... args) -> return_type
		{ return std::make_pair(num, 1.0); };

		kernel_type _kernel = unit_kernel;	// callable encoding change of quantum states and return value
		cpx opVal = 1.0;					// const return value of operator acting on given state
		void init() {
			CONSTRUCTOR_CALL;
			_extra_debug(
				std::cout << FUN_SIGNATURE << "::\n\toperator initialized with: "
					<< var_name_value(this->L, 0) 				 << "\t" 
					<< var_name_value(std::real(this->opVal), 0) << "\t" 
					<< var_name_value(std::imag(this->opVal), 0) << std::endl;
			)
		}
	public:
		unsigned int L = 0;										// length of state in given basis (Kondo has octal basis)

		//friend class operator_sum;
		//! -------------------------------------------------------------------------- CONSTRUCTORS
		generic_operator() { init(); };
		~generic_operator() { DESTRUCTOR_CALL; };

		generic_operator(int _L)
			: L(_L)
		{ init(); };
		
		generic_operator(int _L, cpx _opVal)
			: L(_L), opVal(_opVal)
		{ init(); };

		explicit generic_operator(int _L, kernel_type&& new_kernel, cpx _opVal = 1.0)
			: L(_L), _kernel(std::move(new_kernel)), opVal(_opVal)
		{ init(); };

		template <callable_type F>
		explicit generic_operator(int _L, F&& new_kernel, cpx _opVal = 1.0)
			: L(_L), _kernel(std::forward<F>(new_kernel)), opVal(_opVal)
		{ init(); };

		// copy and move
		generic_operator(const generic_operator& other) { *this = other; init(); };
		generic_operator(generic_operator&& other) noexcept { *this = std::move(other); init(); };

		// assign and move-assign constructors
		generic_operator& operator=(const generic_operator& other)
		{
			this->L = other.L;
			this->_kernel = other._kernel;
			this->opVal = other.opVal;
			return *this;
		};
		generic_operator& operator=(generic_operator&& other)
			noexcept
		{
			this->L = std::move(other.L);
			this->_kernel = std::move(other._kernel);
			this->opVal = std::move(other.opVal);
			return *this;
		}


		//! -------------------------------------------------------------------------- GETTERS & SETTERS
		//! ---------------------- operator value
		const
		auto get_operator_value()
			const { return this->opVal; };

		void set_operator_value(cpx opVal) 
			{ this->opVal = opVal; };

		//! ---------------------- operator kernel
		const
		auto get_operator_kernel()
			const { return this->_kernel; };

		void set_operator_kernel(kernel_type&& new_kernel)
			{ this->_kernel = std::move(new_kernel); };

		template <callable_type F>
		void set_operator_kernel(F&& new_kernel)
			{ this->_kernel = std::forward<F>(new_kernel); };

		//! ---------------------- assigning of new kernel
		//template <_typename F>
		//auto operator=(F&& new_kernel)
		//{ this->_kernel = std::forward<F>(new_kernel); };

		void operator=(kernel_type&& new_kernel)
			{ this->_kernel = std::move(new_kernel); };

		//! -------------------------------------------------------------------------- OPERATOR ACTING ON INPUT STATE AND VIA ACCESS
		
		// accessing return state via input

		/// @brief 
		/// @param num 
		/// @param ...args 
		/// @return 
		auto operator()(u64 num, _ty... args) const
		{
			auto [state, returnVal] = this->_kernel(num, std::forward<_ty>(args)...);
			return std::make_pair(state, opVal * returnVal);
		}

		
		/// @brief 
		/// @param args 
		/// @return 
		auto operator()(std::tuple<u64, _ty...>&& args) const
		{
			auto [state, returnVal] = std::apply(this->_kernel, args);
			return std::make_pair(state, opVal * returnVal);
		}

		//! -------------------------------------------------------------------------- ALGEBRA OF GENERIC OPERATORS
		//! ----------------------------------------------- overloaded multiplication
		//! -------------------- with other objects
		friend 
		auto operator*(cpx arg, const generic_operator<_ty...>& _operator)
			-> generic_operator<_ty...>
		{ 
			generic_operator<_ty...> new_operator(_operator);
			new_operator.opVal *= arg; 
			return std::move(new_operator); 
		}
		
		friend 
		auto operator*(const generic_operator<_ty...>& _operator, cpx arg)
			-> generic_operator<_ty...>
		{ return arg * _operator;}

		void operator*=(cpx arg)
		{ this->opVal *= arg; }
		
		//! -------------------- with another class instance
		template <typename..._ty2>
		auto operator*(const generic_operator<_ty2...>& op)
			const -> generic_operator<_ty..., _ty2...>;
		
		template <typename..._ty2>
		auto operator*=(const generic_operator<_ty2...>& op)
			{_assert_((false), 
				"Not possible operatotion, since cannot expand variadic template on (*this) at run-time. See operator%= for possible solution!");}

		//! -------------------- with another function/lambda:
		//! --	 X = generic_operator<...> * fun<...> implementation
		template <typename..._ty2>
		auto operator*(const std::function<return_type(u64, _ty2...)>& opFun)
			const -> generic_operator<_ty..., _ty2...>;

		//! --	 X = fun<...> * generic_operator<...> implementation
		template <typename..._ty2>
		friend 
		auto operator*(const kernel_type& fun,
			const generic_operator<_ty2...>& _operator)
			-> generic_operator<_ty..., _ty2...>
		{
			auto fun_result = fun * _operator._kernel;
			return generic_operator(_operator.L, std::move(fun_result), _operator.opVal);
		};

		//! ----------------------------------------------- overloaded multiplication with no template expansion (of the same type)
		//! -------------------- with another class instance
		auto operator%(const generic_operator& op)
			const -> generic_operator;

		void operator%=(const generic_operator& op);

		//! -------------------- with another function/lambda:
		//! --	 X = generic_operator<...> * fun<...> implementation
		auto operator%(const kernel_type& opFun)
			const -> generic_operator;

		//! --	 X = fun<...> * generic_operator<...> implementation
		friend
		auto operator%(const kernel_type& fun,
				const generic_operator& _operator)
			-> generic_operator
		{
			auto fun_result = fun % _operator._kernel;
			return generic_operator(_operator.L, std::move(fun_result), _operator.opVal);
		};

		//! -------------------------------------------------------------------------- ADDITIONALL OVERLOADED OPERATORS
		//auto operator^ -- exponent or indices?
		//auto operator== -- how to compare std::functions?
		//auto operator


		//! --------------------------------------------------- GETTERS OF RETURN VALUES

		/// @brief Calculate operator representative value (i.e. e^(-ik) for transaltion)
		/// @param op input operator
		/// @return operator representative value (complex)
		friend
		cpx chi(const generic_operator& op)
			{ return op.opVal; };

		/// @brief Calculate operator return value after acting on state num
		/// @param op input operator
		/// @param num input state to act on
		/// @param ...args additional arguments for operator (i.e. site acting on)
		/// @return complex value
		friend
		cpx chi(const generic_operator& op, u64 num, _ty... args)
		{
			auto [state, returnVal] = op._kernel(num, std::forward<_ty>(args)...);
			return op.opVal * returnVal;
		}

		//! -------------------------------------------------------------------------- ACTING ON QUANTUM STATES
		template <typename _ty_state>
		auto multiply(const arma::Col<_ty_state>& state, _ty... args)
			const -> arma::cx_vec
			{
				arma::cx_vec output_state(state.size(), arma::fill::zeros);
				for(u64 k = 0; k < state.size(); k++){            
					auto [idx, val] = this->operator()(k, args...);
					output_state(idx) += val * state(k);
				}

				return output_state;
			}
			
		//! -------------------------------------------------------------------------- GENERATE MATRIX ELEMENTS
		
		arma::sp_cx_mat to_matrix(u64 dim, _ty... args);
		
		template <typename _hilbert>
		arma::sp_cx_mat to_reduced_matrix(const _hilbert& hilbert_space, _ty... args);

		// template <typename _hilbert1, typename _hilbert2>
		// arma::sp_cx_mat to_matrix(const _hilbert1& hilbert_space1, const _hilbert2& hilbert_space2, _ty... args);
	};
}

//TODO: add template for output argument (change algebra of functions etc.)
//TODO: add template instantiation (or child) for generic_operators with no templates (other than output argument)
#include "generic_operator_algebra.hpp"
#include "generic_operator_matrix.hpp"