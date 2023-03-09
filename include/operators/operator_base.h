#ifndef _OPERATOR_CLASS
#define _OPERATOR_CLASS

//#include "operator_config.h"
#include "function_algebra/funAlgebra.h"
#include "compiler_setup/compiler_setup_operators.h"
#include "builtin_operators.h"
//! ---------------------------------------------------------------------------------------------------------------------------------------------------------
//! ------------------------------------------------------------------OPERATOR CLASS-------------------------------------------------------------------------
namespace op {
	//! ------------------------------------------------------------------------------------------------
	//! generic class for operators (single operator or operator products)
	template <typename... _ty>
	class generic_operator;

	//! ------------------------------------------------------------------------------------------------
	//! operator class with set indices (only input is quantum state)
	template <typename... _ty>
	class operator_const_index;

	//<! special class of correlation operator (act on 2 sites)
	class correlation_operator;

	//<! special class of local generic_operator constructed from input array (operation on digits)
	class local_operator;

	//<! class containing sums of operators of any type
	class operator_sum;
};

#include "generic_operator_impl.hpp"
//#include "generic_operator_const_index_impl.h"
//#include "local_operators.h"
//#include "symmetries.hpp"

#include "compiler_setup/typedef_operators.h"
#endif