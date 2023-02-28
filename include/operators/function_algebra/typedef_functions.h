#pragma once

namespace op {
	typedef std::pair<u64, cpx> return_type;		// return type of operator, resulting state and value

	//<! std::function type-wrapper
	template <typename _return>
	struct _func {
		using return_type = _return;
		using global = std::function<_return(u64)>;
		using local = std::function<_return(u64, int)>;
		using correlator = std::function<_return(u64, int, int)>;
		
		template <typename... _types>
		using input = std::function<_return(u64, _types...)>;
	};

	using _global_fun		= _func<return_type>::global;		//<! global function acting on whole product state (most likely symmetry generator)	
	using _local_fun		= _func<return_type>::local;		//<! local function acting on single site
	using _correlation_fun	= _func<return_type>::correlator;	//<! correlation function acting on pair of sites
	using _ifun				= _func<int>::local;				//<! function returning integer value


};