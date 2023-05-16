#pragma once

namespace op{

    /// @brief 
    /// @tparam ..._ty 
    /// @param dim 
    /// @param ...args 
    /// @return 
    template <typename... _ty>
	inline 
	arma::sp_cx_mat 
	generic_operator<_ty...>::
		to_matrix(u64 dim, _ty... args)
	{
		arma::sp_cx_mat matrix(dim, dim);
        for(u64 k = 0; k < dim; k++){            
			auto [state, val] = this->operator()(k, args...);
            matrix(state, k) += val;
        }

		return matrix;
	}

	// /// @brief 
	// /// @tparam ..._ty 
	// /// @param hilbert_space1 
	// /// @param hilbert_space2 
	// /// @return 
	// template <typename... _ty>
    // template <typename _hilbert>
	// inline 
	// arma::sp_cx_mat 
	// generic_operator<_ty...>::
	// 	to_matrix(const _hilbert& hilbert_space, _ty... args
	// 				)
	// {
		
	// }

	// /// @brief 
	// /// @tparam _hilbert1 
	// /// @tparam _hilbert2 
	// /// @param hilbert_space1 
	// /// @param hilbert_space2 
	// /// @param ...args 
	// /// @return 
	// template <typename... _ty>
    // template <typename _hilbert1, typename _hilbert2>
	// inline 
	// arma::sp_cx_mat 
	// generic_operator<_ty...>::
	// 	to_matrix(const _hilbert1& hilbert_space1, 
	// 					const _hilbert2& hilbert_space2, _ty... args
	// 				)
    // {

    // }



};