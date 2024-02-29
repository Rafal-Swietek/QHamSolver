#pragma once

namespace QOps{

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

	/// @brief 
	/// @tparam ..._ty 
	/// @param hilbert_space1 
	/// @param hilbert_space2 
	/// @return 
	template <typename... _ty>
    template <typename _hilbert>
	inline 
	arma::sp_cx_mat 
	generic_operator<_ty...>::
		to_reduced_matrix(const _hilbert& hilbert_space, _ty... args
					)
	{
		const u64 dim = hilbert_space.get_hilbert_space_size();
		arma::sp_cx_mat matrix(dim, dim);
		auto set_matrix_elements = [&matrix, &hilbert_space](u64 k, cpx value, u64 new_idx){
			u64 idx = hilbert_space.find(new_idx);
			try {
				matrix(idx, k) += value;
			} 
			catch (const std::exception& err) {
				std::cout << "Exception:\t" << err.what() << "\n";
				std::cout << "SHit ehhh..." << std::endl;
				printSeparated(std::cout, "\t", 14, true, new_idx, idx, hilbert_space(k), value);
			}
		};
        for(u64 k = 0; k < dim; k++){            
			auto [state, val] = this->operator()(hilbert_space(k), args...);
			set_matrix_elements(k, val, state);
            // matrix(state, k) += val;
        }

		return matrix;
	}

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