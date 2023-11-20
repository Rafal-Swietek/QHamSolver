#pragma once

namespace lanczos {

	/// @brief conversion of input state to templated basis
	/// @tparam _ty template for element type
	/// @tparam base basis type of input state
	/// @param state_in state to transform
	/// @return transformed state in chosen basis
	template <typename _ty, converge converge_type>
	template <base_type base>
	inline
	auto BlockLanczos<_ty, converge_type>::conv_to(const arma::Col<_ty>& state_in) const -> arma::Col<_ty>
	{
		if constexpr (base == base_type::hilbert)
			return this->conv_to_hilbert_space(state_in);
		else if constexpr (base == base_type::krylov)
			return this->conv_to_krylov_space(state_in);
		else{
			static_check((base == base_type::hilbert) || (base == base_type::krylov), "Not implemented other Basis type. Choose among: hilbert, krylov.");
		}
	};

	//! ------------------------------------------------------ from: KRYLOV -> to: HILBERT

	/// @brief conversion of input state to original Hilbert space
	/// @tparam _ty template for element type
	/// @param state_id index of state to transform
	/// @return transformed state in Hilbert space
	template <typename _ty, converge converge_type>
	inline
	auto BlockLanczos<_ty, converge_type>::conv_to_hilbert_space(
			const arma::Col<_ty>& state_lanczos	//<! state to transform
		) const -> arma::Col<_ty>
	{

		_assert_(state_lanczos.size() == this->matrix_size,
			"Wrong state dimensions! Required dim is the number of (lanczos steps) x (size of bundle) "
		);
		arma::Col<_ty> state(this->N, arma::fill::zeros);	//<! output state

		if (this->use_krylov){
			state = this->krylov_space * state_lanczos;
		}
		else 
		{
			_assert_(this->use_krylov, "Note implemented generating states without krylov subspace.");
		}
		return state;
	};

	/// @brief conversion of eigenstate (by index) to original Hilbert space
	/// @tparam _ty template for element type
	/// @param state_id index of state to transform
	/// @return transformed state in Hilbert space
	template <typename _ty, converge converge_type>
	inline
	auto
	BlockLanczos<_ty, converge_type>::conv_to_hilbert_space(
			int state_id						//<! index of state to transform
		) const -> arma::Col<_ty>
	{
		_assert_(!this->H_lanczos.is_empty(), "Diagonalize!!");
		return this->conv_to_hilbert_space(this->eigenvectors.col(state_id));
	}

	//! ------------------------------------------------------ from: HILBERT -> to: KRYLOV

	/// @brief conversion of eigenstate to Krylov basis
	/// @tparam _ty template for element type
	/// @param input state to transform
	/// @return transformed state in Krylov basis
	template <typename _ty, converge converge_type>
	inline
	auto BlockLanczos<_ty, converge_type>::conv_to_krylov_space(
		const arma::Col<_ty>& input		//<! state to transform from hilbert to krylov
	) const -> arma::Col<_ty>
	{

		_assert_(input.size() == this->N,
			 "Wrong state dimensions! Required dim is the original hilbert space size "
		);

		arma::Col<_ty> transformed_input(
			this->matrix_size,
			arma::fill::zeros
			);

		if (this->use_krylov){
			transformed_input = this->krylov_space.t() * input;
		}
		else 
		{
			_assert_(this->use_krylov, "Note implemented generating states without krylov subspace.");
		}
		return transformed_input;
	}
}