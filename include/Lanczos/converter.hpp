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
	auto Lanczos<_ty, converge_type>::conv_to(const arma::Col<_ty>& state_in) const -> arma::Col<_ty>
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
	/// @param state_lanczos  state in Krylov basis to transform
	/// @return state in Hilbert basis
	template <typename _ty, converge converge_type>
	inline
	auto Lanczos<_ty, converge_type>::conv_to_hilbert_space(
			const arma::Col<_ty>& state_lanczos	//<! state to transform
		) const -> arma::Col<_ty>
	{

		_assert_(state_lanczos.size() == this->lanczos_steps,
			"Wrong state dimensions! Required dim is the number of lanczos steps "
		);
		arma::Col<_ty> state(this->N, arma::fill::zeros);	//<! output state

		if (this->use_krylov){
			state = this->krylov_space * state_lanczos;
		}
		else 
		{
			arma::Col<_ty> fi_next(this->N, arma::fill::zeros);
			//if (this->mymemory_over_performance)
			//	this->model->hamil_vec_onthefly(rand, fi_next);
			//else
			fi_next = this->H * this->initial_random_vec;

			_ty alfa = cdot(this->initial_random_vec, fi_next);
			fi_next = fi_next - alfa * this->initial_random_vec;
			arma::Col<_ty> fi_prev = this->initial_random_vec;
			for (int j = 1; j < this->lanczos_steps; j++) {
				_ty beta = arma::norm(fi_next);
				arma::Col<_ty> fi = fi_next / beta;

				state += state_lanczos(j) * fi;

				//if (this->mymemory_over_performance)
				//	this->hamil_vec_onthefly(fi, fi_nextdot+
				//else
				fi_next = this->H * fi;

				alfa = arma::cdot(fi, fi_next);
				fi_next = fi_next - alfa * fi - beta * fi_prev;

				fi_prev = fi;
			}
		}
		return arma::normalise(state);
	};
	// inline
	// auto Lanczos::conv_to_hilbert_space(
	// 		const arma::vec& state_lanczos	//<! state to transform
	// 	) const -> arma::Col<_ty>
	// {
	// 	auto state = cpx_real_vec(state_lanczos);
	// 	return this->conv_to_hilbert_space(state);
	// }

	
	/// @brief conversion of eigenstate (by index) to original Hilbert space
	/// @tparam _ty template for element type
	/// @param state_id index of eigenstate to transform
	/// @return transformed eigenstate to Hilbert basis
	template <typename _ty, converge converge_type>
	inline
	auto
	Lanczos<_ty, converge_type>::conv_to_hilbert_space(
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
	auto Lanczos<_ty, converge_type>::conv_to_krylov_space(
		const arma::Col<_ty>& input		//<! state to transform from hilbert to krylov
	) const -> arma::Col<_ty>
	{

		_assert_(input.size() == this->N,
			"Wrong state dimensions! Required dim is the original hilbert space size "
		);

		arma::Col<_ty> transformed_input(
			lanczos_steps,
			arma::fill::zeros
			);

		if (this->use_krylov){
			transformed_input = this->krylov_space.t() * input;
		}
		else 
		{
			transformed_input(0) = arma::cdot(this->initial_random_vec, input); // =1

			arma::Col<_ty> fi_next = H * this->initial_random_vec;
			arma::Col<_ty> fi_prev = this->initial_random_vec;

			_ty alfa = arma::cdot(this->initial_random_vec, fi_next);
			fi_next = fi_next - alfa * this->initial_random_vec;

			//<! lanczos procedure
			for (int j = 1; j < lanczos_steps; j++) {
				_ty beta = arma::norm(fi_next);
				arma::Col<_ty> fi = fi_next / beta;
				transformed_input(j) = arma::cdot(fi, input);
				fi_next = H * fi;

				alfa = arma::cdot(fi, fi_next);
				fi_next = fi_next - alfa * fi - beta * fi_prev;
				fi_prev = fi;
			}
		}
		return arma::normalise(transformed_input);
	}
	// inline
	// auto Lanczos::conv_to_krylov_space(
	// 		const arma::vec& input	//<! state to transform
	// 	) const -> arma::Col<_ty>
	// {
	// 	auto state = cpx_real_vec(input);
	// 	return this->conv_to_krylov_space(state);
	// }
}