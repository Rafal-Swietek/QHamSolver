#pragma once

namespace lanczos {
	//TODO: make separately from the class as standalone and do struct within class?
	//! ------------------------------------------------------ from: KRYLOV -> to: HILBERT
	//<! conversion of input state to original Hilbert space
	template <typename _ty>
	inline
	auto Lanczos<_ty>::conv_to_hilbert_space(
			const arma::Col<_ty>& state_lanczos	//<! state to transform
		) const -> arma::Col<_ty>
	{

		assert(state_lanczos.size() == this->lanczos_steps
			&& "Wrong state dimensions! Required dim is the number of lanczos steps "
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

	//<! conversion of eigenstate (by index) to original Hilbert space
	template <typename _ty>
	inline
	auto
	Lanczos<_ty>::conv_to_hilbert_space(
			int state_id						//<! index of state to transform
		) const -> arma::Col<_ty>
	{
		assert(!this->H_lanczos.is_empty() && "Diagonalize!!");
		return this->conv_to_hilbert_space(this->eigenvectors.col(state_id));
	}

	//! ------------------------------------------------------ from: HILBERT -> to: KRYLOV
	template <typename _ty>
	inline
	auto Lanczos<_ty>::conv_to_krylov_space(
		const arma::Col<_ty>& input		//<! state to transform from hilbert to krylov
	) const -> arma::Col<_ty>
	{

		assert(input.size() == this->N
			&& "Wrong state dimensions! Required dim is the original hilbert space size "
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