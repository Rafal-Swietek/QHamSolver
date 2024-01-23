#pragma once

namespace lanczos{

	/// @brief Find time evolved state at time t with step dt
	/// @tparam _ty template type for state type
	/// @param _state 
	/// @param time 
	/// @param dt 
	/// @param M 
	/// @return 
	template <typename _ty, converge converge_type>
	inline
	void Lanczos<_ty, converge_type>::time_evolution(
		arma::Col<_ty>& _state,
		double time,
		double dt
	)
	{
		for(double t = 0; t < time; t += dt)
			this->time_evolution_step(_state, dt);
	}

	/// @brief 
	/// @tparam _ty 
	/// @tparam converge_type 
	/// @param _state 
	/// @param dt 
	template <typename _ty, converge converge_type>
	inline
	void Lanczos<_ty, converge_type>::time_evolution_step(
		arma::Col<_ty>& _state,
		double dt
	)
	{
		this->use_full_convergence = false;
		this->diagonalization(_state);
		arma::cx_vec state_in_krylov = this->conv_to_krylov_space(_state);
		arma::cx_vec evolved_state(state_in_krylov.size(), arma::fill::zeros);
		for(int l = 0; l < this->lanczos_steps; l++){
			cpx overlap = dot_prod(this->eigenvectors.col(l), state_in_krylov);
			evolved_state += std::exp(-1i * this->eigenvalues(l) * dt) * overlap * arma::normalise(this->eigenvectors.col(l));
		}
		_state = arma::normalise(this->conv_to_hilbert_space(evolved_state));
		
		// arma::cx_vec evolved_state(_state.size(), arma::fill::zeros);
		// for(int l = 0; l < this->lanczos_steps; l++){
		// 	arma::Col<_ty> psi_l = this->conv_to_hilbert_space(l);
		// 	cpx overlap = arma::cdot(psi_l, _state);
		// 	evolved_state += std::exp(-1i * this->eigenvalues(l) * dt) * overlap * arma::normalise(psi_l);
		// }
		// _state = arma::normalise(evolved_state);
	}
}