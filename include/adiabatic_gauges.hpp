
namespace adiabatics{

	/// @brief 
	/// @tparam _ty 
	/// @param mat_elem 
	/// @param eigenvalues 
	/// @param L 
	/// @return 
	template <typename _ty>
	inline
	auto 
	gauge_potential(
    	const arma::Mat<_ty>& mat_elem,
    	const arma::vec& eigenvalues,
    	unsigned int L
    ) -> std::tuple<double, double, double, arma::vec> 
	{
        const size_t N = eigenvalues.size();
		const double lambda = 1 / double(N);
		const size_t mu = long(0.5 * N);

		double E_av = arma::trace(eigenvalues) / double(N);
		auto i = min_element(begin(eigenvalues), end(eigenvalues), [=](double x, double y) {
			return abs(x - E_av) < abs(y - E_av);
		});
		const long E_av_idx = i - begin(eigenvalues);
		long int E_min = E_av_idx - long(mu / 2);
		long int E_max = E_av_idx + long(mu / 2);

        double AGP = 0.0;
		double typ_susc = 0.0;
		double susc = 0.0;
		arma::vec susc_vec(N, arma::fill::zeros);
    #pragma omp parallel for reduction(+ : AGP, susc, typ_susc)
		for (long int i = 0; i < N; i++)
		{
			double susc_tmp = 0;
			for (long int j = 0; j < N && j != i; j++)
			{
				const double nominator = std::abs(mat_elem(i, j) * std::conj(mat_elem(i, j)));
				const double omega_ij = eigenvalues(j) - eigenvalues(i);
				const double denominator = omega_ij * omega_ij + lambda * lambda;
				const double value = omega_ij * omega_ij * nominator / (denominator * denominator);
				AGP 	 += value;
				susc_tmp += value;
			}
			susc_vec(i) = susc_tmp;
			if (susc_tmp > 0 && (i >= E_min && i < E_max))
			{
				typ_susc += std::log(susc_tmp);
				susc += susc_tmp;
			}
		}
        return std::make_tuple(AGP / double(N), std::exp(typ_susc / double(mu)), susc / double(mu), susc_vec);
    }

	/// @brief Calculate Adiabatic Gauge Potential at finite temperature
	/// @tparam _ty 
	/// @param mat_elem 
	/// @param eigenvalues 
	/// @param betas
	/// @param L 
	/// @return 
	template <typename _ty>
	inline
	auto 
	gauge_potential_finite_T(
    	const arma::Mat<_ty>& mat_elem,
    	const arma::vec& eigenvalues,
		const arma::vec& betas,
		const arma::vec& energy_density
    ) -> std::tuple<arma::vec, arma::vec, arma::vec, arma::vec, arma::vec, arma::vec> 
	{
        const size_t N = eigenvalues.size();
		const double lambda = 1 / double(N);
		
		arma::vec Z(betas.size(), arma::fill::zeros);
		arma::vec AGP_T(betas.size(), arma::fill::zeros);
		arma::vec AGP_T_reg(betas.size(), arma::fill::zeros);

		arma::vec count(energy_density.size()-1, arma::fill::zeros);
		arma::vec AGP_E(energy_density.size()-1, arma::fill::zeros);
		arma::vec AGP_E_typ(energy_density.size()-1, arma::fill::zeros);

		const double bandwidth = eigenvalues(N-1) - eigenvalues(0);
		const double E0 = eigenvalues(0);
		for (long int i = 0; i < N; i++)
		{
			const double Ei = eigenvalues(i);
			const double E_dens_i = (Ei - eigenvalues(0)) / bandwidth;
			
			double agp_r_tmp = 0;
			double agp_tmp = 0;
		#pragma omp parallel for reduction(+ : agp_tmp, agp_r_tmp)
			for (long int j = 0; j < i; j++)
			{
				const double Ej = eigenvalues(j);
				const double E_dens_j = (Ej - eigenvalues(0)) / bandwidth;
				const double e_mean = (E_dens_i + E_dens_j) / 2;

				const double nominator = 2 * std::abs(mat_elem(i, j) * std::conj(mat_elem(i, j)));
				const double omega_ij = Ej - Ei;
				const double denominator = omega_ij * omega_ij + lambda * lambda;

				const double _reg = omega_ij * omega_ij * nominator / (denominator * denominator);
				const double _std = nominator / (omega_ij * omega_ij);

				agp_r_tmp += _reg;
				agp_tmp += _std;

				for(int e = 0; e < energy_density.size()-1; e++)
				{
					double e_lower = energy_density(e);
					double e_upper = energy_density(e+1);
					
					if(e_mean > e_lower && e_mean <= e_upper)
					{
						count(e) += 1;
						AGP_E(e) 	 += (_reg);
						AGP_E_typ(e) += std::log(_reg);
					}
				}
			}
			for(int b = 0; b < betas.size(); b++)
			{
				double beta = betas(b);
				Z(b) += std::exp(-beta * (Ei - E0) );
				AGP_T(b) += std::exp(-beta * (Ei - E0)) * agp_tmp;
				AGP_T_reg(b) += std::exp(-beta * (Ei - E0)) * agp_r_tmp;
			}
		}
        return std::make_tuple(Z, count, AGP_T, AGP_T_reg, (AGP_E), (AGP_E_typ));
    }



};