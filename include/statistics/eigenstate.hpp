#pragma once

namespace statistics{

    //! ---------------------------------------------------------------- IPR
    //<! calculate participation ratio of input state for any q (q=1 is the typically used one)
    // template <typename _type> 
    // [[nodiscard]]
    // inline
    // double participation_ratio_plane_wave(
    //     const arma::Col<_type>& _state,   //<! input state
    //     double q = 1.0
    //     ) {
    //     double pr = 0;
    //     const size_t N = _state.size();
    // #pragma omp parallel for reduction(+: pr)
    // 	for (int n = 0; n < N; n++) {
    //         cpx overlap = 0;
    //         double k = two_pi * n / double(N);
    //         for(int l = 0; l < N; l++)
    //             overlap += std::exp(1i * k * double(l)) * _state(l);
    // 		double value = abs(conj(overlap) * overlap);
    // 		pr += std::pow(value, q);
    // 	}
    // 	return pr  / std::pow(double(N), q);
    // }

    //<! calculate participation ratio of input state for any q (q=1 is the typically used one)
    template <typename _type> 
    [[nodiscard]]
    inline
    double participation_ratio(
        const arma::Col<_type>& _state,   //<! input state
        double q = 1.0
        ) {
        double pr = 0;
        const size_t N = _state.size();
        // if(q == 1)
        // {
        // #pragma omp parallel for reduction(+: pr)
        //     for (int n = 0; n < N; n++) {
        //         double value = std::abs(std::conj(_state(n)) * _state(n));
        //         pr += (std::abs(value) > 0) ? -value * std::log(value) : 0;
        //     }
        // } else 
        {
        #pragma omp parallel for reduction(+: pr)
            for (int n = 0; n < N; n++) {
                double value = std::abs(std::conj(_state(n)) * _state(n));
                pr += std::pow(value, q);
            }
        }
        return pr;
    }

    //<! calculate inverse participation ratio of input state
    template <typename _type> 
    [[nodiscard]]
    inline
    double inverse_participation_ratio(
        const arma::Col<_type>& _state   //<! input state
        ) {
        double ipr = 0;
        const size_t N = _state.size();
    #pragma omp parallel for reduction(+: ipr)
        for (int n = 0; n < N; n++) {
            double value = abs(conj(_state(n)) * _state(n));
            ipr += value * value;
        }
        return 1.0 / ipr;
    }


    //! ---------------------------------------------------------------- INFORMATION ENTROPY
    //<! calculate information entropy of input state in computational basis (full)
    template <typename _type> 
    [[nodiscard]]
    inline
    double information_entropy(
        const arma::Col<_type>& _state   //<! inout state
        ) {
        double ent = 0;
        const size_t N = _state.size();
    #pragma omp parallel for reduction(+: ent)
        for (int k = 0; k < N; k++) {
            double val = abs(conj(_state(k)) * _state(k));
            ent += val * log(val);
        }
        return -ent / log(0.48 * N);
    }

    //<! calculate information entropy of input state in another eigenbasis set within range
    template <typename _type>
    [[nodiscard]]
    inline 
    double information_entropy(
        const arma::Col<_type>& _state,     //<! inout state
        const arma::Mat<_type>& new_basis,  //<! new eigenbasis to find overlap with _state
        u64 _min,                           //<! first state in new basis
        u64 _max                            //<! last eigenstate in new basis
        ) {
        const size_t N = _state.size();
        double ent = 0;
    #pragma omp parallel for reduction(+: ent)
        for (long k = (long)_min; k < (long)_max; k++) 
        {
            cpx c_k = cdot(new_basis.col(k), _state);
            double val = abs(conj(c_k) * c_k);
            ent += val * log(val);
        }
        return -ent / log(0.48 * N);
    }
    //<! same but without range (taken full space)
    template <typename _type>
    [[nodiscard]]
    inline 
    double information_entropy(
        const arma::Col<_type>& _state,     //<! inout state
        const arma::Mat<_type>& new_basis  //<! new eigenbasis to find overlap with _state
        ) 
        { return information_entropy(_state, new_basis, 0, new_basis.n_cols); }


};