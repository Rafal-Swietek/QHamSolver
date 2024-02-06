#pragma once

namespace statistics{

    /// @brief Find mean energy for given temperature (target energy)
    /// @tparam filter_func template bool -> use filtering?
    /// @param energies input energies
    template <filters filter_func>
    inline
    void SFF<filter_func>::get_mean(const arma::vec& E)
    {
        this->mean = 0.0;
        double partition_fun = 0;
    #pragma omp parallel
        for(long n; n < E.size(); n++){
            double Z_E = std::exp(-this->beta * E(n));
            this->mean += E(n) * Z_E;
            partition_fun += Z_E;
        }
        this->mean /= partition_fun;
    }

    /// @brief Find mean energy for given temperature (target energy)
    /// @tparam filter_func template bool -> use filtering?
    /// @param energies input energies
    template <filters filter_func>
    inline
    double SFF<filter_func>::_filter_(double E)
    {
        if constexpr (filter_func == filters::gauss)
            return std::exp( -(E - this->mean) * (E - this->mean) 
                            / (2.0 * this->eta * this->eta * this->stddev * this->stddev ) );
        else
            return std::exp(-this->beta * E);
    }
    /// @brief Calculate raw (unfiltered) spectral form factor at time step t
    /// @tparam filter_func template bool -> use filtering?
    /// @param energies input energies
    /// @param t time point
    /// @return sff at time point
    template <filters filter_func>
    inline
    cpx SFF<filter_func>::raw(const arma::vec& energies, double t)
    {
        cpx sff = 0.0;
        this->Z = 0; 
        this->A = 1.0;  this->B = 1.0;      //<! is normalized well with no filter
        for (long n = 0; n < energies.size(); n++) {
            this->Z += std::exp(- this->beta * energies(n));
            sff += std::exp(- (this->beta + 1.0i * t) * energies(n));
        }
        return sff;
    }


    /// @brief Calculate filtered spectral form factor at time step t
    /// @tparam filter_func template bool -> use filtering?
    /// @param energies input energies
    /// @param t time point
    /// @return sff at time point
    template <filters filter_func>
    inline
    cpx SFF<filter_func>::filtered(const arma::vec& energies, double t)
    {
        this->get_mean(energies);
        this->stddev = arma::stddev(energies);
        const double denom = 2.0 * this->eta * this->eta * this->stddev * this->stddev;
        cpx sff = 0.0;
        this->Z = 0; this->A = 0;  this->B = 0;
        for (long n = 0; n < energies.size(); n++) {
            const double filter = this->_filter_(energies(n));
            this->Z += std::abs(filter * filter);
            this->B += filter;
            sff += filter * std::exp(-1.0i * two_pi * energies(n) * t);
        }
        this->A = std::abs(this->B * this->B);
        return sff;
    }


    /// @brief Calculate spectral form factor at time step t
    /// @tparam filter_func template bool -> use filtering?
    /// @param energies input energies
    /// @param t time point
    /// @return sff at time point
    template <filters filter_func>
    inline
    arma::cx_vec SFF<filter_func>::calculate(const arma::vec& energies, const arma::vec& times)
    {
        arma::cx_vec sff(times.size(), arma::fill::zeros);
    #pragma omp parallel
        for(int k = 0; k < times.size(); k++){
            if constexpr (filter_func == filters::raw)
                sff(k) = this->raw(energies, times(k) );
            else
                sff(k) = this->filtered(energies, times(k) );
        }
        return sff;
    }

}