#pragma once

namespace statistics{

    /// @brief Find mean energy for given temperature (target energy)
    /// @tparam filter_func template -> which filtering?
    /// @tparam _ensemble template -> use MC or GC ensemble?
    /// @param energies input energies
    template <filters filter_func, ensemble _ensemble>
    inline
    void SFF<filter_func, _ensemble>::get_mean(const arma::vec& E)
    {
        double partition_fun = 0;
        if constexpr (_ensemble == ensemble::GC) 
        {
            this->mean = 0.0;
            for(long n = 0; n < E.size(); n++){
                double Z_E = std::exp(-this->inv_temperature * ( E(n) - E(0) ) );
                this->mean += E(n) * Z_E;
                partition_fun += Z_E;
            }
            this->mean = this->mean / partition_fun;
        } else{
            this->mean = E(0) + this->energy_density * ( E(E.size()-1) - E(0) );
        }
        // printSeparated(std::cout, "\t", 20, true, this->mean, partition_fun, this->energy_density, this->inv_temperature);
    }

    /// @brief Find mean energy for given temperature (target energy)
    /// @tparam filter_func template -> which filtering?
    /// @tparam _ensemble template -> use MC or GC ensemble?
    /// @param energies input energies
    template <filters filter_func, ensemble _ensemble>
    inline
    double SFF<filter_func, _ensemble>::_filter_(double E, double E0)
    {
        if constexpr (filter_func == filters::gauss)
            return std::exp( -(E - this->mean) * (E - this->mean) 
                            / (2.0 * this->eta * this->eta * this->stddev * this->stddev ) );
        else
            return std::exp(-this->inv_temperature * (E - E0));
    }

    /// @brief Find mean energy for given temperature (target energy)
    /// @tparam filter_func template -> which filtering?
    /// @tparam _ensemble template -> use MC or GC ensemble?
    /// @param energies input energies
    template <filters filter_func, ensemble _ensemble>
    inline
    void SFF<filter_func, _ensemble>::_set_filters_(const arma::vec& energies)
    {
        this->filters = arma::vec(energies.size(), arma::fill::zeros);
        for (long n = 0; n < energies.size(); n++) 
            this->filters(n) = this->_filter_(energies(n), energies(0));
    }

    /// @brief Calculate raw (unfiltered) spectral form factor at time step t
    /// @tparam filter_func template -> which filtering?
    /// @tparam _ensemble template -> use MC or GC ensemble?
    /// @param energies input energies
    /// @param t time point
    /// @return sff at time point
    template <filters filter_func, ensemble _ensemble>
    inline
    cpx SFF<filter_func, _ensemble>::raw(const arma::vec& energies, double t)
    {
        this->Z = 0; 
        this->A = 1.0;  this->B = 1.0;      //<! is normalized well with no filter
        double sff_re = 0, sff_im = 0;
        for (long n = 0; n < energies.size(); n++) {
            const double filter = this->filters(n);
            this->Z += filter * filter;
            cpx _sff_ = filter * std::exp(- 1.0i * t * energies(n));
            sff_re += std::real(_sff_);
            sff_im += std::imag(_sff_);
        }
        return cpx(sff_re, sff_im);
    }


    /// @brief Calculate filtered spectral form factor at time step t
    /// @tparam filter_func template -> which filtering?
    /// @tparam _ensemble template -> use MC or GC ensemble?
    /// @param energies input energies
    /// @param t time point
    /// @return sff at time point
    template <filters filter_func, ensemble _ensemble>
    inline
    cpx SFF<filter_func, _ensemble>::filtered(const arma::vec& energies, double t)
    {
        const double denom = 2.0 * this->eta * this->eta * this->stddev * this->stddev;
        
        this->Z = 0; this->A = 0;  this->B = 0;
        double sff_re = 0, sff_im = 0;
        for (long n = 0; n < energies.size(); n++) {
            // const double filter = this->_filter_(energies(n));
            const double filter = this->filters(n);
            this->Z += std::abs(filter * filter);
            this->B += filter;
            cpx _sff_ = filter * std::exp(-1.0i * two_pi * energies(n) * t);
            sff_re += std::real(_sff_);
            sff_im += std::imag(_sff_);
        }
        this->A = std::abs(this->B * this->B);
        return cpx(sff_re, sff_im);
    }


    /// @brief Calculate spectral form factor at time step t
    /// @tparam filter_func template -> which filtering?
    /// @tparam _ensemble template -> use MC or GC ensemble?
    /// @param energies input energies
    /// @param t time point
    /// @return sff at time point
    template <filters filter_func, ensemble _ensemble>
    inline
    arma::cx_vec SFF<filter_func, _ensemble>::calculate(const arma::vec& energies, const arma::vec& times)
    {
        arma::cx_vec sff(times.size(), arma::fill::zeros);
        
        this->get_mean(energies);
        this->stddev = arma::stddev(energies);

        this->_set_filters_(energies);
    #pragma omp parallel for
        for(int k = 0; k < times.size(); k++){
            if constexpr (filter_func == filters::raw)
                sff(k) = this->raw(energies, times(k) );
            else
                sff(k) = this->filtered(energies, times(k) );
        }
        return sff;
    }

}