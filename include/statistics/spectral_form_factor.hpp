#pragma once

namespace statistics{

    // /// @brief Spectral unfolding
    // /// @param eigenvalues energies to unfold
    // /// @param n order of polynomial in unfolding procedure
    // /// @return unfolded spectrum
    // [[nodiscard]]
    // inline
    // arma::vec unfolding(const arma::vec& eigenvalues, int n = 6){
    //     const size_t N = eigenvalues.size();

    //     // calculate cummulative distribution function (cdf)
    //     arma::vec cdf(eigenvalues.size(), arma::fill::zeros);
    //     std::iota(cdf.begin(), cdf.end(), 0);
        
    //     // fit polynomial order 10 to cdf
    //     auto p = arma::polyfit(eigenvalues, cdf, n);

    //     // evaluate fit at each energy: result is the unfolded energy
    //     arma::vec res = arma::polyval(p, eigenvalues);
        
    //     return res;
    // }

    enum class filters{
        raw, gauss
    };
    enum class ensemble{
        MC, GC
    };

    template <filters filter_func = filters::gauss, ensemble _ensemble = ensemble::GC>
    class SFF{
    protected:
        
        arma::vec filters;
        //<! filter options
        double eta = 0.5;           //!< gaussian filter width
        double stddev = 1.0;        //!< standard deviation of energies
        double mean = 1.0;          //!< mean energy
        double kappa = 0.0;         //!< openness of the system (dephasing)

        //<! finite temperature sff
        double inv_temperature = 0.0;   //<! inverse temperature
        double energy_density = 0.0;    //<! target energy density

        //<! store normalizations
        double Z = 0;               //<! normalization of unconnected sff
        double A = 0;               //<! 1st normalization of connected sff
        double B = 0;               //<! 2nd normalization of connected sff


        bool cut_edges = false;     //<! cut spectral edges (5-10 points) due to unfolding?
        
        //<! helper functions
        void get_mean(const arma::vec& energies);

        double _filter_(double E, double E0);
        void _set_filters_(const arma::vec& energies);

        cpx filtered(const arma::vec& energies, double time);
        cpx raw(const arma::vec& energies, double time);

    public:    

        /// @brief Constructor for spectral form factor calculations
        /// @param _eta filter wifth
        /// @param _beta inverse temperature
        /// @param _cut_edges cut spectral edges (5-10 points) due to unfolding?
        explicit SFF(double _eta, double _beta = 0.0, double _eps = 0.5, double _kappa = 0.0, bool _cut_edges = false)
            : eta(_eta), inv_temperature(_beta), energy_density(_eps), kappa(_kappa), cut_edges(_cut_edges)
            {};

        auto calculate(const arma::vec& E, const arma::vec& times) ->  arma::cx_vec;
        auto get_norms() { return std::make_tuple(this->Z, this->A, this->B); }

    };
}

#include "spectral_form_factor_aux.hpp"