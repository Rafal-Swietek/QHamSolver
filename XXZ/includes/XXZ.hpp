#pragma once

#ifndef _XXZ
#define _XXZ

#include "../../include/hilbert_space/u1.hpp"
using U1Hilbert = U1_hilbert_space<U1::spin>;

/// @brief Base class for XXZ model 
/// @tparam _ty element type of matrix
/// @tparam hilbert hilbert space (full or symmetric)
template <  typename _ty, 
            class hilbert
            >
class XXZbase : 
    public hamiltonian_base<_ty, hilbert>
{
protected:
    double _hz = 0.5;                       // uniform longitudinal field
    double _hx = 0.5;                       // uniform transverse field
    double _J1 = 1.0;                       // nearest neighbour coupling amplitude
    double _J2 = 0.0;                       // next-nearest neighbour coupling amplitude
    double _delta1 = 0.55;                  // nearest neighbour interaction amplitude
    double _delta2 = 0.0;                   // next-nearest neighbour interaction amplitude
    double _eta1 = 0.55;                    // nearest neighbour anisotropy in XY
    double _eta2 = 0.0;                     // next-nearest neighbour anisotropy in XY

    bool _add_edge_fields = false;          // add additional edge fields needed for SUSY in OBC

public:

    virtual arma::SpMat<_ty> create_supercharge() = 0;
};
//! ^^^ might not be necesarry ^^^


/// @brief Fully anisotropic spin chain (XXZ)
class XXZ : 
    public hamiltonian_base<double, U1Hilbert>
{
    //<! ----------------------------------------------------- INHERIT TYPEDEFs FROM BASE
    typedef typename hamiltonian_base<double, U1Hilbert>::matrix        matrix;
    typedef typename hamiltonian_base<double, U1Hilbert>::sparse_matrix sparse_matrix;

    //<! ----------------------------------------------------- MODEL PARAMETERS
private:
    disorder<double> disorder_generator;    // generator for random disorder and couplings
    
    arma::vec _disorder;                    // disorder array on Z field
    
    double _w = 0.5;                        // disorder value on top of uniform field
    double _hz = 0.5;                       // uniform longitudinal field
    double _J1 = 1.0;                       // nearest neighbour coupling amplitude
    double _J2 = 0.0;                       // next-nearest neighbour coupling amplitude
    double _delta1 = 0.55;                  // nearest neighbour interaction amplitude
    double _delta2 = 0.0;                   // next-nearest neighbour interaction amplitude

    float Sz = 0.0;                         // magnetization sector

    u64 _seed = std::random_device{}();     // seed for random generator
    
    bool _add_parity_breaking = 0;          // add parity breaking term on edge
    bool _use_disorder = 0;                 // use disordered XXZ model
    // bool _add_edge_fields = false;          // add additional edge fields needed for SUSY in OBC

    //<! ----------------------------------------------------- INITIALIZE MODEL
    virtual void init() override
    {   
        // initialize hilbert space
        this->_hilbert_space = U1Hilbert(this->system_size, this->Sz);
        this->dim = this->_hilbert_space.get_hilbert_space_size();

        // initialize disorder
        disorder_generator = disorder<double>(this->_seed);
        this->_disorder = arma::vec(this->system_size, arma::fill::zeros);

        // create hamiltonian
        this->create_hamiltonian();
        // std::cout << "Mapping:\n" << this->_hilbert_space.get_mapping() << std::endl;
        // std::cout << "Hamiltonian:\n" << arma::mat(this->H) << std::endl;
    }

public:
    //<! ----------------------------------------------------- CONSTRUCTORS
    XXZ() = default;
    XXZ(std::istream& os);
    XXZ(int _BC, unsigned int L, double J1, double J2, double delta1, double delta2, double hz, float Sz = 0.0, 
                bool add_parity_breaking = false, double w = 0, const u64 seed = std::random_device{}());

    //<! ----------------------------------------------------- HAMILTONIAN BUILDERS
    virtual void create_hamiltonian() override;
    virtual sparse_matrix create_local_hamiltonian(int site) override;
    virtual void set_hamiltonian_elements(u64 k, double value, u64 new_idx) override;

    //<! ----------------------------------------------------- OVERRIDEN OPERATORS
    virtual std::ostream& write(std::ostream&) const override;
    virtual std::istream& read(std::istream&) override;
};

#endif