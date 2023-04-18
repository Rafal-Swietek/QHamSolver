#pragma once

#ifndef _XYZ
#define _XYZ

/// @brief Fully anisotropic spin chain (XYZ)
class XYZ : 
    public hamiltonian_base<double, full_hilbert_space>
{
    //<! ----------------------------------------------------- INHERIT TYPEDEFs FROM BASE
    typedef typename hamiltonian_base<double, full_hilbert_space>::matrix        matrix;
    typedef typename hamiltonian_base<double, full_hilbert_space>::sparse_matrix sparse_matrix;

    //<! ----------------------------------------------------- MODEL PARAMETERS
private:
    disorder<double> disorder_generator;    // generator for random disorder and couplings
    
    arma::vec _disorder;                    // disorder array on Z field
    
    double _w = 0.5;                        // disorder value on top of uniform field
    double _hz = 0.5;                       // uniform longitudinal field
    double _hx = 0.5;                       // uniform transverse field
    double _J1 = 1.0;                       // nearest neighbour coupling amplitude
    double _J2 = 0.0;                       // next-nearest neighbour coupling amplitude
    double _delta1 = 0.55;                  // nearest neighbour interaction amplitude
    double _delta2 = 0.0;                   // next-nearest neighbour interaction amplitude
    double _eta1 = 0.55;                    // nearest neighbour anisotropy in XY
    double _eta2 = 0.0;                     // next-nearest neighbour anisotropy in XY

    u64 _seed = std::random_device{}();     // seed for random generator
    
    bool _add_parity_breaking = 0;          // add parity breaking term on edge
    bool _use_disorder = 0;                 // use disordered XYZ model
    bool _add_edge_fields = false;          // add additional edge fields needed for SUSY in OBC

    //<! ----------------------------------------------------- INITIALIZE MODEL
    virtual void init() override
    {   
        // initialize hilbert space
        this->_hilbert_space = full_hilbert_space(this->system_size);
        this->dim = this->_hilbert_space.get_hilbert_space_size();

        // initialize disorder
        disorder_generator = disorder<double>(this->_seed);
        this->_disorder = arma::vec(this->system_size, arma::fill::zeros);

        // create hamiltonian
        this->create_hamiltonian();
    }

public:
    //<! ----------------------------------------------------- CONSTRUCTORS
    XYZ() = default;
    XYZ(std::istream& os);
    XYZ(int _BC, unsigned int L, double J1, double J2, double delta1, double delta2, double eta1, double eta2,
            double hx, double hz, bool add_parity_breaking = false, bool add_edge_fields = false);//double w = 0, const u64 seed = std::random_device{}());

    //<! ----------------------------------------------------- HAMILTONIAN BUILDERS
    virtual void create_hamiltonian() override;
    virtual sparse_matrix create_local_hamiltonian(int site) override;
    virtual void set_hamiltonian_elements(u64 k, double value, u64 new_idx) override;

    //<! ----------------------------------------------------- OVERRIDEN OPERATORS
    virtual std::ostream& write(std::ostream&) const override;
    virtual std::istream& read(std::istream&) override;
};

#endif