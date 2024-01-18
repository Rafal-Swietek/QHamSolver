#pragma once

#ifndef _scaledIsing
#define _scaledIsing

/// @brief Fully anisotropic spin chain (XYZ)
class pertIsing : 
    public QHS::hamiltonian_base<double, QHS::full_hilbert_space>
{
    //<! ----------------------------------------------------- INHERIT TYPEDEFs FROM BASE
    typedef typename QHS::hamiltonian_base<double, QHS::full_hilbert_space>::matrix        matrix;
    typedef typename QHS::hamiltonian_base<double, QHS::full_hilbert_space>::sparse_matrix sparse_matrix;

    //<! ----------------------------------------------------- MODEL PARAMETERS
private:
    disorder<double> disorder_generator;    // generator for random disorder and couplings
    
    arma::mat _couplings;                   // random Z_i Z_j couplings
    arma::vec _disorder;                    // disorder array on Z field
    
    double _g = 0.0;                        // disorder value on top of uniform field
    
    u64 _seed = std::random_device{}();     // seed for random generator
    
    //<! ----------------------------------------------------- INITIALIZE MODEL
    virtual void init() override
    {   
        // initialize hilbert space
        this->_hilbert_space = QHS::full_hilbert_space(this->system_size);
        this->dim = this->_hilbert_space.get_hilbert_space_size();

        // initialize disorder
        disorder_generator = disorder<double>(this->_seed);
        this->_disorder = arma::vec(this->system_size, arma::fill::zeros);
        this->_couplings = arma::mat(this->system_size, this->system_size, arma::fill::zeros);

        // create hamiltonian
        this->create_hamiltonian();
    }

public:
    //<! ----------------------------------------------------- CONSTRUCTORS
    pertIsing() = default;
    pertIsing(std::istream& os);
    pertIsing(int _BC, unsigned int L, double g, const u64 seed = std::random_device{}());

    //<! ----------------------------------------------------- HAMILTONIAN BUILDERS
    virtual void create_hamiltonian() override;
    virtual sparse_matrix create_local_hamiltonian(int site) override;
    virtual void set_hamiltonian_elements(u64 k, double value, u64 new_idx) override;

    //<! ----------------------------------------------------- OVERRIDEN OPERATORS
    virtual std::ostream& write(std::ostream&) const override;
    virtual std::istream& read(std::istream&) override;
};

#endif