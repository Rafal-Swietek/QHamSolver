#pragma once
#ifndef _ANDERSON
#define _ANDERSON

#if DIM == 1
    #define lattice_type lattice1D
#elif DIM == 2
    #define lattice_type lattice2D
#else
    #define lattice_type lattice3D
#endif

#include "../../include/lattice.hpp"
/// @brief Model for EBT, Anderson model
class Anderson : 
    public hamiltonian_base<double, full_hilbert_space>
{
    //<! ----------------------------------------------------- INHERIT TYPEDEFs FROM BASE
    typedef typename hamiltonian_base<double, full_hilbert_space>::matrix        matrix;
    typedef typename hamiltonian_base<double, full_hilbert_space>::sparse_matrix sparse_matrix;

    //<! ----------------------------------------------------- MODEL PARAMETERS
private:
    lattice_type lattice;                   // lattice of system (cubic by default)
    disorder<double> disorder_generator;    // generator for random disorder and couplings
    arma::vec _disorder;                    // disorder array on Z field
    
    double _w = 0.5;                        // disorder value on top of uniform field
    double _J = 1.;
    u64 _seed = std::random_device{}();     // seed for random generator
    
    
    //<! ----------------------------------------------------- INITIALIZE MODEL
    virtual void init() override
    {   
        // initialize lattice
        this->lattice = lattice_type(this->system_size);
        this->dim = lattice.volume;

        // initialize disorder
        this->disorder_generator = disorder<double>(this->_seed);

        // create hamiltonian
        this->create_hamiltonian();
    }

public:
    //<! ----------------------------------------------------- CONSTRUCTORS
    Anderson() = default;
    Anderson(std::istream& os);
    Anderson(int L, double J, double w, const u64 seed = std::random_device{}());

    //<! ----------------------------------------------------- HAMILTONIAN BUILDERS
    virtual void create_hamiltonian() override;
    virtual sparse_matrix create_local_hamiltonian(int site) override;
    
    virtual void set_hamiltonian_elements(u64 k, double value, u64 new_idx) override {};

    //<! ----------------------------------------------------- OVERRIDEN OPERATORS
    virtual std::ostream& write(std::ostream&) const override;
    virtual std::istream& read(std::istream&) override;

    //<! ----------------------------------------------------- OTHERS
    auto& get_lattice() const { return this->lattice; }
    auto& get_randGen() const { return this->disorder_generator; }

};

#endif