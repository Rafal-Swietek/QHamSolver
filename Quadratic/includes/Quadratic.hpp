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
class Quadratic : 
    public hamiltonian_base<double, full_hilbert_space>
{
    //<! ----------------------------------------------------- INHERIT TYPEDEFs FROM BASE
    typedef typename hamiltonian_base<double, full_hilbert_space>::matrix        matrix;
    typedef typename hamiltonian_base<double, full_hilbert_space>::sparse_matrix sparse_matrix;

    //<! ----------------------------------------------------- MODEL PARAMETERS
private:
    lattice_type lattice;                   // lattice of system (cubic by default)
    disorder<double> disorder_generator;    // generator for random disorder and couplings
    GOE random_matrix;                      // generator of random matrices (GOE,GUE,...)
    //<! Add GUE case as well, change type of matrix

    arma::vec _disorder;                    // disorder array on Z field
    
    double _w = 0.5;                        // disorder value on top of uniform field
    double _J = 1.;
    double _g = 0.0;                        // periodicity for the Aubry-Andre model
    u64 _seed = std::random_device{}();     // seed for random generator
    
    
    //<! ----------------------------------------------------- INITIALIZE MODEL
    virtual void init() override
    {   
        // initialize lattice
        this->lattice = lattice_type(this->system_size);
        this->dim = lattice.volume;

        // initialize disorder
        this->disorder_generator = disorder<double>(this->_seed);
        this->random_matrix = GOE(this->_seed);
        // create hamiltonian
        this->create_hamiltonian();
    }

public:
    //<! ----------------------------------------------------- CONSTRUCTORS
    Quadratic() = default;
    Quadratic(std::istream& os);
    Quadratic(int L, double J, double w = 0, const u64 seed = std::random_device{}(), double g = 0.0);

    //<! ----------------------------------------------------- HAMILTONIAN BUILDERS
    virtual void create_hamiltonian() override;
    
    // local hamiltonian has no reason to exist in single-particle models
    virtual sparse_matrix create_local_hamiltonian(int site) override {};
    
    virtual void set_hamiltonian_elements(u64 k, double value, u64 new_idx) override {};

    //<! ----------------------------------------------------- OVERRIDEN OPERATORS
    virtual std::ostream& write(std::ostream&) const override;
    virtual std::istream& read(std::istream&) override;

    //<! ----------------------------------------------------- OTHERS
    auto& get_lattice() const { return this->lattice; }
    auto& get_randGen() const { return this->disorder_generator; }

};

#endif