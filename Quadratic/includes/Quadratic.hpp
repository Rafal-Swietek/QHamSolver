#pragma once
#ifndef _ANDERSON
#define _ANDERSON

#include "../../include/lattices/_base.hpp"
#if DIM == 1
    #define lattice_type lattice::lattice1D
#elif DIM == 2
    #define lattice_type lattice::lattice2D
#else
    #define lattice_type lattice::lattice3D
#endif

/// @brief Model for EBT, Anderson model
class Quadratic : 
    public QHS::hamiltonian_base<double, QHS::full_hilbert_space>
{
    //<! ----------------------------------------------------- INHERIT TYPEDEFs FROM BASE
    typedef typename QHS::hamiltonian_base<double, QHS::full_hilbert_space>::matrix        matrix;
    typedef typename QHS::hamiltonian_base<double, QHS::full_hilbert_space>::sparse_matrix sparse_matrix;

    //<! ----------------------------------------------------- MODEL PARAMETERS
private:
    lattice_type _lattice;                  // lattice of system (cubic by default)
    disorder<double> disorder_generator;    // generator for random disorder and couplings
    #ifdef PLRB
        rmt::uniform_ensemble random_matrix;    // generator of uniform random matrices
    #else
        GOE random_matrix;                      // generator of random matrices (GOE,GUE,...)
    #endif
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
        this->_lattice = lattice_type(this->system_size, this->_boundary_condition);
        this->dim = this->_lattice.volume;

        // initialize disorder
        this->disorder_generator = disorder<double>(this->_seed);
        #ifdef PLRB
            this->random_matrix = rmt::uniform_ensemble(this->_seed);
        #else
            this->random_matrix = GOE(this->_seed);
        #endif
        // create hamiltonian
        this->create_hamiltonian();
    }

public:
    //<! ----------------------------------------------------- CONSTRUCTORS
    Quadratic() = default;
    // Quadratic(std::istream& os);
    Quadratic(int L, double J, double w = 0, const u64 seed = std::random_device{}(), double g = 0.0, bool _bound_cond = 0);

    //<! ----------------------------------------------------- HAMILTONIAN BUILDERS
    virtual void create_hamiltonian() override;
    
    // local hamiltonian has no reason to exist in single-particle models
    virtual sparse_matrix create_local_hamiltonian(int site) override { return sparse_matrix(this->dim, this->dim); };
    
    virtual void set_hamiltonian_elements(u64 k, double value, u64 new_idx) override {};

    //<! ----------------------------------------------------- OVERRIDEN OPERATORS
    virtual std::ostream& write(std::ostream&) const override;
    virtual std::istream& read(std::istream&) override;

    //<! ----------------------------------------------------- OTHERS
    auto& get_lattice() const { return this->_lattice; }
    auto& get_randGen() const { return this->disorder_generator; }

};

#endif