#pragma once

//#define _XXZ_SYM

#ifndef _fermions_SYM
#define _fermions_SYM

#include "../../include/hilbert_space/symmetries.hpp"
#include "../../include/hilbert_space/u1.hpp"
using U1Hilbert = QHS::U1_hilbert_space<QHS::U1::charge>;

// #include "../../include/supersymmetry.hpp"
#ifdef USE_REAL_SECTORS
    using elem_ty = double;
#else
    using elem_ty = cpx;
#endif

/// @brief Fully anisotropic spin chain (XXZ) with point symmetries
class Fermions : 
    public QHS::hamiltonian_base<elem_ty, QHS::point_symmetric<QOps::particle::fermion>>
{
    //<! ----------------------------------------------------- INHERIT TYPEDEFs FROM BASE
    typedef typename QHS::hamiltonian_base<elem_ty, QHS::point_symmetric<QOps::particle::fermion>>::matrix        matrix;
    typedef typename QHS::hamiltonian_base<elem_ty, QHS::point_symmetric<QOps::particle::fermion>>::sparse_matrix sparse_matrix;

    //<! ----------------------------------------------------- MODEL PARAMETERS
private:
    v_1d<QOps::genOp> symmetry_generators;    // list of symmetry generators

    double _t1 = 1.0;                   // nearest neighbour coupling amplitude
    double _t2 = 0.0;                   // next-nearest neighbour coupling amplitude
    double _V1 = 0.55;                  // nearest neighbour interaction amplitude
    double _V2 = 0.0;                   // next-nearest neighbour interaction amplitude
    double _mu = 0.0;                   // chemical potential (brake particle-hole symmetry)

    // bool _add_edge_fields = false;          // add additional edge fields needed for SUSY in OBC

    //<! Symmetry contained in struct
    struct {
        int k_sym;                          // quasimomentum symmetry sector
        int p_sym;                          // parity symmetry sector
        int zx_sym;                         // spin flip in X symmetry sector
        int N;                              // number of particles
    } syms;
    bool _use_symmetries = true;            // [temporary] boolean choose if use symmetries

    //<! ----------------------------------------------------- INITIALIZE MODEL
    virtual void init() override;
    void set_symmetry_generators();

public:
    //<! ----------------------------------------------------- CONSTRUCTORS
    ~Fermions() { DESTRUCTOR_CALL; }
    Fermions() = default;
    Fermions(std::istream& os);
    Fermions(int _BC, unsigned int L, unsigned int N, double t1, double t2, double V1, double V2, double mu,
                int ksym, int psym, int zxsym = 1, bool use_syms = true);

    //<! ----------------------------------------------------- HAMILTONIAN BUILDERS
    virtual void create_hamiltonian() override;
    virtual sparse_matrix create_local_hamiltonian(int site) override;
    virtual void set_hamiltonian_elements(u64 k, elem_ty value, u64 new_idx) override;

    //<! ----------------------------------------------------- OVERRIDEN OPERATORS
    virtual std::ostream& write(std::ostream&) const override;
    virtual std::istream& read(std::istream&) override;

    //<! ----------------------------------------------------- ADDITIONAL METHODS
    
    
};


#endif