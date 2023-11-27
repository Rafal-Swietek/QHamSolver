#pragma once

//#define _XXZ_SYM

#ifndef _XXZ_SYM
#define _XXZ_SYM

#include "../../include/hilbert_space/symmetries.hpp"
#include "../../include/hilbert_space/u1.hpp"
using U1Hilbert = QHS::U1_hilbert_space<QHS::U1::spin>;

// #include "../../include/supersymmetry.hpp"
#ifdef USE_REAL_SECTORS
    using elem_ty = double;
#else
    using elem_ty = cpx;
#endif

/// @brief Fully anisotropic spin chain (XXZ) with point symmetries
class XXZsym : 
    public QHS::hamiltonian_base<elem_ty, QHS::point_symmetric>
{
    //<! ----------------------------------------------------- INHERIT TYPEDEFs FROM BASE
    typedef typename QHS::hamiltonian_base<elem_ty, QHS::point_symmetric>::matrix        matrix;
    typedef typename QHS::hamiltonian_base<elem_ty, QHS::point_symmetric>::sparse_matrix sparse_matrix;

    //<! ----------------------------------------------------- MODEL PARAMETERS
private:
    v_1d<op::genOp> symmetry_generators;    // list of symmetry generators

    double _hz = 0.5;                       // uniform longitudinal field
    double _J1 = 1.0;                       // nearest neighbour coupling amplitude
    double _J2 = 0.0;                       // next-nearest neighbour coupling amplitude
    double _delta1 = 0.55;                  // nearest neighbour interaction amplitude
    double _delta2 = 0.0;                   // next-nearest neighbour interaction amplitude

    // bool _add_edge_fields = false;          // add additional edge fields needed for SUSY in OBC

    //<! Symmetry contained in struct
    struct {
        int k_sym;                          // quasimomentum symmetry sector
        int p_sym;                          // parity symmetry sector
        int zx_sym;                         // spin flip in X symmetry sector
        float Sz;                           // magnetization sector
    } syms;

    //<! ----------------------------------------------------- INITIALIZE MODEL
    virtual void init() override;
    void set_symmetry_generators();

public:
    //<! ----------------------------------------------------- CONSTRUCTORS
    ~XXZsym() { DESTRUCTOR_CALL; }
    XXZsym() = default;
    XXZsym(std::istream& os);
    XXZsym(int _BC, unsigned int L, double J1, double J2, double delta1, double delta2, double hz, 
                int ksym, int psym, int zxsym = 1, float Sz = 0.0);

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