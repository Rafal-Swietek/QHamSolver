#pragma once

//#define _XYZ_SYM

#ifndef _XYZ_SYM
#define _XYZ_SYM

#include "../../include/hilbert_space/symmetries.hpp"
#ifdef USE_REAL_SECTORS
    using elem_ty = double;
#else
    using elem_ty = cpx;
#endif

/// @brief Fully anisotropic spin chain (XYZ) with point symmetries
class XYZsym : 
    public hamiltonian_base<elem_ty, point_symmetric>
{
    //<! ----------------------------------------------------- INHERIT TYPEDEFs FROM BASE
    typedef typename hamiltonian_base<elem_ty, point_symmetric>::matrix        matrix;
    typedef typename hamiltonian_base<elem_ty, point_symmetric>::sparse_matrix sparse_matrix;

    //<! ----------------------------------------------------- MODEL PARAMETERS
private:
    v_1d<op::genOp> symmetry_generators;    // list of symmetry generators

    double _hz = 0.5;                       // uniform longitudinal field
    double _hx = 0.5;                       // uniform transverse field
    double _J1 = 1.0;                       // nearest neighbour coupling amplitude
    double _J2 = 0.0;                       // next-nearest neighbour coupling amplitude
    double _delta1 = 0.55;                  // nearest neighbour interaction amplitude
    double _delta2 = 0.0;                   // next-nearest neighbour interaction amplitude
    double _eta1 = 0.55;                    // nearest neighbour anisotropy in XY
    double _eta2 = 0.0;                     // next-nearest neighbour anisotropy in XY

    //<! Symmetry contained in struct
    struct {
        int k_sym;                          // quasimomentum symmetry sector
        int p_sym;                          // parity symmetry sector
        int zx_sym;                         // spin flip in X symmetry sector
        int zz_sym;                         // spin flip in Z symmetry sector
    } syms;

    //<! ----------------------------------------------------- INITIALIZE MODEL
    virtual void init() override;
    void set_symmetry_generators();

public:
    //<! ----------------------------------------------------- CONSTRUCTORS
    XYZsym() = default;
    XYZsym(std::istream& os);
    XYZsym(int _BC, unsigned int L, double J1, double J2, double delta1, double delta2, double eta1, double eta2,
            double hx, double hz, int ksym, int psym, int zzsym = 1, int zxsym = 1);

    //<! ----------------------------------------------------- HAMILTONIAN BUILDERS
    virtual void create_hamiltonian() override;
    virtual sparse_matrix create_local_hamiltonian(int site) override;
    virtual void set_hamiltonian_elements(u64 k, elem_ty value, u64 new_idx) override;

    //<! ----------------------------------------------------- OVERRIDEN OPERATORS
    virtual std::ostream& write(std::ostream&) const override;
    virtual std::istream& read(std::istream&) override;
};


#endif