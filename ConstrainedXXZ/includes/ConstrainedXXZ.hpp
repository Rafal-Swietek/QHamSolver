#pragma once

//#define _XYZ_SYM

#ifndef _ConstrainedXXZ
#define _ConstrainedXXZ

#include "../../include/hilbert_space/symmetries.hpp"
#include "../../include/hilbert_space/constrained.hpp"
#ifdef USE_REAL_SECTORS
    using elem_ty = double;
#else
    using elem_ty = cpx;
#endif

// using elem_ty = cpx;
/// @brief Fully anisotropic spin chain (XYZ) with point symmetries
class ConstrainedXXZ : 
    public QHS::hamiltonian_base<elem_ty, QHS::point_symmetric>
{
    //<! ----------------------------------------------------- INHERIT TYPEDEFs FROM BASE
    typedef typename QHS::hamiltonian_base<elem_ty, QHS::point_symmetric>::matrix        matrix;
    typedef typename QHS::hamiltonian_base<elem_ty, QHS::point_symmetric>::sparse_matrix sparse_matrix;

    //<! ----------------------------------------------------- MODEL PARAMETERS
private:
    v_1d<QOps::genOp> symmetry_generators;    // list of symmetry generators

    //<! Symmetry contained in struct
    struct {
        int k_sym;
        int p_sym;
        float _Sz;
    } syms;

    double _J = 0.5;                        // model parameter
    double _delta = 0.0;                    // interaction
    bool _use_symmetries = true;            // [temporary] boolean choose if use symmetries

    //<! ----------------------------------------------------- INITIALIZE MODEL
    virtual void init() override;
    void set_symmetry_generators();

public:
    //<! ----------------------------------------------------- CONSTRUCTORS
    ConstrainedXXZ() = default;
    ConstrainedXXZ(std::istream& os);
    ConstrainedXXZ(int _BC, unsigned int L, double J, double delta, int ksym, int psym, float Sz = 0.0, bool use_syms = true);

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