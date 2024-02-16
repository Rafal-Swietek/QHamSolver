#pragma once

//#define _XYZ_SYM

#ifndef _TIFP
#define _TIFP

#include "../../include/hilbert_space/symmetries.hpp"
// #ifdef USE_REAL_SECTORS
//     using elem_ty = double;
// #else
//     using elem_ty = cpx;
// #endif

using elem_ty = cpx;
/// @brief Fully anisotropic spin chain (XYZ) with point symmetries
class TIFP : 
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
        int z2_sym;
        int z1_sym;
        int zz_sym;
    } syms;

    double _J = 0.5;                        // model parameter
    double _c = 0.0;                        // coupling to Q5
    bool _use_symmetries = true;            // [temporary] boolean choose if use symmetries
    

    //<! ----------------------------------------------------- INITIALIZE MODEL
    virtual void init() override;
    void set_symmetry_generators();

public:
    //<! ----------------------------------------------------- CONSTRUCTORS
    TIFP() = default;
    TIFP(std::istream& os);
    TIFP(int _BC, unsigned int L, double J, double c, int zzsym, int z1sym, int z2sym, bool use_syms = true);

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