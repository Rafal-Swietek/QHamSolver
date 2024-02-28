#pragma once
#ifndef _QUANTUM_SUN
#define _QUANTUM_SUN

#ifndef ENSEMBLE
    #define ENSEMBLE GOE
    #pragma message ("--> Using implicit random matrix ensemble: i.e., Gaussian Orthogonal Ensemble")
#endif

#include "../../include/hilbert_space/u1.hpp"
using U1Hilbert = QHS::U1_hilbert_space<QHS::U1::spin>;

/// @brief Model for EBT, QuantumSun model
class QuantumSunU1 : 
    public QHS::hamiltonian_base<double, U1Hilbert>
{
    //<! ----------------------------------------------------- INHERIT TYPEDEFs FROM BASE
    typedef typename QHS::hamiltonian_base<double, U1Hilbert>::matrix        matrix;
    typedef typename QHS::hamiltonian_base<double, U1Hilbert>::sparse_matrix sparse_matrix;

    //<! ----------------------------------------------------- MODEL PARAMETERS
private:
    disorder<double> disorder_generator;    // generator for random disorder and couplings
    disorder<int> neighbor_generator;       // generator for random neighbor in interaction term 
    
    arma::vec _long_range_couplings;        // random coupling, i.e. distance of spins to grain
    arma::vec _disorder;                    // disorder array on Z field
    
    ENSEMBLE grain;                         // ergodic grain drawn from some ensemble

    double Sz = 0.0;                        // total magnetization sector

    double _w = 0.5;                        // disorder value on top of uniform field
    double _J = 1.;                         // coupling amplitude
    double _hz = 1.0;                       // longitudinal uniform field
    double _alfa = 0.75;                    // coupling base -- controls long-range interaction
    double _zeta = 0.2;                     // randomness in long range coupling (random distance between localised spins abd grain)

    double _gamma = 1.0;                    // prefactor to ergodic grain (controls ergodicity)
    u64 _seed = std::random_device{}();     // seed for random generator
    
    int num_of_spins;                       // number of localised spins
    int grain_size = 3;                     // ergodic grain size
    bool _initiate_avalanche = 1;           // start first coupling with =1.0 . (i.e. exponent u_0 = 0)
    bool _norm_grain = 1;                   // normalize grain to unit hilbert-schmidt norm?
    
    //<! ----------------------------------------------------- INITIALIZE MODEL
    virtual void init() override
    {   
        // initialize hilbert space
        this->_hilbert_space = U1Hilbert(this->system_size, this->Sz);
        this->dim = this->_hilbert_space.get_hilbert_space_size();

        // initialize disorder
        this->disorder_generator = disorder<double>(this->_seed);
        this->neighbor_generator = disorder<int>(this->_seed);
	    this->grain = ENSEMBLE(this->_seed);

        // create hamiltonian
        this->create_hamiltonian();
    }

public:
    //<! ----------------------------------------------------- CONSTRUCTORS
    QuantumSunU1() = default;
    QuantumSunU1(std::istream& os);
    QuantumSunU1(int L, double J, double alfa, double gamma,
            double w, double hz, double Sz = 0.0, const u64 seed = std::random_device{}(), 
            int M = 3, double zeta = 0.2, bool initiate_avalanche = true, bool normalize_grain = true);

    //<! ----------------------------------------------------- HAMILTONIAN BUILDERS
    virtual void create_hamiltonian() override;
    virtual sparse_matrix create_local_hamiltonian(int site) override;
    virtual void set_hamiltonian_elements(u64 k, double value, u64 new_idx) override;

    //<! ----------------------------------------------------- OVERRIDEN OPERATORS
    virtual std::ostream& write(std::ostream&) const override;
    virtual std::istream& read(std::istream&) override;
};

#endif