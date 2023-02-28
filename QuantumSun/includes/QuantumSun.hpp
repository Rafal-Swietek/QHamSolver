#pragma once
#ifndef _QUANTUM_SUN
#define _QUANTUM_SUN

/// @brief Model for EBT, QuantumSun model
class QuantumSun : 
    public hamiltonian_base<double, full_hilbert_space>
{
    //<! ----------------------------------------------------- INHERIT TYPEDEFs FROM BASE
    typedef typename hamiltonian_base<double, full_hilbert_space>::matrix        matrix;
    typedef typename hamiltonian_base<double, full_hilbert_space>::sparse_matrix sparse_matrix;

    //<! ----------------------------------------------------- MODEL PARAMETERS
private:
    disorder<double> disorder_generator;    // generator for random disorder and couplings
    
    arma::vec _long_range_couplings;        // random coupling, i.e. distance of spins to grain
    arma::vec _disorder;                    // disorder array on Z field
    
    double _w = 0.5;                        // disorder value on top of uniform field
    double _J = 1.;                         // coupling amplitude
    double _hz = 1.0;                       // longitudinal uniform field
    double _alfa = 0.75;                    // coupling base -- controls long-range interaction
    double _zeta = 0.2;                     // randomness in long range coupling (random distance between localised spins abd grain)

    double _gamma = 1.0;                    // prefactor to ergodic grain (controls ergodicity)
    u64 _seed = std::random_device{}();     // seed for random generator
    
    int grain_size = 3;                     // ergodic grain size
    bool _initiate_avalanche = 1;           // start first coupling with =1.0 . (i.e. exponent u_0 = 0)
    bool _norm_grain = 0;                   // normalize grain to unit hilbert-schmidt norm?
    
    //<! ----------------------------------------------------- INITIALIZE MODEL
    virtual void init() override
    {   
        // initialize hilbert space
        this->_hilbert_space = full_hilbert_space(this->system_size);
        this->dim = this->_hilbert_space.get_hilbert_space_size();

        // initialize disorder
        disorder_generator = disorder<double>(this->_seed);

        // create hamiltonian
        this->create_hamiltonian();
    }

public:
    //<! ----------------------------------------------------- CONSTRUCTORS
    QuantumSun() = default;
    QuantumSun(std::istream& os);
    QuantumSun(int L, double J, double alfa, double gamma,
            double w, double hz, const u64 seed = std::random_device{}(), 
            int M = 3, double zeta = 0.2, bool initiate_avalanche = false, bool normalize_grain = false);

    //<! ----------------------------------------------------- HAMILTONIAN BUILDERS
    virtual void create_hamiltonian() override;
    virtual sparse_matrix create_local_hamiltonian(int site) override;
    virtual void set_hamiltonian_elements(u64 k, double value, u64 new_idx) override;

    //<! ----------------------------------------------------- OVERRIDEN OPERATORS
    virtual std::ostream& write(std::ostream&) const override;
    virtual std::istream& read(std::istream&) override;
};

#endif