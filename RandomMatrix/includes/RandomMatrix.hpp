#pragma once
#ifndef _RANDOM_MATRIX
#define _RANDOM_MATRIX

#ifndef ENSEMBLE
    #define ENSEMBLE _ENSEMBLE
    #pragma message ("--> Using implicit random matrix ensemble: i.e., Gaussian Orthogonal Ensemble")
#endif

/// @brief Model for EBT, QuantumSun model
class RandomMatrix : 
    public hamiltonian_base<double, full_hilbert_space>
{
    //<! ----------------------------------------------------- INHERIT TYPEDEFs FROM BASE
    typedef typename hamiltonian_base<double, full_hilbert_space>::matrix        matrix;
    typedef typename hamiltonian_base<double, full_hilbert_space>::sparse_matrix sparse_matrix;

    //<! ----------------------------------------------------- MODEL PARAMETERS
private:
    ENSEMBLE generator;                     // ergodic grain drawn from some ensemble

    double _b = 0.5;                        // 
    double _alfa = 1.;                      // 
    double _gamma = 1.0;                    // 

    u64 _seed = std::random_device{}();     // seed for random generator
    
    
    //<! ----------------------------------------------------- INITIALIZE MODEL
    virtual void init() override
    {   
	    this->generator = ENSEMBLE(this->_seed);

        // create hamiltonian
        this->create_hamiltonian();
    }

public:
    //<! ----------------------------------------------------- CONSTRUCTORS
    RandomMatrix() = default;
    RandomMatrix(std::istream& os);
    RandomMatrix(u64 dim, double b, double alfa, double gamma,
            const u64 seed = std::random_device{}());

    //<! ----------------------------------------------------- HAMILTONIAN BUILDERS
    virtual void create_hamiltonian() override;
    virtual sparse_matrix create_local_hamiltonian(int site) override;
    virtual void set_hamiltonian_elements(u64 k, double value, u64 new_idx) override {};

    //<! ----------------------------------------------------- OVERRIDEN OPERATORS
    virtual std::ostream& write(std::ostream&) const override;
    virtual std::istream& read(std::istream&) override;
};

#endif