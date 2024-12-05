#pragma once
#ifndef _QUADRATIC_QUANTUM_SUN
#define _QUADRATIC_QUANTUM_SUN

#ifndef ENSEMBLE
    #define ENSEMBLE GOE
    #pragma message ("--> Using implicit random matrix ensemble: i.e., Gaussian Orthogonal Ensemble")
#endif

/// @brief Model for EBT in single-particle sector, QuantumSun model
class QuadQSun : 
    public QHS::hamiltonian_base<double, QHS::full_hilbert_space>
{
    //<! ----------------------------------------------------- INHERIT TYPEDEFs FROM BASE
    typedef typename QHS::hamiltonian_base<double, QHS::full_hilbert_space>::matrix        matrix;
    typedef typename QHS::hamiltonian_base<double, QHS::full_hilbert_space>::sparse_matrix sparse_matrix;

    //<! ----------------------------------------------------- MODEL PARAMETERS
private:
    disorder<double> disorder_generator;    // generator for random disorder and couplings
    disorder<int> neighbor_generator;       // generator for random neighbor in interaction term 
    
    arma::vec _long_range_couplings;        // random coupling, i.e. distance of spins to grain
    arma::vec _disorder;                    // disorder array on Z field
    arma::Col<arma::s32> random_neigh;      // random neighbor
    arma::mat H_grain;                      // hamiltonian of the grain
    
    ENSEMBLE grain;                         // ergodic grain drawn from some ensemble

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
        // this->_hilbert_space = QHS::full_hilbert_space(this->system_size);
        this->dim = this->system_size;//this->_hilbert_space.get_hilbert_space_size();

        // initialize disorder
        this->disorder_generator = disorder<double>(this->_seed);
        this->neighbor_generator = disorder<int>(this->_seed);
	    this->grain = ENSEMBLE(this->_seed);

        // create hamiltonian
        this->create_hamiltonian();
    }

public:
    //<! ----------------------------------------------------- CONSTRUCTORS
    QuadQSun() = default;
    QuadQSun(std::istream& os);
    QuadQSun(int L, double J, double alfa, double gamma,
            double w, double hz, const u64 seed = std::random_device{}(), 
            int M = 3, double zeta = 0.2, bool initiate_avalanche = true, bool normalize_grain = true);

    //<! ----------------------------------------------------- HAMILTONIAN BUILDERS
    virtual void create_hamiltonian() override;
    virtual sparse_matrix create_local_hamiltonian(int site) override;
    virtual void set_hamiltonian_elements(u64 k, double value, u64 new_idx) override;

    //<! ----------------------------------------------------- OVERRIDEN OPERATORS
    virtual std::ostream& write(std::ostream&) const override;
    virtual std::istream& read(std::istream&) override;


    //<! ----------------------------------------------------- GETTERS
    auto get_grain()	    const { return this->H_grain; }		            // get grain matrix
    auto get_neighs()	    const { return this->random_neigh; }		    // get random neighbours
    auto get_disorder()	    const { return this->_disorder; }		        // get disorder array
    auto get_interaction()	const { return this->_long_range_couplings; }   // get interaction array
};

#endif