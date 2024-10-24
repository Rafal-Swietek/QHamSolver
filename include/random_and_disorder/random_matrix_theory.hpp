#pragma once
//  Redefine the RMT theory to a class with builtin switch (beta=1,2,4) to choose ensemble
//  add eigenvalues and distributions later on. Treat Gaussian and Circular seperately.
//  Think what to do about the matrix-type
//
//
//
//
//


namespace rmt{

    
    template <typename _ty>
class random_matrix_theory : public randomGen{
protected:
    /// @brief Initializing function for default constructor
    void init(const std::uint64_t seed) {
        this->init_seed = seed;
        this->reset();

        CONSTRUCTOR_CALL;
        _extra_debug(
            std::cout << FUN_SIGNATURE << "::\n\t disorder initialized with: "
                << var_name_value(this->init_seed, 0) << std::endl;
        )
    }
public:
    virtual ~random_matrix_theory() = 0;
    virtual arma::Mat<_ty> generate_matrix(u64 size) = 0;
};

template <typename _ty>
random_matrix_theory<_ty>::~random_matrix_theory() {}

/// @brief 
class uniform_ensemble : public random_matrix_theory<double>{

public:
    uniform_ensemble(const std::uint64_t seed = std::random_device{}())
        { this->init(seed); }
    virtual arma::Mat<double> generate_matrix(u64 size) override
    {
        arma::mat matrix(size, size);
		std::uniform_real_distribution<double> dist(-1.0, 1.0);
		for(int n = 0; n < size; n++){
			matrix(n, n) = dist(engine);
			for(int m = n + 1; m < size; m++){
				matrix(n, m) = dist(engine);
                matrix(m, n) = matrix(n, m);
            }
        }
		return matrix;
    }
};

//-------------------------------------------------------------------------------------------------- GAUSSIAN ENSEMBLES
/// @brief 
class gaussian_orthogonal_ensemble : public random_matrix_theory<double>{

public:
    gaussian_orthogonal_ensemble(const std::uint64_t seed = std::random_device{}())
        { this->init(seed); }
    virtual arma::Mat<double> generate_matrix(u64 size) override
    {
        arma::mat matrix(size, size);
		std::normal_distribution<double> dist(0.0, 1.0);
		for(int n = 0; n < size; n++)
			for(int m = 0; m < size; m++)
				matrix(n, m) = dist(engine);

        // add proper normalization
		return (matrix + matrix.t()) / std::sqrt(2);
    }
};

/// @brief 
class gaussian_unitary_ensemble : public random_matrix_theory<cpx>{

public:
    gaussian_unitary_ensemble(const std::uint64_t seed = std::random_device{}())
        { this->init(seed); }
    virtual arma::Mat<cpx> generate_matrix(u64 size) override
    {
        arma::cx_mat matrix(size, size);
		std::normal_distribution<double> dist(0.0, 1.0);
		for(int n = 0; n < size; n++)
			for(int m = 0; m < size; m++)
				matrix(n, m) = dist(engine) + 1i * dist(engine);

        // add proper normalization
		return (matrix + matrix.t()) / std::sqrt(2);
    }
};

/// @brief 
class gaussian_symplectic_ensemble : public random_matrix_theory<cpx>{

public:
    gaussian_symplectic_ensemble(const std::uint64_t seed = std::random_device{}())
        { this->init(seed); }
    virtual arma::Mat<cpx> generate_matrix(u64 size) override
    {
        arma::cx_mat matrix1(size, size), matrix2(size, size);
		std::normal_distribution<double> dist(0.0, 1.0);
		for(int n = 0; n < size; n++){
			for(int m = 0; m < size; m++){
				matrix1(n, m) = dist(engine) + 1i * dist(engine);
                matrix2(n, m) = dist(engine) + 1i * dist(engine);
            }
        }
        
        // add proper normalization
        arma::mat A(2,2, arma::fill::zeros);    A(0,0) = 1.0;
		arma::cx_mat matrix = arma::kron(A, matrix1);
        A.zeros();  A(1,1) = 1.0;
        matrix += arma::kron(A, arma::conj(matrix1));
        A.zeros();  A(0,1) = 1.0;
        matrix += arma::kron(A, matrix2);
        A.zeros();  A(1,0) = 1.0;
        matrix += arma::kron(A, -arma::conj(matrix2));
	    return (matrix + matrix.t()) / std::sqrt(2);
    }
};


//-------------------------------------------------------------------------------------------------- CIRCULAR ENSEMBLES


/// @brief Cirtucal orthogonal ensemble
class circular_orthogonal_ensemble : public random_matrix_theory<double>{

public:
    circular_orthogonal_ensemble(const std::uint64_t seed = std::random_device{}())
        { this->init(seed); }

    virtual arma::Mat<double> generate_matrix(u64 size) override
    {
        arma::cx_mat U = this->haar_random(size);
		return arma::real(U.st() * U);
    }
};

/// @brief 
class circular_unitary_ensemble : public random_matrix_theory<cpx>{

public:
    circular_unitary_ensemble(const std::uint64_t seed = std::random_device{}())
        { this->init(seed); }

    virtual arma::Mat<cpx> generate_matrix(u64 size) override
        { return this->haar_random(size); }
};

/// @brief 
class circular_symplectic_ensemble : public random_matrix_theory<cpx>{

public:
    circular_symplectic_ensemble(const std::uint64_t seed = std::random_device{}())
        { this->init(seed); }

    virtual arma::Mat<cpx> generate_matrix(u64 size) override
    {
        arma::cx_mat U = this->haar_random(2 * size);
        arma::mat Z = arma::diagmat(arma::vec(2*size, arma::fill::ones), 1) - arma::diagmat(arma::vec(2*size, arma::fill::ones), -1);
	    return (Z * U.st() * Z.st()) * U;
    }
};

}

using GOE = rmt::gaussian_orthogonal_ensemble;
using GUE = rmt::gaussian_unitary_ensemble;
using GSE = rmt::gaussian_symplectic_ensemble;
using COE = rmt::circular_orthogonal_ensemble;
using CUE = rmt::circular_unitary_ensemble;
using CSE = rmt::circular_symplectic_ensemble;