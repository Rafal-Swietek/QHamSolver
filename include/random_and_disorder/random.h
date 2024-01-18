#pragma once

#include <random>
#include <ctime>
#include <numeric>

enum class dist{
	uniform,
	normal
};

/// <summary>
/// Random number generator class
/// </summary>
class randomGen {
	uint64_t SeedInit(uint64_t n) const
	{
		std::vector<uint64_t> s(16, 0);
		for (int i = 0; i < 16; i++)
		{
			n ^= n >> 12;   // a
			n ^= n << 25;   // b
			n ^= n >> 27;   // c
			s[i] = n * 2685821657736338717LL;                               // 2685821657736338717 = 72821711 * 36882155347, from Pierre L'Ecuyer's paper
		}
		return std::accumulate(s.begin(), s.end(), 0.0);
	}
protected:
	std::mt19937_64 engine;
	//XoshiroCpp::Xoshiro256PlusPlus engine;
	std::uint64_t init_seed;
public:
	~randomGen() { DESTRUCTOR_CALL; };
	explicit randomGen(const std::uint64_t seed = std::random_device{}()) {
		this->init_seed = seed;
		this->engine = std::mt19937_64(seed);
        CONSTRUCTOR_CALL;
        #if defined(EXTRA_DEBUG)
            std::cout << FUN_SIGNATURE << "::\n\t randomGen initialized with: "
                << var_name_value(this->init_seed, 0) << std::endl;
        #endif
	}
	void reset()
		{this->engine = std::mt19937_64(this->init_seed);}
	[[nodiscard]] auto get_seed() const { return this->init_seed; }
	
	//------------------------------------------------------------------------------ WRAPPERS ON RANDOM FUNCTIONS
	//<! ------------------------------------------------ UNIFORM
	std::complex<double> cpx_uniform(std::complex<double> _min = 0, std::complex<double> _max = 1) 
	{
		std::uniform_real_distribution<double> dist(std::real(_min), std::real(_max));
		return std::complex<double>(dist(engine), dist(engine));
	}
	double real_uniform(double _min = 0, double _max = 1) 
		{ return std::uniform_real_distribution<double>(_min, _max)(engine); }
	uint64_t int_uniform(int _min, int _max) 
		{ return std::uniform_int_distribution<uint64_t>(_min, _max)(engine); }


	template <typename _type> 
	_type uniform_dist(_type _min, _type _max) 
		{ return std::uniform_real_distribution<_type>(_min, _max)(engine); }
	
	//<! ------------------------------------------------ NORMAL (GAUSSIAN)
	std::complex<double> cpx_normal(double _mean = 0, double _var = 1) 
	{
		std::normal_distribution<double> dist(_mean, _var / 2.);
		return std::complex<double>(dist(engine), dist(engine));
	}
	double real_normal(double _mean = 0.0, double _var = 1.0)
		{ return std::normal_distribution<double>(_mean, _var)(engine); }

	template <typename _type> 
	_type normal(double _mean = 0.0, double _var = 1.0)
		{ return std::normal_distribution<_type>(_mean, _var)(engine); }


	//<! ------------------------------------------------ HELPERS

	template<typename _type, dist dist_type>
	_type distribution(_type arg1, _type arg2){
		if constexpr (dist_type == dist::uniform)
			return uniform_dist<_type>(arg1, arg2);
		else if constexpr (dist_type == dist::normal)
			return normal<_type>(arg1, arg2);
		else
			static_check((dist_type == dist::uniform) || (dist_type == dist::normal), 
					"Not implemented other diatributions than uniform and normal");
	}
	//<! ------------------------------------------------ RANDOM STATES
	/// @brief Generate Random vector with unform distribution
	/// @tparam _type type of random numbers
	/// @param size dimension of vector
	/// @param h width of distribution ( values in [-h, h] )
	/// @return random vector uniformly distributed
	template <typename _type>
	arma::Col<_type> create_random_vec(const uint64_t size, double h) 
	{
		arma::Col<_type> random_vec(size, arma::fill::zeros);
		for (u64 j = 0; j <= size / 2.; j++) {
			u64 idx = size / (long)2 - j;
			random_vec(idx) = this->uniform_dist<_type>(-h, h);
			idx += 2 * j;
			if (idx < size) random_vec(idx) = this->uniform_dist<_type>(-h, h);
		}
		return random_vec;
	}

	/// @brief Generate Random vector with given distribution
	/// @tparam _type type of random numbers
	/// @tparam dist_type template type of distribution (uniform, normal, ...)
	/// @param size dimension of vector
	/// @param arg1 first argument (min of dist, mean, ...) <- distribution dependent
	/// @param arg2 second argument (max of dist, var, ...) <- distribution dependent
	/// @return random vector distributed by template distribution
	template <typename _type, dist dist_type> 
	arma::Col<_type> create_random_vec(const uint64_t size, _type arg1 = _type(0), _type arg2 = _type(1)) 
	{
		arma::Col<_type> random_vec(size, arma::fill::zeros);
		for (u64 j = 0; j <= size / 2.; j++) {
			u64 idx = size / (long)2 - j;
			random_vec(idx) = this->distribution<_type, dist_type>(arg1, arg2);
			idx += 2 * j;
			if (idx < size) random_vec(idx) = this->distribution<_type, dist_type>(arg1, arg2);
		}
		return random_vec;
	}

	/// @brief Generate Random matrix with given distribution
	/// @tparam _type type of random numbers
	/// @tparam dist_type template type of distribution (uniform, normal, ...)
	/// @param size dimension of matrix (size x size)
	/// @param arg1 first argument (min of dist, mean, ...) <- distribution dependent
	/// @param arg2 second argument (max of dist, var, ...) <- distribution dependent
	/// @return random matrix distributed by template distribution
	template <typename _type, dist dist_type> 
	arma::Mat<_type> random_matrix(const uint64_t size, _type arg1 = _type(0), _type arg2 = _type(1)) 
	{
		arma::Mat<_type> matrix(size, size);
		for(int n = 0; n < size; n++)
			for(int m = 0; m < size; m++)
				matrix(n, m) = this->distribution<_type, dist_type>(arg1, arg2);
		return matrix;
	}


	/// @brief Sample Haar-random matrix through QR-decomposition of complex matrix
	/// @param size dimension of matrix
	/// @return Haar-random matrix
	arma::cx_mat haar_random(u64 size){
		arma::cx_mat matrix(size, size);
		std::normal_distribution<double> dist(0.0, 1.0);
		for(int n = 0; n < size; n++)
			for(int m = 0; m < size; m++)
				matrix(n, m) = dist(engine) + 1i * dist(engine);
		arma::cx_mat Q, R;
		arma::qr(Q, R, matrix);
		return Q;
	};
};

//<! ------------------------------------------------ UNIFORM
template <> 
inline 
int 
	randomGen::uniform_dist<int>(
		int _min, int _max)
	{ return int_uniform(_min, _max); }

template <> 
inline 
unsigned long long int 
	randomGen::uniform_dist<unsigned long long int>(
		unsigned long long int _min, unsigned long long int _max)
	{ return int_uniform(_min, _max); }

template <> 
inline 
std::complex<double> 
	randomGen::uniform_dist<std::complex<double>>(
		std::complex<double> _min, std::complex<double> _max)
	{ return cpx_uniform(_min, _max); }

//<! ------------------------------------------------ NORMAL (GAUSSIAN)
template <> 
inline 
std::complex<double> 
	randomGen::normal<std::complex<double>>(
		double _mean, double _var)
	{ return cpx_normal(_mean, _var); }