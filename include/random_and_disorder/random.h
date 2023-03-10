#pragma once

#include <random>
#include <ctime>
#include <numeric>

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
	std::complex<double> randomCpx_uni(double _min = 0, double _max = 1) 
	{
		std::uniform_real_distribution<double> dist(_min, _max);
		return std::complex<double>(dist(engine), dist(engine));
	}
	double randomReal_uni(double _min = 0, double _max = 1) 
		{ return std::uniform_real_distribution<double>(_min, _max)(engine); }
	uint64_t randomInt_uni(int _min, int _max) 
		{ return std::uniform_int_distribution<uint64_t>(_min, _max)(engine); }


	template <typename _type> 
	_type random_uni(double _min, double _max) 
		{ return std::uniform_real_distribution<_type>(_min, _max)(engine); }

	template <typename _type> 
	arma::Col<_type> create_random_vec(const uint64_t size, double h) 
	{
		arma::Col<_type> random_vec(size, arma::fill::zeros);
		for (u64 j = 0; j <= size / 2.; j++) {
			u64 idx = size / (long)2 - j;
			random_vec(idx) = random_uni<_type>(-h, h);
			idx += 2 * j;
			if (idx < size) random_vec(idx) = random_uni<_type>(-h, h);
		}
		return random_vec;
	}

	template <typename _type> 
	arma::Col<_type> create_random_vec(const uint64_t size, _type min = _type(0), _type max = _type(1)) 
	{
		arma::Col<_type> random_vec(size, arma::fill::zeros);
		for (u64 j = 0; j <= size / 2.; j++) {
			u64 idx = size / (long)2 - j;
			random_vec(idx) = random_uni<_type>(min, max);
			idx += 2 * j;
			if (idx < size) random_vec(idx) = random_uni<_type>(min, max);
		}
		return random_vec;
	}
};

template <> 
inline 
int 
	randomGen::random_uni<int>(
		double _min, double _max)
	{ return randomInt_uni(_min, _max); }

template <> 
inline 
std::complex<double> 
	randomGen::random_uni<std::complex<double>>(
		double _min, double _max)
	{ return randomCpx_uni(_min, _max); }
