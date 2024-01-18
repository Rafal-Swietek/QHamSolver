#pragma once
#include "random.h"

#define enable_if_convertible(type_in, type_base) \
            static_check(traits::is_convertible_v<type_in, type_base>, __FILE__"(line=" LINE_STR "): " NOT_CONVERTIBLE)

// @brief BASE CLASS FOR DISORDER LANDSCAPE
/// @tparam _ty template argument
template <typename _ty>
class disorder : public randomGen{

    typedef arma::Col<_ty> disorder_vec;
    typedef arma::Mat<_ty> random_mat;

protected:

    /// @brief Initializing function for default constructor
    void init(const std::uint64_t seed) {
        this->init_seed = seed;
        this->reset();

        CONSTRUCTOR_CALL;
        #if defined(EXTRA_DEBUG)
            std::cout << FUN_SIGNATURE << "::\n\t disorder initialized with: "
                << var_name_value(this->init_seed, 0) << std::endl;
        #endif
    }

public:
    disorder() = default;
    disorder(const std::uint64_t seed) { init(seed); };
    ~disorder() { DESTRUCTOR_CALL; };

//<! ------------------------------------------------------------------------ GENERATE DISORDER WITH VARIOUS DISTRIBUTIONS
    
    //<! ---------------------------------- UNIFORM DISTRIBUTION
    /// @brief Generate random array with uniformly distributed values
    /// @param length size of array
    /// @param _bound values contained in [- _bound, _bound]
    /// @return array with uniform random numbers
    disorder_vec uniform(u64 length, _ty _bound)
        { return this->template create_random_vec<_ty, dist::uniform>(length, _bound); }

    /// @brief Generate random array with uniformly distributed values
    /// @param length size of array
    /// @param _min minimal value of distribution
    /// @param _max maximal value of distribution
    /// @return array with uniform random numbers
    disorder_vec uniform(u64 length, _ty _min, _ty _max)
        { return this->template create_random_vec<_ty, dist::uniform>(length, _min, _max); }

    /// @brief Generate random matrix with uniformly distributed values
    /// @param length size of matrix
    /// @param _min minimal value of distribution
    /// @param _max maximal value of distribution
    /// @return matrix with uniform random numbers
    random_mat uniform_matrix(u64 length, _ty _min, _ty _max)
        { return this->template random_matrix<_ty, dist::uniform>(length, _min, _max); }


    //<! ---------------------------------- GAUSSIAN DISTRIBUTION
    /// @brief Generate random array with values filled due to gaussian distribution
    /// @param length lenght of array (size of state)
    /// @param _mean mean value of distribution
    /// @param _var variance of distribution
    /// @return array with random numbers
    disorder_vec gaussian(u64 length, _ty _mean, _ty _var)
        { return this->template create_random_vec<_ty, dist::normal>(length, _mean, _var); }

    /// @brief Generate random matrix with values filled due to gaussian distribution
    /// @param length size of matrix (length x length)
    /// @param _mean mean value of distribution
    /// @param _var variance of distribution
    /// @return matrix with gaussian random numbers
    random_mat gaussian_matrix(u64 length, _ty _mean, _ty _var)
        { return this->template random_matrix<_ty, dist::normal>(length, _mean, _var); }
    
    //<! ---------------------------------- QUASIPERIODIC DISTRIBUTION
    disorder_vec quasiperiodic(_ty amplitude, _ty phi);


    //<! ---------------------------------- LOCAL IMPURITY DISTRIBUTION
    disorder_vec impurity(u64 length, int location, _ty value, bool add_edge_impurity = true, _ty edge_value = 0.1)
    {
        disorder_vec _array(length, arma::fill::zeros);
        _array(location) = value;
        if(add_edge_impurity)
            _array(0) = edge_value;
        return _array;
    }

    disorder_vec impurity(u64 length, v_1d<u64> locations, v_1d<_ty> values)
    {   
        _assert_(locations.size() == values.size(), "ERROR: size of input arrays does not match");
        auto [min, max] = std::minmax_element(locations.begin(), locations.end());
        _assert_( (*max < length && *min >= 0), "ERROR: Positions of impurity exceed system length");
        
        disorder_vec _array(length, arma::fill::zeros);
        int i = 0;
        for(auto& loc : locations)
            _array(loc) = values[i++];
        return _array;
    }
};



//<! ----------------------------------------------------------------------------------------------- RANDOM MATRIX THEORY
#include "random_matrix_theory.hpp"


//<! ----------------------------------------------------------------------------------------------- ROUTINES WITH DISORDER
template <dist dist_type> 
inline void checkRandom(unsigned int seed) 
{
	disorder<double> my_gen(seed); 
	std::cout << "test randoms \n" << my_gen.distribution<double, dist_type>(0., 1.) << "\t" << my_gen.distribution<double, dist_type>(0., 1.) << "\t" << my_gen.distribution<double, dist_type>(0., 1.) << std::endl;
	my_gen = disorder<double>(seed);
	std::cout << "reset seed!" << std::endl;
	std::cout << my_gen.distribution<double, dist_type>(0., 1.) << "\t" << my_gen.distribution<double, dist_type>(0., 1.) << "\t" << my_gen.distribution<double, dist_type>(0., 1.) << std::endl;
	std::cout << "Same? Good continue!\n\n";
}