#pragma once

/// @brief Calculate elapsed time from start
/// @param start starting time_point to substract from current time and get distance
/// @return 
inline 
double tim_s(clk::time_point start) {
	return double(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::duration(\
		clk::now() - start)).count()) / 1000.0;
}
//-------------------------------------------------------------------------------------------------------------- OPERATION ON STRINGS


/// @brief Finds bit representation of number
/// @param num input number
/// @param size number of bits
/// @return bit representation
inline
std::string to_binary(u64 num, int size){
    std::string bin_num = "";
    while (num > 0)
    {
        int bin = num % 2;
        bin_num += std::to_string(bin);
        num /= 2;
    }
    std::reverse(bin_num.begin(), bin_num.end());
    if(bin_num.size() < size){
        bin_num = std::string(size - bin_num.size(), '0') + bin_num;
    }
    return bin_num;
}

/// @brief Transforms bit representation (as string) to decimal
/// @param bit_mask string bit represented number
/// @return decimal representation
inline
u64 to_int(std::string bit_mask){
	u64 bit_mask_num = 0;
	int L = bit_mask.size();
	for(int i = 0; i < L; i++)
		if(bit_mask[i] == '1')
			bit_mask_num += ULLPOW(L - 1 - i);
	return bit_mask_num;
}

/// @brief checking if string is a number
/// @param str 
/// @return 
inline 
bool isNumber(const std::string& str) {
	bool found_dot = false;
	bool found_minus = false;
	for (char const& c : str) {
		if (c == '.' && !found_dot) {
			found_dot = true;
			continue;
		}
		if (c == '-' && !found_minus) {
			found_minus = true;
			continue;
		}
		if (std::isdigit(c) == 0) return false;
	}
	return true;
}

/// @brief split string into pieces divided by 'delimeter' 
/// @param s 
/// @param delimiter 
/// @return vector of strings separated by delimieter
inline 
std::vector<std::string> 
split_str(
	std::string s,			//<! string to split
	std::string delimiter	//<! symbol serving as split value
) {
	size_t pos_start = 0, pos_end, delim_len = delimiter.length();
	std::string token;
	std::vector<std::string> res;

	while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
		token = s.substr(pos_start, pos_end - pos_start);
		pos_start = pos_end + delim_len;
		res.push_back(token);
	}

	res.push_back(s.substr(pos_start));
	return res;
}

/// @brief Abs function to remove ambiguous calls to std::abs
/// @tparam _ty template type
/// @param value input value
/// @return abs(value)
template <typename _ty>
inline
_ty my_abs(_ty value)
	{ return value >= 0? value : -value; }

/// @brief finds the order of magnitude of number +1 (only for <1 numbers to find filename format)
/// @tparam T 
/// @param a_value 
/// @return 
template <typename _ty>
inline
int order_of_magnitude(const _ty a_value) {
	if(a_value != 0){
		std::string num_str = std::to_string(a_value - int(a_value) );

        num_str.erase ( num_str.find_last_not_of('0') + 1, std::string::npos );
        num_str.erase ( num_str.find_last_not_of('.') + 1, std::string::npos );
		int len = num_str.find_last_of('.');

		num_str.erase(0,len+1);
		len = num_str.length();
        if(num_str == "0")
            len = 0;
		return len;
	}
	else return 0;
}

/// @brief Print value of any type to custom precision
/// @tparam _type template type of value
/// @param a_value argument to convert parse to stream
/// @param n precision/number of digits after comma
/// @return converted value to string with given precision
template <typename _type> 
inline 
std::string to_string_prec(
	_type a_value,
	int n = -1
) {
	if(my_abs(a_value) < 1e-10) a_value = _type(0);
	if(n < 0)	
		n = order_of_magnitude(a_value);
	std::ostringstream outie;
	outie.precision(n);
	outie << std::fixed << a_value;
	return outie.str();
}

//-------------------------------------------------------------------------------------------------------------- PERMUTATION AND SORTING OF DATA

/// @brief Sorts the vector and saves the permutation with a lambda like function compare
/// @tparam container template for any container class with access operator[]
/// @tparam F template for callable lambda to define sorting criterion
/// @param vec input vector to find permutation on
/// @param compare callable/invocable -- sorting criterion
/// @return permutation of the sortation (indices after shullfing of data)
template <has_access_operator container, callable_type F> 
inline 
std::vector<std::size_t> sort_permutation(
	const container& vec,
	F&& compare
) {
	std::vector<std::size_t> p(vec.size());
	std::iota(p.begin(), p.end(), 0);
	std::sort(p.begin(), p.end(),
		[&](std::size_t i, std::size_t j) {
			return compare(vec[i], vec[j]);
		});
	return p;
}

/// @brief Applies permutation on a given vector
/// @tparam container template for any container class with access operator[]
/// @param vec input vector to apply permutation on (as reference)
/// @param p permutation of the sortation
template <has_access_operator container> 
inline 
void apply_permutation(
	container& vec,			//<! vector to permute
	const std::vector<std::size_t>& p	//<! permutation on input vector
) {
	std::vector<bool> done(vec.size());
	for (std::size_t i = 0; i < vec.size(); ++i) {
		if (done[i]) continue;
		done[i] = true;
		std::size_t prev_j = i;
		std::size_t j = p[i];
		while (i != j) {
			std::swap(vec[prev_j], vec[j]);
			done[j] = true;
			prev_j = j;
			j = p[j];
		}
	}
}

//-------------------------------------------------------------------------------------------------------------- ADDITIONAL TOOLS

inline
std::complex<double> pow_im(unsigned int power){
	const int num = power % 4;
	switch(num){
		case 0: return std::complex<double>(1.0, 0.0);		//  1
		case 1: return std::complex<double>(0.0, 1.0);		//  i
		case 2: return std::complex<double>(-1.0, 0.0);		// -1
		case 3: return std::complex<double>(0.0, -1.0);		// -i
	default:
		_assert_(false, "Only integer positive powers. You fucked uo mate!");
		return 0.0;
	}
}

/// @brief Gets the sign of input value
/// @tparam T template parameter
/// @param val input value
/// @return returns sign of value
template <typename T> 
inline 
int sgn(T val) {
	return (T(0) < val) - (val < T(0));
}

/// @brief Overriding the ostream operator for pretty printing vectors.
/// @tparam _type template type for vector (needs to have outpu operator defined)
/// @param os output stream
/// @param vec input vector to print
/// @return output stream where vector is printed
template <has_output_operator _type> 
inline 
std::ostream& operator<<(
	std::ostream& os,				//<! output stream
	const std::vector<_type>& vec	//<! vector to print
) {
	if (vec.size() != 0) {
		std::copy(vec.begin(), vec.end() - 1, std::ostream_iterator<_type>(os, " "));
		os << vec.back() << '\n';
	}
	else
		os << "Empty container!" << std::endl;
	return os;
}
template <> 
inline 
std::ostream& operator<<(
	std::ostream& os,				//<! output stream
	const std::vector<bool>& vec	//<! vector to print
) {
	if (vec.size() != 0) {
		for(auto elem : vec)
			std::cout << elem;
	}
	else
		os << "Empty container!" << std::endl;
	return os;
}