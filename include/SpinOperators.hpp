#pragma once

#ifndef SPIN
	#define SPIN 0.5
	#pragma message("---> define spin value as pragma macro before including package")
#endif
const double _Spin = SPIN;

namespace operators{
    
    inline
    std::pair<cpx, u64> 
    sigma_x(u64 base_vec, int L, std::vector<int> sites) {
		for (auto& site : sites) 
			base_vec = flip(base_vec, BinaryPowers[L - 1 - site], L - 1 - site);
		return std::make_pair(_Spin, base_vec);
	};

    inline
	std::pair<cpx, u64> 
    sigma_y(u64 base_vec, int L, std::vector<int> sites) {
		auto tmp = base_vec;
		cpx val = 1.0;
		for (auto& site : sites) {
			val *= _Spin * (checkBit(tmp, L - 1 - site) ? im : -im);
			tmp = flip(tmp, BinaryPowers[L - 1 - site], L - 1 - site);
		}
		return std::make_pair(val, tmp);
	};
	
    inline
    std::pair<cpx, u64> 
    sigma_z(u64 base_vec, int L, std::vector<int> sites) {
		auto tmp = base_vec;
		double val = 1.0;
		for (auto& site : sites) 
			val *= checkBit(tmp, L - 1 - site) ? _Spin : -_Spin;
		return std::make_pair(val, base_vec);
	};

}