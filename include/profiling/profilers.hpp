#pragma once

#ifndef _CPU_PROFILE
    #include "cpu.hpp"
#endif
#ifndef _MEMORY_PROFILE
    #include "memory.hpp"
#endif

/// @brief Print profiling info (CPU Load, Memory usage) at given point in programme
inline void
_current_profiling_info()
{
    auto [free_mem, used_mem] = _profile_memory_();
    std::cout << "\n----------------------------------------" << std::endl;
    std::cout << "Current Free Memory:\t\t" << translate_bytes(free_mem) << std::endl;
    std::cout << "Current Memory usage:\t\t" << translate_bytes(used_mem) << std::endl;
    std::cout << "Current CPU Load:\t\t" << 100 * _profile_cpu_() << "%" << std::endl;
    std::cout << "----------------------------------------\n" << std::endl;
}