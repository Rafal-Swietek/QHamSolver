#pragma once 

// #define EXTRA_DEBUG
// #define NODEBUG

#define SPIN 1    //<! value of spin (1/2 -> spin operators, 1 -> pauli matrices)
#define CONFIG 2    //<! on-site configuration (local hilbert space dimension)


#if defined(MY_MAC) // use only on personal device

    //<! Macro to set element type to double for real momentum sectors
    #ifndef USE_REAL_SECTORS
        #define USE_REAL_SECTORS
    #endif

#endif


#include "compiler.hpp"