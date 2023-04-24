#pragma once 

//#define EXTRA_DEBUG

#define SPIN 0.5    //<! value of spin (1/2 -> spin operators, 1 -> pauli matrices)
#define CONFIG 2    //<! on-site configuration (local hilbert space dimension)

#if defined(MY_MAC) // use only on personal device

    //<! Macro to control if main routines are set for symmetric or non-symmetric model
    #ifndef USE_SYMMETRIES
        #define USE_SYMMETRIES
    #endif

    //<! Macro to set element type to double for real momentum sectors
    #ifndef USE_REAL_SECTORS
        #define USE_REAL_SECTORS
    #endif

#endif


#include "compiler.hpp"