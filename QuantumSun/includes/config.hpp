#pragma once 

// #define NODEBUG
// #define EXTRA_DEBUG

#define SPIN 0.5

#ifdef MY_MAC
    #ifndef CONF_DISORDER
        // #define CONF_DISORDER
    #endif

    #ifndef SCALED_DISORDER
        // #define SCALED_DISORDER
    #endif
#endif



//<! ------------------------------------------------- translate macros to variables
#ifdef CONF_DISORDER
    #pragma message("Using configurational disorder!")
    constexpr bool conf_disorder = 1;
#else
    constexpr bool conf_disorder = 0;
#endif

#ifdef SCALED_DISORDER
    #pragma message("Using disorder scaled by L (number of spins)!")
    constexpr bool scaled_disorder = 1;
#else
    constexpr bool scaled_disorder = 0;
#endif