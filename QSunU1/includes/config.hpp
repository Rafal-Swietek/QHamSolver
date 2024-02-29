#pragma once 

// #define EXTRA_DEBUG
#ifndef SPIN
    #define SPIN 0.5
#endif

#ifndef SCALED_DISORDER
    // #define SCALED_DISORDER
#endif



//<! ------------------------------------------------- translate macros to variables
#ifdef SCALED_DISORDER
    #pragma message("Using disorder scaled by L (number of spins)!")
    constexpr bool scaled_disorder = 1;
#else
    constexpr bool scaled_disorder = 0;
#endif

// #ifdef SPIN
//     constexpr bool _spin = SPIN;
// #endif