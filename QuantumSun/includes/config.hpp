#pragma once 

#define EXTRA_DEBUG
#define SPIN 0.5

#ifndef CONF_DISORDER
    #define CONF_DISORDER 0
#endif





//<! ------------------------------------------------- translate macros to variables
#if CONF_DISORDER == 1
    #pragma message("Using configurational disorder!")
#endif
constexpr bool conf_disorder = CONF_DISORDER;