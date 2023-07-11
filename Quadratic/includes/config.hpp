#pragma once 

// #define EXTRA_DEBUG
#define CONFIG 2

#ifndef DIM
    #define DIM 1
#endif


#ifndef MODEL
    #define MODEL 1
#endif


//------------------- Translate Macro

#if MODEL == 0
    #define ANDERSON
#elif MODEL == 1
    #define SYK
#else
    #define FREE_FERMIONS
#endif