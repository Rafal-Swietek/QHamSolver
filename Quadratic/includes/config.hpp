#pragma once 

// #define EXTRA_DEBUG
#undef NODEBUG
#define CONFIG 2

#ifndef DIM
    #define DIM 1
#endif


#ifndef MODEL
    #define MODEL 3
#endif


//------------------- Translate Macro

#if MODEL == 0
    #define ANDERSON
    #pragma message("Chosen Anderson model {DIM}-dimensional!")
#elif MODEL == 1
    #define SYK
    #pragma message("Chosen SYK2 model with GOE matrix elements in {DIM}-dimensions!")
#elif MODEL == 2
    #define AUBRY_ANDRE
    #pragma message("Chosen Aubry-Andre model with phi=0 in {DIM}-dimensions!")
#elif MODEL == 3
    #define FREE_FERMIONS
    #pragma message("Chosen Free fermion model {DIM}-dimensional!")
#else
    #pragma message("No model chosen!!! Leaving empty matrix")
#endif