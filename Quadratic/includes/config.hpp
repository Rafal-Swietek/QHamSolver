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
    #define print_model(x)  "Chosen Anderson model in " #x "-dimensions!"
    #define pprint_model(x) print_model(x)
#elif MODEL == 1
    #define SYK
    #define print_model(x) "Chosen SYK2 model with GOE matrix elements in " #x "-dimensions!"
    #define pprint_model(x) print_model(x)
#elif MODEL == 2
    #define AUBRY_ANDRE
    #define print_model(x) "Chosen Aubry-Andre model with phi=0 in " #x "-dimensions!"
    #define pprint_model(x) print_model(x)
#elif MODEL == 3
    #define FREE_FERMIONS
    #define print_model(x) "Chosen Free fermion model in " #x "-dimensions!"
    #define pprint_model(x) print_model(x)
#else
    #define print_model(x) "DEFAULT: Chosen SYK2 model with GOE matrix elements in " #x "-dimensions!"
    #define pprint_model(x) print_model(x)
#endif


#pragma message(pprint_model(DIM))