#pragma once 

// #define EXTRA_DEBUG

#define SPIN 0.5

#undef NODEBUG
#define CONFIG 2

#ifndef DIM
    #define DIM 1
#endif


#ifndef MODEL
    #define MODEL 5
#endif

//------------------- Translate Macro

#if MODEL == 0
    #define ANDERSON
    #define print_model(x)  "Chosen Anderson model in " #x "-dimensions!"
    #define pprint_model(x) print_model(x)
    const auto model = "Anderson";
#elif MODEL == 1
    #define SYK
    #define print_model(x) "Chosen SYK2 model with GOE matrix elements!"
    #define pprint_model(x) print_model(x)
    const auto model = "SYK2";
#elif MODEL == 2
    #define AUBRY_ANDRE
    #define print_model(x) "Chosen Aubry-Andre model with phi=0 in " #x "-dimensions!"
    #define pprint_model(x) print_model(x)
    const auto model = "AubryAndre";
#elif MODEL == 3
    #define FREE_FERMIONS
    #define print_model(x) "Chosen Free fermion model in " #x "-dimensions!"
    #define pprint_model(x) print_model(x)
    const auto model = "FreeFermions";
#elif MODEL == 4
    #define PLRB
    #define print_model(x) "Chosen power-law random banded (PLRB) model with GOE matrix elements in!"
    #define pprint_model(x) print_model(x)
    const auto model = "PLRB";
#elif MODEL == 5
    #define RP
    #define print_model(x) "Chosen Rozenzweig-Porter (RP) model with GOE matrix elements!"
    #define pprint_model(x) print_model(x)
    const auto model = "RP";
#else
    #define print_model(x) "DEFAULT: Chosen SYK2 model with GOE matrix elements in " #x "-dimensions!"
    #define pprint_model(x) print_model(x)
    const auto model = "SYK2";
#endif


#pragma message(pprint_model(DIM))