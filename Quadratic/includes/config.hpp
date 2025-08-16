#pragma once 

// #define EXTRA_DEBUG

#define SPIN 0.5

#undef NODEBUG
#define CONFIG 2

#ifndef _MAT_ENSEMBLE_
    #define _MAT_ENSEMBLE_ 0
#endif
constexpr int _mat_ensemble = _MAT_ENSEMBLE_;

#if _MAT_ENSEMBLE_ == 0
    #define ENSEMBLE GOE
    #define print_ensemble  "Chosen GOE ensemble!"
    const auto _ensemble_ = "GOE";
#elif _MAT_ENSEMBLE_ == 1
    #define ENSEMBLE GUE
    #define print_ensemble  "Chosen GUE ensemble!"
    const auto _ensemble_ = "GUE";
#elif _MAT_ENSEMBLE_ == 2
    #define ENSEMBLE CUE
    #define print_ensemble  "Chosen CUE ensemble! Haar random distirbuted coefficients"
    const auto _ensemble_ = "CUE";
#else
    static_assert(false, "Not chosen any random ensemble! Chosse GOE (0), GUE (1) oe CUE (2)!");
#endif

#ifndef DIM
    #define DIM 3
#endif


#ifndef MODEL
    #define MODEL 0
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
#pragma message(print_ensemble)