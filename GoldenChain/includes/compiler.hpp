#pragma once

#ifdef USE_SYMMETRIES
    #pragma message ("Using model with symmetries as default")
    #ifdef USE_REAL_SECTORS
        #pragma message ("Using real matrix elements. BEWARE: Only applicable for real sectors (k=0, pi) !!!")
    #endif
#endif