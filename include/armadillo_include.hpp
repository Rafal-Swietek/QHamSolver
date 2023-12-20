
DISABLE_WARNING_PUSH // include <armadillo> and suppress its warnings, cause developers suck
	// armadillo flags:
#define ARMA_64BIT_WORD // enabling 64 integers in armadillo obbjects
#define ARMA_BLAS_LONG_LONG // using long long inside LAPACK call
#define ARMA_USE_OPENMP
#define ARMA_ALLOW_FAKE_GCC
#define ARMA_USE_HDF5
//#define ARMA_USE_SUPERLU
//#define ARMA_EXTRA_DEBUG
//-------
DISABLE_OVERFLOW;
#if defined(_MSC_VER)
	DISABLE_WARNING(26812); // unscoped enum
	DISABLE_WARNING(26819); // unannotated fallthrough
	DISABLE_WARNING(26439); // may not throw
	DISABLE_WARNING(6011);  // dereferencing NULL ptr 
	DISABLE_WARNING(26495); // unitialized variable
	DISABLE_WARNING(6993);  // ignore OpenMP: use single-thread
	DISABLE_WARNING(4849);  // ignor OpenMP:collapse
#elif defined(__GNUC__) || defined(__clang__)
	DISABLE_WARNING(-Wenum-compare); // unscoped enum
	DISABLE_WARNING(-Wimplicit-fallthrough); // unannotated fallthrough
	DISABLE_WARNING(-Wuninitialized); // unitialized
	DISABLE_WARNING(-Wopenmp);  // ignore OpenMP warning
#else 
	#pragma message ("not recognized compiler to disable armadillo library warnings");
#endif

#define ARMA_OPENMP_THREADS 256
#include <armadillo>
#undef ARMA_USE_SUPERLU

DISABLE_WARNING_POP
