
#ifndef QHamSolver_INCLUDES
#define QHamSolver_INCLUDES
//-----------------------

#include "compiler_setup_headers/preprocessor_setup.hpp"
#include "compiler_setup_headers/compiler_setup.hpp"

// BOOST
#include <boost/dynamic_bitset.hpp>

// OTHER
#include <string>

#include <algorithm>
#include <complex>
#include <vector>
#include <string>
#include <unordered_map>
#include <iterator>
#include <numeric>
#include <omp.h>

#include <cassert> // assert terminates program
#include <utility> // auto, etc.
#include <memory> // smart ptr
#include <thread>
#include <future>

#include "armadillo_include.hpp"

#include "miscaleneous/constants.hpp"
#include "compiler_setup_headers/typedefs.hpp"

#include "metaprograming/traits.hpp"
#include "metaprograming/structures.hpp"
#include "miscaleneous/tools.hpp"


//<! 
#include "armadillo_wrapper.hpp"

#include "I_O_streaming/stream_wrapper_base.hpp"
#include "headers_to_split.h"

#include "runtime_checks/error_handling.hpp"

#include "gcem/include/gcem.hpp"        //<! constexpr library for math functinos


//<! PROFILING TOOLS
#include "profiling/memory.hpp"
#include "profiling/cpu.hpp"
#include "profiling/profilers.hpp"

//<! MAIN HEADERS
#include "SpinOperators.hpp"
#include "hilbert_space/_base.hpp"
#include "operators/operator_base.h"

#include "Hamiltonian/_base.hpp"
#include "QhamSolver_aux.hpp"

#include "random_and_disorder/disorder.hpp"


#endif