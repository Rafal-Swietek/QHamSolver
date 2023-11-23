#pragma once

/// @brief Handle input exception and print appropriate message
/// @param eptr pointer to thrown exception
/// @param message message to additionally print after exception thrown
inline 
void handle_exception(std::exception_ptr eptr, const std::string& message) {
	try {
		if (eptr) {
			std::rethrow_exception(eptr);
		}
	}
	catch (const std::runtime_error& err){
		_assert_(false, "Runtime error:\t" + std::string(err.what()) + ":\t" + message);
	} catch (const std::bad_alloc& err){
		_assert_(false, "Bad alloc error:\t" + std::string(err.what()) + ":\t" + message);
	} catch (const std::exception& err) {
		_assert_(false, "Exception:\t" + std::string(err.what()) + ":\t" + message);
	} catch (...) {
		_assert_(false, "Unknown error...! " + message);
	}
}


/// @brief Handler for allocating matrix (catches out_of_memory issues)
/// @tparam _ty template type for matrix elements
/// @tparam fill_type template for armadillo fill_type of matrix
/// @param matrix reference to matrix
/// @param name name of referenced to matrix
/// @param n_rows number of rows to allocate
/// @param n_cols number of columns to allocate
/// @param fill fill type of matrix
template <typename _ty, typename fill_type>
inline
void try_allocation(arma::Mat<_ty>& matrix, const std::string& name, u64 n_rows, u64 n_cols, 
						const arma::fill::fill_class<fill_type>& fill)
{
	try{
		matrix = arma::Mat<_ty>(n_rows, n_cols, fill);
	} catch (...) {
		handle_exception(std::current_exception(),
			"\nobject = " + name
            + "\nmemory size = " + matrix_size(n_cols * n_rows * sizeof(matrix(0, 0)))
        );
	}
}
/// @brief Handler for allocating matrix (catches out_of_memory issues). By default initializes matrix with zeros
/// @tparam _ty template type for matrix elements
/// @param matrix reference to matrix
/// @param name name of referenced to matrix
/// @param n_rows number of rows to allocate
/// @param n_cols number of columns to allocate
template <typename _ty>
inline
void try_allocation(arma::Mat<_ty>& matrix, const std::string& name, u64 n_rows, u64 n_cols)
{
	try{
		matrix = arma::Mat<_ty>(n_rows, n_cols, arma::fill::zeros);
	} catch (...) {
		handle_exception(std::current_exception(),
			"\nobject = " + name
            + "\nmemory size = " + matrix_size(n_cols * n_rows * sizeof(matrix(0, 0)))
        );
	}
}
/// @brief Handler for allocating matrix (catches out_of_memory issues). By default initializes matrix with zeros
/// @tparam _ty template type for matrix elements
/// @param matrix reference to matrix
/// @param name name of referenced to matrix
/// @param n_rows number of rows to allocate
/// @param n_cols number of columns to allocate
template <typename _ty>
inline
void try_allocation(arma::Col<_ty>& vector, const std::string& name, u64 size)
{
	try{
		vector = arma::Col<_ty>(size, arma::fill::zeros);
		_extra_debug( std::cout << "\nobject = " << name 
						<< std::endl << "memory size = " + matrix_size(size * sizeof(vector(0))); )
	} catch (...) {
		handle_exception(std::current_exception(),
			"\nobject = " + name
            + "\nmemory size = " + matrix_size(size * sizeof(vector(0)))
        );
	}
}

// #define try_alloc_zeros(matrix, n_rows, n_cols) try_allocation(matrix, #matrix, n_rows, n_cols);
// #define try_alloc_fill(matrix, n_rows, n_cols, fill) try_allocation(matrix, #matrix, n_rows, n_cols, fill);

/// Macro to allocate matrix with known variable name (make general to include otehr fill-types)
#define try_alloc_matrix(matrix, n_rows, n_cols)	try_allocation(matrix, std::string(stringize(matrix)), n_rows, n_cols);
#define try_alloc_vector(vector, size)				try_allocation(vector, std::string(stringize(vector)), size);