#pragma once


// TODO: 
//  Change input array with coefficients to lambda. Same in preamble.hpp


namespace polfed{

    namespace clenshaw{

        /// @brief Calculate a polynomial spanned in the Chebyshev basis using Clenchaw algorithm
        /// @tparam _ty template parameter of matrix elements
        /// @param K order of polynomial
        /// @param coeff array of coefficients (of size K+1)
        /// @param matrix matrix to transform (needs to be square)
        /// @return transformed matrix P^K(matrix)
        template <typename _ty>
        inline 
        arma::SpMat<_ty>
        chebyshev(int K, const arma::Col<_ty>& coeff, const arma::SpMat<_ty>& matrix)
        {
            #ifndef NODEBUG
                _assert_(coeff.size() == K+1, "\t\tDimension mismatch: Array of coefficients has to be of length K+1");
                _assert_(matrix.n_rows == matrix.n_cols, "\t\tInput matrix is note square. Why would you transform non-square matrix? Wtf?");
            #endif
            // initialize recurence objects
            arma::SpMat<_ty> bk_2(matrix.n_cols, matrix.n_cols);
            arma::SpMat<_ty> bk_1(matrix.n_cols, matrix.n_cols);
            arma::SpMat<_ty> bk(matrix.n_cols, matrix.n_cols), eye = arma::eye<arma::SpMat<_ty>>(matrix.n_cols, matrix.n_cols);
            // perform recurence relation in loop
            for(int n = K; n > 0; n--){
                // _extra_debug_( auto start = std::chrono::system_clock::now(); )
                bk = bk_1 * matrix;
                // matmul(bk, bk_1, matrix);
                bk = coeff(n) * eye + 2.0 * bk - bk_2;
                bk_2 = bk_1;
                bk_1 = bk;
                // _extra_debug( std::cout << "Clenchaw: "; printSeparated(std::cout, "\t", 20, true, double(bk.n_nonzero) / double(bk.n_rows * bk.n_cols), matrix_size(bk.n_nonzero * sizeof(bk(0, 0))), n, coeff(n), tim_s(start)); )
            }
            return coeff(0) * eye + bk_1 * matrix - bk_2;
        }

        /// @brief Calculate a polynomial spanned in the Chebyshev basis using Clenchaw algorithm
        /// @tparam _ty template parameter of matrix elements
        /// @param K order of polynomial
        /// @param coeff array of coefficients (of size K+1)
        /// @param values array of values
        /// @return transformed P^K(values)
        template <typename _ty>
        inline 
        arma::Col<_ty>
        chebyshev(int K, const arma::Col<_ty>& coeff, const arma::Col<_ty>& values)
        {
            #ifndef NODEBUG
                _assert_(coeff.size() == K+1, "\t\tDimension mismatch: Array of coefficients has to be of length K+1");
            #endif
            // initialize recurence objects
            auto bk_2 = arma::Col<_ty>(values.size());
            auto bk_1 = arma::Col<_ty>(values.size());

            // perform recurence relation in loop
            for(int n = K; n > 0; n--){
                // _extra_debug_( auto start = std::chrono::system_clock::now(); )
                auto bk = coeff(n) + 2.0 * (bk_1 * values) - bk_2;
                bk_2 = bk_1;
                bk_1 = bk;
                // _extra_debug( std::cout << "Clenchaw: "; printSeparated(std::cout, "\t", 20, true, n, coeff(n), tim_s(start)); )
            }
            return coeff(0) + bk_1 * values - bk_2;
        }

        /// @brief Calculate a polynomial spanned in the Chebyshev basis using Clenchaw algorithm
        /// @tparam _ty template parameter of matrix elements
        /// @param K order of polynomial
        /// @param coeff array of coefficients (of size K+1) 
        /// @param sigma value to get polynomial
        /// @return transformed P^K(sigma)
        template <typename _ty>
        inline
        _ty
        chebyshev(int K, const arma::Col<_ty>& coeff, _ty sigma)
        {
            #ifndef NODEBUG
                _assert_(coeff.size() == K+1, "\t\tDimension mismatch: Array of coefficients has to be of length K+1");
            #endif
            // initialize recurence objects
            double bk_2 = 0, bk_1 = 0;

            // perform recurence relation in loop
            for(int n = K; n > 0; n--){
                auto bk = coeff(n) + 2.0 * (bk_1 * sigma) - bk_2;
                bk_2 = bk_1;
                bk_1 = bk;
            }
            return coeff(0) + bk_1 * sigma - bk_2;
        }

        /// @brief Calculate a product polynomial P^K(H) spanned in the Chebyshev  asis with input state
        /// @tparam _ty template parameter of matrix elements
        /// @param K order of polynomial
        /// @param coeff array of coefficients (of size K+1)
        /// @param matrix matrix to 'transform' (needs to be square)
        /// @param state state to act with P^K(H)
        /// @return transformed matrix-vector product P^K(matrix) * state
        template <typename _ty>
        inline 
        arma::Col<_ty>
        chebyshev(int K, const arma::Col<_ty>& coeff, const arma::SpMat<_ty>& matrix, const arma::Col<_ty>& state)
        {
            #ifndef NODEBUG
                _assert_(coeff.size() == K+1, "\t\tDimension mismatch: Array of coefficients has to be of length K+1");
                _assert_(matrix.n_rows == matrix.n_cols, "\t\tInput matrix is note square. Why would you transform non-square matrix? Wtf?");
                _assert_(matrix.n_cols == state.size(), "\t\tDimension mismatch: State to act on is not equalt to number of columns in matrix");
            #endif
            // initialize recurence objects
            arma::Col<_ty> bk_2(state.size(), arma::fill::zeros);
            arma::Col<_ty> bk_1(state.size(), arma::fill::zeros);
            arma::Col<_ty> bk  (state.size(), arma::fill::zeros);
            // perform recurence relation in loop
            for(int n = K; n > 0; n--){
                // _extra_debug_( auto start = std::chrono::system_clock::now(); )
                bk = coeff(n) * state + 2.0 * matrix * bk_1 - bk_2;
                bk_2 = bk_1;
                bk_1 = bk;
                // _extra_debug( std::cout << "Clenchaw: "; printSeparated(std::cout, "\t", 20, true, n, coeff(n), tim_s(start)); )
            }
            return coeff(0) * state + matrix * bk_1 - bk_2;
        }

        /// @brief Calculate a product polynomial P^K(H) spanned in the Chebyshev  asis with input states
        /// @tparam _ty template parameter of matrix elements
        /// @param K order of polynomial
        /// @param coeff array of coefficients (of size K+1)
        /// @param matrix matrix to 'transform' (needs to be square)
        /// @param states states to act with P^K(H)
        /// @return transformed matrix-matrix product P^K(matrix) * states
        template <typename _ty>
        inline 
        arma::Mat<_ty>
        chebyshev(int K, const arma::Col<_ty>& coeff, const arma::SpMat<_ty>& matrix, const arma::Mat<_ty>& states)
        {
            #ifndef NODEBUG
                _assert_(coeff.size() == K+1, "\t\tDimension mismatch: Array of coefficients has to be of length K+1");
                _assert_(matrix.n_rows == matrix.n_cols, "\t\tInput matrix is note square. Why would you transform non-square matrix? Wtf?");
                _assert_(matrix.n_cols == states.n_rows, "\t\tDimension mismatch: State to act on is not equalt to number of columns in matrix");
            #endif
            // initialize recurence objects
            arma::Mat<_ty> bk_2(states.n_rows, states.n_cols, arma::fill::zeros);
            arma::Mat<_ty> bk_1(states.n_rows, states.n_cols, arma::fill::zeros);
            arma::Mat<_ty> bk  (states.n_rows, states.n_cols, arma::fill::zeros);
            // perform recurence relation in loop
            // const int max_th = num_of_threads;
            for(int n = K; n > 0; n--){
                // _extra_debug_( auto start = std::chrono::system_clock::now(); )
                _mult_sp_and_mat(matrix, bk_1, bk, num_of_threads);
                bk = coeff(n) * states + 2.0 * bk - bk_2;
                // bk = coeff(n) * states + 2.0 * matrix * bk_1 - bk_2;
                bk_2 = bk_1;
                bk_1 = bk;
                // _extra_debug( std::cout << "Clenchaw: "; printSeparated(std::cout, "\t", 20, true, n, coeff(n), tim_s(start)); )
            }
            _mult_sp_and_mat(matrix, bk_1, bk, num_of_threads);
            return coeff(0) * states + matrix * bk - bk_2;
            // return coeff(0) * states + matrix * bk_1 - bk_2;
        }
        
    }

}