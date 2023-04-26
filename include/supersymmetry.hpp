
namespace susy{

    /// @brief Create supercharge from local charge for periodic boundary conditions
    /// @tparam elem_ty matrix-element type (given by local charge)
    /// @param size system size (number of spins or particles)
    /// @param q local supercharge
    /// @param boundary_cond choose boundary condition (PBC = 0, OBC = 1)
    /// @param _hilbert_space_1 hilbert space acting on with supercharge
    /// @param _hilbert_space_2 resulting hilbert space after acting (supercharge changes symmetry sector)
    /// @return supercharge (sparse matrix)
    template <typename elem_ty>
    inline
    arma::SpMat<elem_ty> 
    create_supercharge(int size, arma::SpMat<elem_ty> q, bool boundary_cond,
                            point_symmetric _hilbert_space_1, point_symmetric _hilbert_space_2) 
    {
        
        //auto check_spin = op::__builtins::get_digit(size);
        const u64 dim1 = ULLPOW(size);
        const u64 dim2 = ULLPOW(size + 1);

        arma::SpMat<elem_ty>  supercharge(dim2, dim1);
        for(int j = 0; j < size; j++){
            u64 dim_left = ULLPOW(j);
            u64 dim_right = ULLPOW(size - j - 1);
            arma::SpMat<elem_ty>  ham = arma::kron(arma::eye<arma::SpMat<elem_ty>>(dim_right, dim_right), arma::kron(q, arma::eye<arma::SpMat<elem_ty>>(dim_left, dim_left)));
            supercharge += (j % 2 == 0)? -ham : ham;
        }
        // for(long k = 0; k < dim1; k++){
        //     auto base_state = k;
        //     for(int j = 0; j <= size; j++){
        //         int site = j;
        //         int sign = j % 2 == 1? 1 : -1;
        //         if(j == size){
        //             site = size - 1;
        //             sign = (size % 2 == 0)? -1 : 1;
        //         }
        //         arma::vec spin = check_spin(base_state, site)? up : down;
        //         arma::vec state = q * spin;
        //       
        //         for(int ii = 0; ii < state.size(); ii++){
        //             if(std::abs(state(ii)) > 0)
        //             {
        //                 const u64 right = base_state % ULLPOW(size - site);
        //                 const u64 left = base_state - right;
        //
        //                 const u64 _add = ii * ULLPOW(size - site - 1);
        //
        //                 u64 new_idx = 2 * left + right + _add;
        //                 if(j == size){
        //                     new_idx = std::get<0>( T_op(new_idx) );
        //                 }
        //
        //                 printSeparated(std::cout, "\t", 16, true, to_binary(base_state, size), site, left, right, _add, to_binary(new_idx, size + 1));
        //                
        //                 supercharge(new_idx, k) += state(ii) * double(sign); // define in same basis as other charge
        //             }
        //         }
        //     }
        // }
        
        arma::SpMat<elem_ty> U1 = (_hilbert_space_1.symmetry_rotation());
        arma::SpMat<elem_ty> U2 = (_hilbert_space_2.symmetry_rotation());
        if(boundary_cond)
            return U2.t() * supercharge * U1 / std::sqrt(2);
        else 
        {
        
        #if defined(USE_REAL_SECTORS) || !defined(USE_SYMMETRIES)
            arma::sp_mat T = arma::real(op::_translation_symmetry(size + 1, 0).to_matrix(dim2));
        #else
            arma::sp_cx_mat T = (op::_translation_symmetry(size + 1, 0).to_matrix(dim2));
        #endif
            u64 dim_rest = ULLPOW(size - 1);
            arma::SpMat<elem_ty> Q0 = arma::kron(arma::eye<arma::SpMat<elem_ty>>(dim_rest, dim_rest), q);
            supercharge += T * Q0 * ( (size % 2 == 0)? -1.0 : 1.0);
            return std::sqrt(size / (size + 1.0)) * U2.t() * supercharge * U1 / std::sqrt(2);
        }
    };
    

    /// @brief Generate supersymmetric hamiltonian from local supercharge
    /// @tparam elem_ty matrix-element type (given by local charge)
    /// @param size system size
    /// @param q local supercharge
    /// @param boundary_cond choose boundary condition (PBC = 0, OBC = 1)
    /// @param _hilbert_space hilbert space of the model
    /// @return hamiltonian (sparse matrix) 
    template <typename elem_ty>
    inline
    arma::SpMat<elem_ty> 
    hamiltonian(int size, arma::SpMat<elem_ty>  q, bool boundary_cond, point_symmetric _hilbert_space) 
    {

        u64 dim = ULLPOW(size);
        arma::SpMat<elem_ty>  H(dim, dim);
        arma::SpMat<elem_ty>  e(2,2);   e(0,0) = 1.0;   e(1,1) = 1.0;
        arma::SpMat<elem_ty>  ham;
        for(int j = 0; j < size - 1; j++){
            ham = -arma::kron(e, q.t()) * arma::kron(q, e) - arma::kron(q.t(), e) * arma::kron(e, q) 
                                + q*q.t() + 1. / 2. * (arma::kron(e, q.t() * q) + arma::kron(q.t() * q, e));
            u64 dim_left = ULLPOW(j);
            u64 dim_right = ULLPOW(size - j - 2);
            ham = arma::kron(arma::kron(arma::eye<arma::SpMat<elem_ty> >(dim_left, dim_left), ham), arma::eye<arma::SpMat<elem_ty> >(dim_right, dim_right));
            H += ham;
        }
        if(boundary_cond){
            u64 dim_rest = ULLPOW(size - 1);
            ham = 1. / 2. * ( arma::kron(arma::eye<arma::SpMat<elem_ty> >(dim_rest, dim_rest), q.t() * q) + arma::kron(q.t() * q, arma::eye<arma::SpMat<elem_ty> >(dim_rest, dim_rest)) );
            return (H + (ham)) / 2.0;
        } else {
            int k_sec = size % 2 == 0? size / 2 : 0;
            auto U = _hilbert_space.symmetry_rotation();
        #if defined(USE_REAL_SECTORS) || !defined(USE_SYMMETRIES)
            arma::sp_mat T = arma::real(op::_translation_symmetry(size, k_sec).to_matrix(dim));
        #else
            arma::sp_cx_mat T = (op::_translation_symmetry(size, k_sec).to_matrix(dim));
        #endif

            H += T * ham * T.t();
            return U.t() * H * U / 2.0;// * std::sqrt(double(size) / double(size + 1.0));
        }
    };
};