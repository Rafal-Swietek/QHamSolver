arma::mat H(dim, dim, arma::fill::zeros);

    arma::vec states(dim);
    arma::vec E_dis = states;
//#pragma omp parallel for
    for(long k = 0; k < dim; k++){
        double ener = 0;
        states(k) = k;
        int fermion_num = __builtin_popcountll(k);
        for(int j = 0; j < this->L; j++){
            int nei = (j + 1) % this->L;
            int sign = ( (nei == 0) && ( (fermion_num - 1) % 2) )? -1 : 1; // jump over fermions except first and last site
            bool ci = checkBit(k, j);
            bool ci_1 = checkBit(k, nei);
            if( ci )
                ener -= std::cos(two_pi * (j + 0.0) / double(this->L));
            if(ci == 1 && ci_1 == 0){
                auto [val1, op_k] = operators::sigma_x(k, this->L, { j });
                auto [val2, opop_k] = operators::sigma_x(op_k, this->L, { nei });
                H(opop_k, k) += sign / 2.0;
            }
            if(ci == 0 && ci_1 == 1){
                auto [val1, op_k] = operators::sigma_x(k, this->L, { j });
                auto [val2, opop_k] = operators::sigma_x(op_k, this->L, { nei });
                H(opop_k, k) += sign / 2.0;
            }
        }
        E_dis(k) = ener;
    }
    Efull = arma::eig_sym(H);