#pragma once

namespace single_particle{

    /// @brief Draw randomly many-body configurations from product states of single-particle states
    /// @param num_of_states number of states to draw
    /// @param volume volume of single particle system
    /// @param random_gen random generator
    /// @param filling filling fraction (probability for bit to be set); for filling < 0 the canonical ensemble is chosen
    /// @return 
    inline
    std::vector<boost::dynamic_bitset<>> mb_config_all(long int volume, double filling = 0.5)
    {
        std::vector<boost::dynamic_bitset<>> mb_states;
        int num_particles = int(filling * volume);

        std::vector<bool> ints(volume, 0);
        for (int i = 0; i < num_particles; i++)
            ints[i] = 1;
            
        std::sort(ints.begin(), ints.end());
        do{
            boost::dynamic_bitset<> state(volume);
            for(long j = 0; j < volume; j++)
                state[j] = ints[j];
            mb_states.emplace_back(state);
        }while(std::next_permutation(ints.begin(), ints.end()));
        
    //     auto n = ints.size();
    // #pragma omp for
    //     for (long i = 0; i < n; ++i) 
    //     {
    //         // Make a copy of v with the element at 'it' rotated to the beginning
    //         auto vprime = ints;
    //         std::rotate(vprime.begin(), vprime.begin() + i, vprime.begin() + i + 1);
    //         // The above guarantees that vprime[1:] is still sorted.
    //         // Since vprime[0] is constant, we only need to permute vprime[1:]
    //         while (std::next_permutation(vprime.begin() + 1, vprime.end()) ) 
    //         {
    //             boost::dynamic_bitset<> state(volume);
    //             for(long j = 0; j < volume; j++)
    //                 state[j] = vprime[j];
    //             mb_states.emplace_back(state);
    //         }
    //     }
        return mb_states;
    }

    /// @brief Draw randomly many-body configurations from product states of single-particle states
    /// @param num_of_states number of states to draw
    /// @param volume volume of single particle system
    /// @param random_gen random generator
    /// @param filling filling fraction (probability for bit to be set); for filling < 0 the canonical ensemble is chosen
    /// @return 
    inline
    std::vector<boost::dynamic_bitset<>> mb_config(u64 num_of_states, long int volume, disorder<double>& random_gen, double filling = 0.5)
    {
        std::vector<boost::dynamic_bitset<>> mb_states;

    #pragma omp parallel for
        for(u64 id = 0; id < num_of_states; id++){
            long num_up = int(volume * filling);
            long num_down = volume - num_up;

            boost::dynamic_bitset<> state(volume);
            for(long j = 0; j < volume; j++){
                float p = random_gen.random_uni<double>(0.0, 1.0);

                if(num_down == 0 || (p <= double(num_up) / double(num_up + num_down) && num_up > 0)){
                    num_up--;
                    state[j] = 1;
                } else {
                    num_down--;
                    // state.push_back(0);
                }
            }
            #pragma omp critical
            {
                mb_states.emplace_back(state);
            }
        } 
        
        // Search for repeating states
        std::vector<u64> indices;
    #pragma omp parallel for
        for(u64 n = 0; n < mb_states.size(); n++){
            for(u64 m = 0; m < mb_states.size() && m != n; m++){
                if(mb_states[m] == mb_states[n]){
                #pragma omp critical
                    {
                        indices.push_back(n);
                    }
                }
            }
        }
        // for(auto& idx : indices)
        //     printSeparated(std::cout, "\t", 16, "Found reapeating state", mb_states[idx]);
        std::sort(indices.begin(), indices.end());
        removeIndicesFromVector(mb_states, indices);
        return mb_states;
    }

    /// @brief Draw randomly many-body configurations for free fermions at energy E = 0 and Q = log[ (-1)^N ] = 0 v pi
    /// @param num_of_states number of states to draw
    /// @param volume volume of single particle system
    /// @param random_gen random generator
    /// @param filling filling fraction (probability for bit to be set); for filling < 0 the canonical ensemble is chosen
    /// @return 
    inline
    std::vector<boost::dynamic_bitset<>> mb_config_free_fermion(u64 num_of_states, long int volume, disorder<double>& random_gen, double filling = 0.5)
    {
        std::vector<boost::dynamic_bitset<>> mb_states;
        int num_particles = int(filling * volume);

    #pragma omp parallel for
        for(u64 id = 0; id < num_of_states; id++){
            arma::Col<int> positions = random_gen.create_random_vec<int>(num_particles / 2, volume / 4, 3 * volume / 4 - 1);
            boost::dynamic_bitset<> state(volume);
            // std::cout << positions.t();
            for(long j : positions){
                state[j] = 1;
                if( j > volume / 2 )        state[3 * volume / 2 - j] = 1;
                else if( j == volume / 4 )  state[3 * volume / 4    ] = 1;
                else                        state[    volume / 2 - j] = 1;
            }
            
            #pragma omp critical
            {
                mb_states.emplace_back(state);
            }
        }

        // Search for repeating states
        std::vector<u64> indices;
    #pragma omp parallel for
        for(u64 n = 0; n < mb_states.size(); n++){
            for(u64 m = 0; m < mb_states.size() && m != n; m++){
                if(mb_states[m] == mb_states[n]){
                #pragma omp critical
                    {
                        indices.push_back(n);
                    }
                }
            }
        }
        // for(auto& idx : indices)
        //     printSeparated(std::cout, "\t", 16, "Found reapeating state", mb_states[idx]);
        std::sort(indices.begin(), indices.end());
        removeIndicesFromVector(mb_states, indices);
        return mb_states;
    }


    namespace correlators{

        /// @brief Calculate one-body correlation matrix for eigenstate state with single particle states in orbitals. The size of matrix is set by VA
        /// @param orbitals single particle wavefunctions
        /// @param state many-body eigenstate (configuration -- product state as boost::dynamic_bitset)
        /// @param VA subsystem size (can be whole system)
        /// @param J_m reference to one-body correlation matrix to add new entries
        /// @param lambda reference to single-site one-body correlation
        /// @param prefactor prefactor for correlation matrix (by default = 1.0)
        inline
        void one_body(const arma::cx_mat& orbitals, const boost::dynamic_bitset<>& state, int VA, arma::cx_mat& J_m, cpx& lambda, double prefactor = 1.0)
        {
            arma::uvec col_idx(long(state.count()));
            int idx = 0;
            for(int q = 0; q < orbitals.n_cols; q++){
                double n_q = int(state[q]);
                lambda += prefactor * n_q * std::abs(orbitals(q, VA) * std::conj(orbitals(q, VA)));
                if(n_q == 1)
                    col_idx(idx++) = q;
            }
            if(VA > 0){
                arma::uvec row_idx = arma::regspace<arma::uvec>(0, VA-1);
                auto W = orbitals.submat(row_idx, col_idx);
                J_m += prefactor * W * W.t();
            }
            // for(int q = 0; q < orbitals.n_cols; q++){
            //     double n_q = int(state[q]);
            //     lambda += prefactor * n_q * std::abs(orbitals(q, VA) * std::conj(orbitals(q, VA)));
            //     if(VA > 0){
            //         auto orbital = orbitals.col(q).rows(0, VA - 1);
            //         J_m += prefactor * n_q * orbital * orbital.t();
            //     }
            // }
        }
    }


    namespace entanglement{

        /// @brief Calcute term in  sum for single lambda (also used when size of one-body correlation matrix is 1)
        /// @param lambda input eigenvalue of one-body correlation matrix
        /// @return entanglement entropy given by the eigenvalue of correlation matrix
        inline
        double vonNeumann_helper(double lambda){
            if(std::abs( std::abs(lambda) - 1.0 ) < 1e-14)
                return 0;
            else{
                double lam1 = (1 + lambda) / 2.;
                double lam2 = (1 - lambda) / 2.;
                return -lam1 * std::log(lam1) - lam2 * std::log(lam2); 
            }
        }


        /// @brief Calculate von Neumann entanglement entropy from eigenvalues of one-body correlation matrix
        /// @param corr_coeff eigenvalues of one-body correlation matrix
        /// @return entanglement entropy
        inline
        double vonNeumann(arma::vec corr_coeff){
            double S = 0;
            for(auto& lam : corr_coeff)
                S += vonNeumann_helper(lam);
            return S;
        }

    }
};