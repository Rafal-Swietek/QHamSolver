#pragma once

namespace single_particle{

    /// @brief Draw randomly many-body configurations from product states of single-particle states
    /// @param num_of_states number of states to draw
    /// @param volume volume of single particle system
    /// @param random_gen random generator
    /// @param num_particles number of particles
    /// @return 
    inline
    std::vector<boost::dynamic_bitset<>> mb_config_all(long int volume, int num_particles = 1)
    {
        std::vector<boost::dynamic_bitset<>> mb_states;

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

        return mb_states;
    }

    /// @brief Draw randomly many-body configurations from product states of single-particle states
    /// @param num_of_states number of states to draw
    /// @param volume volume of single particle system
    /// @param random_gen random generator
    /// @param num_particles number of particles
    /// @return 
    inline
    std::vector<boost::dynamic_bitset<>> mb_config(u64 num_of_states, long int volume, disorder<double>& random_gen, int num_particles = 1)
    {
        std::vector<boost::dynamic_bitset<>> mb_states;
        num_of_states = u64(std::min((double)num_of_states, binom(volume, num_particles)));
    #pragma omp parallel for
        for(u64 id = 0; id < num_of_states; id++){
            long num_up = num_particles;
            long num_down = volume - num_up;

            boost::dynamic_bitset<> state(volume);
            for(long j = 0; j < volume; j++){
                float p = random_gen.random_uni<double>(0.0, 1.0);

                if(num_up == 0 && num_down == 0)    continue;
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
            for(u64 m = n+1; m < mb_states.size() && m != n; m++){
                if(mb_states[m] == mb_states[n]){
                #pragma omp critical
                    {
                        indices.push_back(m);
                    }
                }
            }
        }
        std::sort(indices.begin(), indices.end());
        auto last = std::unique(indices.begin(), indices.end());
        // indices now holds {1 2 3 4 5 x x}, where 'x' is indeterminate
        indices.erase(last, indices.end());

        removeIndicesFromVector(mb_states, indices);
        return mb_states;
    }

    /// @brief Draw randomly many-body configurations for free fermions at energy E = 0 and Q = log[ (-1)^N ] = 0 v pi
    /// @param num_of_states number of states to draw
    /// @param volume volume of single particle system
    /// @param random_gen random generator
    /// @param num_particles number of particles
    /// @return 
    inline
    std::vector<boost::dynamic_bitset<>> mb_config_free_fermion(u64 num_of_states, long int volume, int num_particles = 1)
    {
        std::vector<boost::dynamic_bitset<>> mb_states;

        std::vector<bool> ints(volume, 0);
        for (int i = 0; i < num_particles; i++)
            ints[i] = 1;
            
        std::sort(ints.begin(), ints.end());
        do{
            boost::dynamic_bitset<> state(volume);
            double E = 0;
            int Q = 0, N = 0;
            for(long q = 0; q < volume; q++){
                double n_q = int(ints[q]);
				if( n_q ){
				    E += 2*std::cos(two_pi * q / double(volume));
					Q += q;
					N++;
				}
                state[q] = ints[q];
            }
            if( (N == num_particles) && (Q % volume == 0) && (std::abs(E) < 1e-12) )
                mb_states.emplace_back(state);
        }while(std::next_permutation(ints.begin(), ints.end()));
        return mb_states;
    }

}