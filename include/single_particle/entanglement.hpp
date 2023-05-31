#pragma once

namespace entanglement{

    /// @brief Draw randomly many-body configurations from product states of single-particle states
    /// @param num_of_states number of states to draw
    /// @param volume volume of single particle system
    /// @param random_gen random generator
    /// @param filling filling fraction (probability for bit to be set); for filling < 0 the canonical ensemble is chosen
    /// @return 
    inline
    std::vector<std::string> mb_configurations(u64 num_of_states, long int volume, disorder<double>& random_gen, double filling = 0.5)
    {
        std::vector<std::string> mb_states;
    #pragma omp parallel for
        for(u64 id = 0; id < num_of_states; id++){
            long num_up = int(volume * filling);
            long num_down = volume - num_up;

            std::string state = "";
            for(long j = 0; j < volume; j++){
                float p = random_gen.random_uni<double>(0.0, 1.0);

                if(num_down == 0 || (p <= double(num_up) / double(num_down) && num_up > 0)){
                    num_up--;
                    state += "1";
                } else {
                    num_down--;
                    state += "0";
                }
            }
            #pragma omp critical
            {
                mb_states.emplace_back(state);
            }
        }
        return mb_states;
    }

    namespace entropy{

        /// @brief 
        /// @param lambda 
        /// @return 
        inline
        double vonNeumann_helper(double lambda){ 
            double lam1 = (1 + lambda) / 2.;
            double lam2 = (1 - lambda) / 2.;
            return -lam1 * std::log(lam1) - lam2 * std::log(lam2); 
        }


        /// @brief 
        /// @param corr_coeff 
        /// @return 
        inline
        double vonNeumann(arma::vec corr_coeff){
            double S = 0;
            for(auto& lam : corr_coeff)
                S += vonNeumann_helper(lam);
            return S;
        }

    }
};