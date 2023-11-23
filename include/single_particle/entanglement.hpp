#pragma once

namespace QHS{
    namespace single_particle{

        namespace entanglement{

            /// @brief Calcute term in  sum for single lambda (also used when size of one-body correlation matrix is 1)
            /// @param lambda input eigenvalue of one-body correlation matrix
            /// @return entanglement entropy given by the eigenvalue of correlation matrix
            inline
            double vonNeumann_helper(double lambda){
                if(std::abs(lambda) > 1.0 - 1e-12)
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

}