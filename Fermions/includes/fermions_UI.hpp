#pragma once

#ifndef UI
#define UI

#include "config.hpp"

#include "../../include/user_interface_sym.hpp"
#include "fermions.hpp"
#include "../../include/single_particle/correlators.hpp"
#include "../../include/single_particle/entanglement.hpp"


// ----------------------------------------------------------------------------- UI QUANTUM SUN -----------------------------------------------------------------------------

namespace Fermions_UI{
    class ui : public user_interface_sym<Fermions>{
        
    protected:
        double t1, t1s;
        double t2, t2s;
        double V1, V1s;
        double V2, V2s;
        double mu;
        int t1n, t2n, V1n, V2n;
        
        struct {
            int k_sym;
            int p_sym;
            int zx_sym;
            int N;
        } syms;
        
        // bool add_edge_fields;
        bool add_parity_breaking;

        /// @brief 
        /// @param ks 
        /// @return 
        bool k_real_sec(int ks) const { return (ks == 0 || ks == this->L / 2.) || (this->boundary_conditions == 1); };
        
        /// @brief 
        /// @return 
        bool use_flip_X(int ks = 0) const { return false; }//( this->syms.N == this->L / 2. && (ks == this->L / 4. || ks == 3*this->L / 4.)); }

        
        typedef typename user_interface_sym<Fermions>::model_pointer model_pointer;
        typedef typename user_interface_sym<Fermions>::element_type element_type;
        
    public:
		// ----------------------------------- CONSTRUCTORS
		ui() = default;
		ui(int argc, char** argv);														// standard constructor
		
        // ----------------------------------- PARSER FUNCTION FOR HELP
		void print_help() const override;
		
        // ----------------------------------- HELPING FUNCIONS
		void set_default() override;													// set default parameters
		void printAllOptions() const override;

		// ----------------------------------- REAL PARSER
		void parse_cmd_options(int argc, std::vector<std::string> argv) override;		// the function to parse the command line

		// ----------------------------------- SIMULATION
		void make_sim() override;														// make default simulation
        virtual model_pointer create_new_model_pointer() override;
	    virtual void reset_model_pointer() override;

        // ----------------------------------- MODEL DEPENDENT FUNCTIONS
        virtual std::string set_info(std::vector<std::string> skip = {}, 
										std::string sep = "_") const override;
        void compare_energies();
        void compare_hamiltonian();
        void check_symmetry_generators();
        
        template <
			typename callable, 
			typename... _types
			> 
			void loopSymmetrySectors(
			callable& lambda, //!< callable function
			_types... args										   //!< arguments passed to callable interface lambda
		) {
            
			// const int k_end = (this->boundary_conditions) ? 1 : this->L;
            
            // std::cout << this->L << "\t\t" << this->hx << "\t\t" << zxsec << std::endl;
            // std::cout << this->L << "\t\t" << this->hz << "\t\t" << zzsec << std::endl;
		// #pragma omp parallel for num_threads(outer_threads)// schedule(dynamic)
        // #ifdef USE_REAL_SECTORS
        //     for(int ks : (this->L%2? v_1d<int>({0}) : v_1d<int>({0, (int)this->L/2})) ){
        // #else
		// 	for (int ks = 1; ks < this->L/2.0; ks++) {
        // #endif
        for (int ks = 0; ks <= (this->L-1) * (1-this->boundary_conditions); ks++) {
				v_1d<int> psec = k_real_sec(ks)? v_1d<int>({-1, 1}) : v_1d<int>({1});
                v_1d<int> zxsec = (this->use_flip_X(ks))? v_1d<int>({-1, 1}) : v_1d<int>({1});
                std::cout << ks << "\t\t" << psec << std::endl;
                for(auto& ps : psec){
                    for(auto& zxs : zxsec){
                            //<! create local lambda for multithreading enivorontment
							auto dummy_lambda = [&lambda](int k, int p, int zx, auto... args){
								lambda(k, p, zx, args...);
							};
							dummy_lambda(ks, ps, zxs, args...);
                    }
                }
            }
		}
    
        
        // ----------------------------------- OVERRIDEN METHODS
        // arma::SpMat<ui::element_type> create_supercharge(bool dagger = false);
        virtual arma::Col<element_type> cast_state(const arma::Col<element_type>& state) override;

        virtual arma::sp_mat energy_current() override;

        virtual element_type
        jE_mat_elem_kernel(
            const arma::Col<element_type>& state1, 
            const arma::Col<element_type>& state2,
            int i, u64 k, const QOps::_ifun& check_spin
            ) override;


        virtual void eigenstate_entanglement() override;
        void purity();
    };
}

#endif