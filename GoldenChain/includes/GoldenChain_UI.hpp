#pragma once

#ifndef UI
#define UI

#include "config.hpp"

#include "../../include/user_interface_sym.hpp"

#include "GoldenChain.hpp"

// ----------------------------------------------------------------------------- UI QUANTUM SUN -----------------------------------------------------------------------------

namespace GoldenChain_UI{
    class ui : public user_interface_sym<GoldenChain>{
        
    protected:
        double J, Js, c, cs;
        int Jn, cn;
        
        struct {
            int k_sym;
            int p_sym;
        } syms;
        
        /// @brief 
        /// @param ks 
        /// @return 
        bool k_real_sec(int ks) const { return (ks == 0 || ks == this->L / 2.) || (this->boundary_conditions == 1); };
        
        typedef typename user_interface_sym<GoldenChain>::model_pointer model_pointer;
        typedef typename user_interface_sym<GoldenChain>::element_type element_type;
        
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
            
			const int k_end = (this->boundary_conditions) ? 1 : this->L;
		// #pragma omp parallel for num_threads(outer_threads)// schedule(dynamic)
        // #ifdef USE_REAL_SECTORS
        //     for(int ks : (this->L%2 || this->boundary_conditions? v_1d<int>({0}) : v_1d<int>({0, (int)this->L/2})) ){
        // #else
		// 	for (int ks = 1; ks < k_end; ks++) {
        // #endif
			for (int ks = 0; ks < this->L; ks++) {
				v_1d<int> psec = (k_real_sec(ks) && this->c == 0)? v_1d<int>({-1, 1}) : v_1d<int>({1});
                std::cout << ks << "\t\t" << psec << std::endl;
                for(auto& ps : psec){
                    //<! create local lambda for multithreading enivorontment
                    auto dummy_lambda = [&lambda](int k, int p, auto... args){
                        lambda(k, p, args...);
                    };
                    dummy_lambda(ks, ps, args...);
                }
			}
		}
    
        
        // ----------------------------------- HELPER FUNCTIONS
        arma::SpMat<ui::element_type> create_supercharge(bool dagger = false);
        
	    virtual arma::sp_mat energy_current() override;
	    
        virtual element_type
        jE_mat_elem_kernel(
            const arma::Col<element_type>& state1, 
            const arma::Col<element_type>& state2,
            int i, u64 k, const QOps::_ifun& check_spin
            ) override;
    };
}

#endif