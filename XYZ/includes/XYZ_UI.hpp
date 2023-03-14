#pragma once

#ifndef UI
#define UI

#include "config.hpp"
#include "../../include/user_interface.hpp"
#include "XYZ.hpp"
#include "XYZ_sym.hpp"

#ifdef USE_SYMMETRIES
    #define XYZUIparent user_interface<XYZsym>
#else
    #define XYZUIparent user_interface<XYZ>
#endif
// ----------------------------------------------------------------------------- UI QUANTUM SUN -----------------------------------------------------------------------------

namespace XYZ_UI{
    class ui : public XYZUIparent{
        
    protected:
        double J1, J1s;
        double J2, J2s;
        double delta1, delta1s;
        double delta2, delta2s;
        double eta1, eta2;
        double hx, hxs, hz, hzs;
        double w, ws;
        int J1n, J2n, delta1n, delta2n, hxn, hzn, wn;
        
        struct {
            int k_sym;
            int p_sym;
            int zx_sym;
            int zz_sym;
        } syms;
        
        /// @brief 
        /// @param ks 
        /// @return 
        bool k_real_sec(int ks) const { return (ks == 0 || ks == this->L / 2.) || (this->boundary_conditions == 1); };
        
        /// @brief 
        /// @return 
        bool use_flip_X() const { return ( this->hz == 0 ); }

        /// @brief 
        /// @return 
        bool use_flip_Z() const { return ( this->hx == 0 && (this->L % 2 == 0 || this->hz != 0) ); }

        bool add_parity_breaking;
        
        typedef typename XYZUIparent::model_pointer model_pointer;
        typedef typename XYZUIparent::element_type element_type;
        
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

	    virtual void eigenstate_entanglement() override;

        template <
			typename callable, 
			typename... _types
			> 
			void loopSymmetrySectors(
			callable& lambda, //!< callable function
			_types... args										   //!< arguments passed to callable interface lambda
		) {
            
			const int k_end = (this->boundary_conditions) ? 1 : this->L;
			v_1d<int> zxsec = (this->use_flip_X())? v_1d<int>({-1, 1}) : v_1d<int>({1});
			v_1d<int> zzsec = (this->use_flip_Z())? v_1d<int>({-1, 1}) : v_1d<int>({1});
		#pragma omp parallel for num_threads(outer_threads)// schedule(dynamic)
			for (int ks = 0; ks < k_end; ks++) {
				v_1d<int> psec = k_real_sec(ks)? v_1d<int>({-1, 1}) : v_1d<int>({1});
                std::cout << psec << std::endl;
                for(auto& ps : psec){
                    for(auto& zxs : zxsec){
                        for(auto& zzs : zzsec){
                            //<! create local lambda for multithreading enivorontment
							auto dummy_lambda = [&lambda](int k, int p, int zx, int zz, auto... args){
								lambda(k, p, zx, zz, args...);
							};
							dummy_lambda(ks, ps, zxs, zzs, args...);
                        }
                    }
                }
			}
		}
    };
}

#endif