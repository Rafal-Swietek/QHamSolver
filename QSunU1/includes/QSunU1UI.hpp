#pragma once

#ifndef UI
#define UI

#include "config.hpp"
#include "../../include/user_interface_dis.hpp"
#include "QuantumSunU1.hpp"
// ----------------------------------------------------------------------------- UI QUANTUM SUN -----------------------------------------------------------------------------

namespace QSunU1UI{
    class ui : public user_interface_dis<QuantumSunU1>{
    protected:
        
        double Sz = 0.0;
        double J, Js;
        double alfa, alfas, gamma, gammas;
        double h, hs;
        double w, ws;
        double zeta;
        int Jn, alfan, hn, wn, gamman;
        int L_loc;
        
        int grain_size;
        bool initiate_avalanche;

        typedef typename user_interface_dis<QuantumSunU1>::model_pointer model_pointer;
        typedef typename user_interface_dis<QuantumSunU1>::element_type element_type;
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

        // ----------------------------------- OVERRIDEN METHODS
        virtual arma::Col<element_type> cast_state(const arma::Col<element_type>& state) override;
    };
}

#endif