#pragma once

#ifndef UI
#define UI

#include "config.hpp"
#include "../../include/user_interface.hpp"
#include "QuantumSun.hpp"
// ----------------------------------------------------------------------------- UI QUANTUM SUN -----------------------------------------------------------------------------

namespace QSunUI{
    class ui : public user_interface<QuantumSun>{
    protected:
        double J, Js;
        double alfa, alfas, gamma, gammas;
        double h, hs;
        double w, ws;
        double zeta;
        int Jn, alfan, hn, wn, gamman;
        
        int grain_size;
        bool initiate_avalanche;

        typedef typename user_interface<QuantumSun>::model_pointer model_pointer;
        typedef typename user_interface<QuantumSun>::element_type element_type;
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
        void diagonal_matrix_elements();
	    virtual void eigenstate_entanglement() override {};
    };
}

#endif