#pragma once

#ifndef UI
#define UI

#include "includes/config.h"
#include "../../include/user_interface.hpp"
#include "QuantumSun.hpp"
// ----------------------------------------------------------------------------- UI QUANTUM SUN -----------------------------------------------------------------------------

namespace QSunUI{
    class ui : public user_interface<QuantumSun>{
    protected:
        double J, Js;
        double alfa, alfas;
        double h, hs;
        double w, ws;
        double zeta;
        int Jn, alfan, hn, wn;
        
        int grain_size;
        bool initiate_avalanche;
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

        // ----------------------------------- MODEL DEPENDENT FUNCTIONS
        virtual std::string set_info(std::vector<std::string> skip = {}, 
										std::string sep = "_") const override;
    };
}

#endif