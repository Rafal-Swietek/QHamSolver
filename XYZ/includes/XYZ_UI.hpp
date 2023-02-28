#pragma once

#ifndef UI
#define UI

#include "config.hpp"
#include "../../include/user_interface.hpp"
#include "XYZ.hpp"
#include "XYZ_sym.hpp"

// ----------------------------------------------------------------------------- UI QUANTUM SUN -----------------------------------------------------------------------------

namespace XYZ_UI{
    class ui : public user_interface<XYZ>{
    protected:
        double J1, J1s;
        double J2, J2s;
        double delta1, delta1s;
        double delta2, delta2s;
        double eta1, eta2;
        double hx, hxs, hz, hzs;
        double w, ws;
        int J1n, J2n, delta1n, delta2n, hxn, hzn, wn;
        
        bool add_parity_breaking;
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