#pragma once

#ifndef UI
#define UI

#include "config.hpp"
#include "../../include/user_interface_quadratic.hpp"
#include "Quadratic.hpp"

#ifdef RP
    #include "rp_def.hpp"
#endif

// ----------------------------------------------------------------------------- UI QUANTUM SUN -----------------------------------------------------------------------------

namespace QuadraticUI{
    class ui : public user_interface_quadratic<Quadratic>{
    protected:
        double J, Js, g, gs;
        double w, ws;
        int Jn, wn, gn;
        
        typedef typename user_interface_quadratic<Quadratic>::model_pointer model_pointer;
        typedef typename user_interface_quadratic<Quadratic>::element_type element_type;
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
        void set_volume();
        // ----------------------------------- OVERLOAD UI FUNCTIONS FOR SPECIFIC MODEL
        void spectrals();
        void quench();
    };
}

#endif