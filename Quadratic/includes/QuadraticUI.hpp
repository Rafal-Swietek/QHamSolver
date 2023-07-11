#pragma once

#ifndef UI
#define UI

#include "config.hpp"
#include "../../include/user_interface_dis.hpp"
#include "../../include/hilbert_space/u1.hpp"
#include "Quadratic.hpp"

#include "../../include/single_particle/entanglement.hpp"
// ----------------------------------------------------------------------------- UI QUANTUM SUN -----------------------------------------------------------------------------

namespace QuadraticUI{
    class ui : public user_interface_dis<Quadratic>{
    protected:
        double J, Js;
        double w, ws;
        int Jn, wn;
        int V;

        typedef typename user_interface_dis<Quadratic>::model_pointer model_pointer;
        typedef typename user_interface_dis<Quadratic>::element_type element_type;
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

        // ----------------------------------- OVERLOAD UI FUNCTIONS FOR SPECIFIC MODEL
	    virtual void eigenstate_entanglement() 	override;
    };
}

#endif