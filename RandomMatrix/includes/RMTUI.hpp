#pragma once

#ifndef UI
#define UI

#include "config.hpp"
#include "../../include/user_interface_dis.hpp"
#include "RandomMatrix.hpp"
// ----------------------------------------------------------------------------- UI QUANTUM SUN -----------------------------------------------------------------------------

namespace QSunUI{
    class ui : public user_interface_dis<RandomMatrix>{
    protected:
        u64 dim;
        
        double b, bs;
        double alfa, alfas, gamma, gammas;
        int bn, alfan, gamman;

        typedef typename user_interface_dis<RandomMatrix>::model_pointer model_pointer;
        typedef typename user_interface_dis<RandomMatrix>::element_type element_type;
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
    };
}

#endif