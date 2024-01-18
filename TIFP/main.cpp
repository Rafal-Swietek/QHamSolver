#include "includes/TIFP_UI.hpp"


int main(const int argc, char* argv[]) {
	std::unique_ptr<user_interface_sym<TIFP>> intface = std::make_unique<TIFP_UI::ui>(argc, argv);
	intface->make_sim();
	return 0;
}