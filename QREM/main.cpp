#include "includes/QREM_UI.hpp"


int main(const int argc, char* argv[]) {
	std::unique_ptr<user_interface_dis<QREM>> intface = std::make_unique<QREM_UI::ui>(argc, argv);
	intface->make_sim();
	return 0;
}