#include "includes/pertIsing_UI.hpp"


int main(const int argc, char* argv[]) {
	std::unique_ptr<user_interface_dis<pertIsing>> intface = std::make_unique<pertIsing_UI::ui>(argc, argv);
	intface->make_sim();
	return 0;
}