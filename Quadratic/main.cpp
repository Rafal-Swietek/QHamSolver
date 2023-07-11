#include "includes/QuadraticUI.hpp"


int main(const int argc, char* argv[]) {
	std::unique_ptr<user_interface_quadratic<Quadratic>> intface = std::make_unique<QuadraticUI::ui>(argc, argv);
	intface->make_sim();
	return 0;
}