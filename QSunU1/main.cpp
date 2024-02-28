#include "includes/QSunU1UI.hpp"


int main(const int argc, char* argv[]) {
	std::unique_ptr<user_interface_dis<QuantumSunU1>> intface = std::make_unique<QSunU1UI::ui>(argc, argv);
	intface->make_sim();
	return 0;
}