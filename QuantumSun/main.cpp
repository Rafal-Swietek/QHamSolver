#include "includes/QSunUI.hpp"


int main(const int argc, char* argv[]) {
	std::unique_ptr<user_interface<QuantumSun>> intface = std::make_unique<QSunUI::ui>(argc, argv);
	intface->make_sim();
	return 0;
}