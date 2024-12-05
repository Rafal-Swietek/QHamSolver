#include "includes/QuadQSunUI.hpp"


int main(const int argc, char* argv[]) {
	arma::arma_version ver;
	std::cout << "ARMA version: "<< ver.as_string() << std::endl;
	std::unique_ptr<user_interface_dis<QuadQSun>> intface = std::make_unique<QuadQSunUI::ui>(argc, argv);
	intface->make_sim();
	std::cout << "TRULY FINISHED." << std::endl;
	return 0;
}