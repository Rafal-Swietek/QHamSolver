#include "includes/GoldenChain_UI.hpp"


int main(const int argc, char* argv[]) {
	std::unique_ptr<user_interface_sym<GoldenChain>> intface = std::make_unique<GoldenChain_UI::ui>(argc, argv);
	intface->make_sim();
	return 0;
}