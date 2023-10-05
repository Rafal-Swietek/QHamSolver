#include "includes/XXZ_UI.hpp"


int main(const int argc, char* argv[]) {
	std::unique_ptr<XXZUIparent> intface = std::make_unique<XXZ_UI::ui>(argc, argv);
	intface->make_sim();
	return 0;
}