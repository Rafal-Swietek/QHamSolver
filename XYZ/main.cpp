#include "includes/XYZ_UI.hpp"


int main(const int argc, char* argv[]) {
	std::unique_ptr<XYZUIparent> intface = std::make_unique<XYZ_UI::ui>(argc, argv);
	intface->make_sim();
	return 0;
}