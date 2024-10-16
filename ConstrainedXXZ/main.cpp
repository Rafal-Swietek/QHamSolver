#include "includes/ConstrainedXXZ_UI.hpp"


int main(const int argc, char* argv[]) {
	std::unique_ptr<user_interface_sym<ConstrainedXXZ>> intface = std::make_unique<ConstrainedXXZ_UI::ui>(argc, argv);
	intface->make_sim();
	return 0;
}