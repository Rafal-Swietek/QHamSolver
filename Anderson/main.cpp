#include "includes/AndersonUI.hpp"


int main(const int argc, char* argv[]) {
	std::unique_ptr<user_interface_dis<Anderson>> intface = std::make_unique<AndersonUI::ui>(argc, argv);
	intface->make_sim();
	return 0;
}