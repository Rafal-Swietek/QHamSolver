#include "includes/XXZ_UI.hpp"

#ifdef __APPLE__
    vm_statistics64_data_t my_vm_stats = {};
	mach_port_t mach_port = mach_host_self();
	mach_msg_type_number_t counterer = sizeof(my_vm_stats) / sizeof(natural_t);
    vm_size_t page_size = 1;
#endif

int main(const int argc, char* argv[]) {
	arma::arma_version ver;
	std::cout << "ARMA version: "<< ver.as_string() << std::endl;
	std::unique_ptr<XXZUIparent> intface = std::make_unique<XXZ_UI::ui>(argc, argv);
	intface->make_sim();
	return 0;
}