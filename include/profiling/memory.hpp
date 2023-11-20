#pragma once
#define _MEMORY_PROFILE

#ifdef __APPLE__

    //<! Profiling memory usage on Macos
    #include <mach/vm_statistics.h>
    #include <mach/mach_types.h>
    #include <mach/mach_init.h>
    #include <mach/mach_host.h>
    extern vm_size_t page_size;
    extern mach_port_t mach_port;
    extern mach_msg_type_number_t counterer;
    extern vm_statistics64_data_t my_vm_stats;
    
    inline
    std::pair<long long, long long> _profile_memory_()
    {
        if (KERN_SUCCESS == host_page_size(mach_port, &page_size) &&
            KERN_SUCCESS == host_statistics64(mach_port, HOST_VM_INFO,
                                            (host_info64_t)&my_vm_stats, &counterer))
        {
            long long free_memory = (int64_t)my_vm_stats.free_count * (int64_t)page_size;

            long long used_memory = ((int64_t)my_vm_stats.active_count +
                                    (int64_t)my_vm_stats.inactive_count +
                                    (int64_t)my_vm_stats.wire_count) *  (int64_t)page_size;
            // printf("free memory: %lld\nused memory: %lld\n", free_memory, used_memory);
            return std::make_pair(free_memory, used_memory);
        } 
        else return std::make_pair(-1, -1);
    }

#elif defined(__linux__)
    //<! Profiling memory usage on linux
    inline
    std::pair<long long, long long> _profile_memory_()
        { return std::make_pair(-1, -1); }
#elif defined(_WIN32)
    //<! Profiling memory usage on linux
    inline
    std::pair<long long, long long> _profile_memory_()
        { return std::make_pair(-1, -1); }
#else
    #pragma message("Not implemented profiling for user operating system. Only MacOSX, LINUX and WINDOWS are available.")
#endif