#pragma once
#define _CPU_PROFILE

#ifdef __APPLE__
    #include <mach/mach_error.h>
    #include <mach/vm_map.h>
    //<! Profiling memory cpu on Macos
    static unsigned long long _previousTotalTicks = 0;
    static unsigned long long _previousIdleTicks = 0;

    inline
    float CalculateCPULoad(unsigned long long idleTicks, unsigned long long totalTicks)
    {
        unsigned long long totalTicksSinceLastTime = totalTicks-_previousTotalTicks;
        unsigned long long idleTicksSinceLastTime  = idleTicks-_previousIdleTicks;
        float ret = 1.0f-((totalTicksSinceLastTime > 0) ? ((float)idleTicksSinceLastTime)/totalTicksSinceLastTime : 0);
        _previousTotalTicks = totalTicks;
        _previousIdleTicks  = idleTicks;
        return ret;
    }
    // Returns 1.0f for "CPU fully pinned", 0.0f for "CPU idle", or somewhere in between
    // You'll need to call this at regular intervals, since it measures the load between
    // the previous call and the current one.
    inline
    float _profile_cpu_()
    {
        host_cpu_load_info_data_t cpuinfo;
        mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;
        if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, (host_info_t)&cpuinfo, &count) == KERN_SUCCESS)
        {
            unsigned long long totalTicks = 0;
            for(int i=0; i<CPU_STATE_MAX; i++) totalTicks += cpuinfo.cpu_ticks[i];
            return CalculateCPULoad(cpuinfo.cpu_ticks[CPU_STATE_IDLE], totalTicks);
        }
        else return -1.0f;
    }

#elif defined(__linux__)
    //<! Profiling cpu usage on linux
    inline float _profile_cpu_(){ return 0.0f;};
#elif defined(_WIN32)
    //<! Profiling cpu usage on linux
    inline float _profile_cpu_(){ return 0.0f;};

#else
    #pragma message("Not implemented profiling for user operating system. Only MacOSX, LINUX and WINDOWS are available.")
#endif