#ifndef CACHE_INFO_H
#define CACHE_INFO_H

#ifdef _WIN32
#include <windows.h>
#elif defined(__linux__)
#include <unistd.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#else
#error "Unsupported platform"
#endif

struct CacheInfo {
    long l1d_size;     // L1D size in bytes
    long line_size;    // Cache line size in bytes
};

CacheInfo get_cache_info() {
    CacheInfo info = {-1, -1};
#ifdef _WIN32
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION* buffer = nullptr;
    DWORD size = 0;
    GetLogicalProcessorInformation(nullptr, &size);
    buffer = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION*)malloc(size);
    if (!buffer) return info;
    if (!GetLogicalProcessorInformation(buffer, &size)) {
        free(buffer);
        return info;
    }
    for (DWORD i = 0; i < size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION); ++i) {
        if (buffer[i].Relationship == RelationCache && 
            buffer[i].Cache.Level == 1 && 
            buffer[i].Cache.Type == CacheData) {
            info.l1d_size = buffer[i].Cache.Size;
            info.line_size = buffer[i].Cache.LineSize;
            break;
        }
    }
    free(buffer);
#elif defined(__linux__)
#ifdef _SC_LEVEL1_DCACHE_SIZE
    info.l1d_size = sysconf(_SC_LEVEL1_DCACHE_SIZE);
    info.line_size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
#endif
#elif defined(__APPLE__)
    size_t len = sizeof(info.l1d_size);
    if (sysctlbyname("hw.l1dcachesize", &info.l1d_size, &len, nullptr, 0) == -1) {
        info.l1d_size = -1;
    }
    len = sizeof(info.line_size);
    if (sysctlbyname("hw.cachelinesize", &info.line_size, &len, nullptr, 0) == -1) {
        info.line_size = -1;
    }
#endif
    return info;
}
#endif