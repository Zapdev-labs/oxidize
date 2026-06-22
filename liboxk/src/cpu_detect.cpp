#include "oxk_common.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <cpuid.h>
#include <immintrin.h>
#define OXK_X86 1
#endif

namespace oxk {

static bool g_avx2 = false;
static bool g_initialized = false;

void cpu_detect_init() {
    if (g_initialized) {
        return;
    }
#ifdef OXK_X86
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        const bool osxsave = (ecx & (1u << 27)) != 0;
        const bool avx = (ecx & (1u << 28)) != 0;
        if (osxsave && avx && __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
            g_avx2 = (ebx & (1u << 5)) != 0 && (ebx & (1u << 12)) != 0; // AVX2 + FMA
        }
    }
#endif
    g_initialized = true;
}

bool has_avx2() {
    if (!g_initialized) {
        cpu_detect_init();
    }
    return g_avx2;
}

} // namespace oxk
