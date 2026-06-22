#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "oxk.h"

int main() {
    oxk_init();
    printf("oxk version %s avx2=%d\n", oxk_version(), oxk_has_avx2());

    const float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float b[] = {0.5f, 1.5f, 2.5f, 3.5f};
    const float dot = oxk_dot_f32(a, b, 4);
    assert(std::fabs(dot - 25.0f) < 1e-5f);

    std::vector<float> x(4, 1.0f);
    std::vector<float> w(4, 2.0f);
    std::vector<float> out(4);
    assert(oxk_rms_norm_f32(x.data(), w.data(), 4, 1e-5f, out.data()) == 0);
    assert(std::fabs(out[0] - 2.0f) < 1e-4f);

    printf("liboxk smoke test passed\n");
    return 0;
}
