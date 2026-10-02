/* test_gguf_hostile.c — C/Rust parser parity on hostile GGUF headers
 * (FINDING-rust-core-001).
 *
 * The same fixtures are asserted to fail in oxidize-core
 * (`rejects_hostile_tensor_count_without_allocating`,
 * `rejects_tensor_rank_above_max_dims` in src/format/gguf.rs), so both
 * parsers reject identical files.
 */
#include <criterion/criterion.h>

#include "oxidize/gguf.h"

#define FIXTURE_DIR "../oxidize-core/tests/fixtures"
#define FIXTURE(name) FIXTURE_DIR "/" name

Test(gguf_hostile, u64_max_tensor_count_rejected)
{
    OcGgufFile f;
    OcError e = oc_gguf_open(FIXTURE("invalid-tensor-count.gguf"), &f);
    cr_assert_eq(e, OC_ERR_FORMAT, "u64::MAX tensor_count must return OC_ERR_FORMAT, got %s",
                 oc_error_msg(e));
}

Test(gguf_hostile, n_dims_above_max_rejected)
{
    OcGgufFile f;
    OcError e = oc_gguf_open(FIXTURE("invalid-n-dims.gguf"), &f);
    cr_assert_eq(e, OC_ERR_FORMAT, "n_dims > OC_GGUF_MAX_DIMS must return OC_ERR_FORMAT, got %s",
                 oc_error_msg(e));
}
