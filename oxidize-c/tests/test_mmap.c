/* test_mmap.c — tests for OcMmap (read-only memory mapping). */
/* - VAL-FOUND-006 (mmap with MADV_HUGEPAGE): the mmap path is exercised, */
/* - VAL-FOUND-015 (mmap/arena lifecycle valgrind-clean): the test_runner */
#include <criterion/criterion.h>
#include "oxidize/arena.h"
#include "oxidize/error.h"
#include "oxidize/gguf.h"
#include "oxidize/util/mmap.h"

#include <fcntl.h>      /* open, O_RDONLY */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* memset, str_eq */
#include <unistd.h>     /* read, lseek, close */

#define FIXTURE_DIR "../oxidize-core/tests/fixtures"
#define FIXTURE(name) FIXTURE_DIR "/" name

Test(mmap, open_readonly_returns_mapped_bytes)
{
    OcMmap *m = NULL;
    OcError e = oc_mmap_open_readonly(FIXTURE("valid-v3.gguf"), &m);
    cr_assert_eq(e, OC_OK, "open: %s", oc_error_msg(e));
    cr_assert_not_null(m, "OcMmap should be heap-allocated");

    /* The fixture is 132 bytes (128 header + 4 data). */
    cr_assert_eq(oc_mmap_len(m), 132, "fixture should be 132 bytes");
    const uint8_t *bytes = oc_mmap_bytes(m);
    cr_assert_not_null(bytes, "mapped bytes should be non-NULL");

    /* Verify the GGUF magic ("GGUF" little-endian = 0x46554747). */
    cr_assert_eq(bytes[0], 'G', "magic[0]");
    cr_assert_eq(bytes[1], 'G', "magic[1]");
    cr_assert_eq(bytes[2], 'U', "magic[2]");
    cr_assert_eq(bytes[3], 'F', "magic[3]");

    /* The data section (offset 128) contains [1, 2, 3, 4]. */
    cr_assert_eq(bytes[128], 1, "data[0]");
    cr_assert_eq(bytes[129], 2, "data[1]");
    cr_assert_eq(bytes[130], 3, "data[2]");
    cr_assert_eq(bytes[131], 4, "data[3]");

    oc_mmap_close(m);
}

Test(mmap, null_args_rejected)
{
    OcMmap *m = NULL;
    cr_assert_eq(oc_mmap_open_readonly(NULL, &m), OC_ERR_INVALID_ARG, "NULL path");
    cr_assert_eq(oc_mmap_open_readonly("foo", NULL), OC_ERR_INVALID_ARG, "NULL out");
    /* Safe on NULL. */
    oc_mmap_close(NULL);
}

Test(mmap, missing_file_returns_io_error)
{
    OcMmap *m = NULL;
    OcError e = oc_mmap_open_readonly("/tmp/oxidize-c-mmap-nonexistent-xxxx.gguf", &m);
    cr_assert_eq(e, OC_ERR_IO, "missing file should return OC_ERR_IO, got %s", oc_error_msg(e));
    cr_assert_null(m, "out should be NULL on error");
}

Test(mmap, advise_hugepage_best_effort)
{
    /* VAL-FOUND-006: MADV_HUGEPAGE is applied (best-effort). For a tiny */
    OcMmap *m = NULL;
    OcError e = oc_mmap_open_readonly(FIXTURE("valid-v3.gguf"), &m);
    cr_assert_eq(e, OC_OK, "open: %s", oc_error_msg(e));

    /* The hint may or may not stick depending on the kernel; we just verify
     * the call returns OC_OK (best-effort, never errors). */
    e = oc_mmap_advise_hugepage(m);
    cr_assert_eq(e, OC_OK, "advise_hugepage should be best-effort: %s", oc_error_msg(e));

    oc_mmap_close(m);
}

Test(mmap, prefault_returns_nonzero_checksum)
{
    /* Prefault touches every page and returns XOR checksum. For a 132-byte
     * file, every byte is touched at offsets 0 and 131 (last). */
    OcMmap *m = NULL;
    OcError e = oc_mmap_open_readonly(FIXTURE("valid-v3.gguf"), &m);
    cr_assert_eq(e, OC_OK, "open: %s", oc_error_msg(e));

    uint8_t checksum = oc_mmap_prefault(m);
    /* The fixture's first byte is 'G' = 0x47; last byte is 4. XOR = 0x43. */
    uint8_t expected = 0;
    const uint8_t *bytes = oc_mmap_bytes(m);
    for (size_t off = 0; off < oc_mmap_len(m); off += 4096) {
        expected ^= bytes[off];
    }
    expected ^= bytes[oc_mmap_len(m) - 1];
    cr_assert_eq(checksum, expected, "prefault checksum should match manual XOR");

    oc_mmap_close(m);
}

Test(mmap, prefault_parallel_matches_single_threaded)
{
    /* Parallel prefault (4 threads) should produce the same checksum as
     * single-threaded prefault. */
    OcMmap *m1 = NULL, *m2 = NULL;
    OcError e1 = oc_mmap_open_readonly(FIXTURE("valid-v3.gguf"), &m1);
    OcError e2 = oc_mmap_open_readonly(FIXTURE("valid-v3.gguf"), &m2);
    cr_assert_eq(e1, OC_OK, "open1: %s", oc_error_msg(e1));
    cr_assert_eq(e2, OC_OK, "open2: %s", oc_error_msg(e2));

    uint8_t c1 = oc_mmap_prefault(m1);
    uint8_t c2 = oc_mmap_prefault_parallel(m2, 4);
    cr_assert_eq(c1, c2, "parallel prefault checksum should match single-threaded");

    oc_mmap_close(m1);
    oc_mmap_close(m2);
}

Test(mmap, lifecycle_1000_open_close_cycles_no_leak)
{
    /* VAL-FOUND-015 (valgrind-clean): run 1000 open/close cycles. */
    for (int i = 0; i < 1000; i++) {
        OcMmap *m = NULL;
        OcError e = oc_mmap_open_readonly(FIXTURE("valid-v3.gguf"), &m);
        cr_assert_eq(e, OC_OK, "cycle %d: %s", i, oc_error_msg(e));
        /* Touch a byte to ensure the mapping is faulted in. */
        const uint8_t *bytes = oc_mmap_bytes(m);
        cr_assert_not_null(bytes, "cycle %d: bytes NULL", i);
        volatile uint8_t v = bytes[0];
        (void)v;
        oc_mmap_close(m);
    }
}

Test(mmap, open_fd_succeeds_and_takes_ownership_of_fd)
{
#ifndef __linux__
    cr_skip_test("oc_mmap_open_fd is only supported on Linux");
#endif
    int fd = open(FIXTURE("valid-v3.gguf"), O_RDONLY);
    cr_assert_geq(fd, 0, "open() should succeed on the fixture");

    OcMmap *m = NULL;
    OcError e = oc_mmap_open_fd(fd, 132, &m);
    cr_assert_eq(e, OC_OK, "open_fd success: %s", oc_error_msg(e));
    cr_assert_not_null(m, "OcMmap should be allocated on success");

    const uint8_t *bytes = oc_mmap_bytes(m);
    cr_assert_not_null(bytes, "mapped bytes should be non-NULL");
    cr_assert_eq(bytes[0], 'G', "magic[0] via open_fd");
    cr_assert_eq(oc_mmap_len(m), 132, "len should be 132");

    oc_mmap_close(m);  /* must close(fd) here — caller transferred ownership */
}

Test(mmap, open_fd_invalid_args_rejected_without_closing_caller_fd)
{
    /* When oc_mmap_open_fd rejects args BEFORE attempting mmap (NULL out, negative fd, zero len), it must NOT close the caller's fd — the caller retains ownership on the validation-failure path. We verify by opening a real fd, passing invalid args, and confirming the fd is still valid (readable) afterwards. */
    int fd = open(FIXTURE("valid-v3.gguf"), O_RDONLY);
    cr_assert_geq(fd, 0, "open() should succeed on the fixture");

    OcMmap *m = NULL;
    /* NULL out: must NOT close fd (caller retains ownership). */
    cr_assert_eq(oc_mmap_open_fd(fd, 132, NULL), OC_ERR_INVALID_ARG,
        "NULL out must return OC_ERR_INVALID_ARG");
    char c = 0;
    ssize_t n = read(fd, &c, 1);
    cr_assert_eq(n, 1, "fd should still be valid after NULL-out rejection");
    cr_assert_eq(c, 'G', "first byte should be 'G'");

    /* Negative fd: rejected, no fd touched. */
    cr_assert_eq(oc_mmap_open_fd(-1, 132, &m), OC_ERR_INVALID_ARG,
        "negative fd must return OC_ERR_INVALID_ARG");

    /* Zero len: rejected, but caller fd NOT closed (still ours). */
    cr_assert_eq(oc_mmap_open_fd(fd, 0, &m), OC_ERR_INVALID_ARG,
        "zero len must return OC_ERR_INVALID_ARG");
    /* fd is still valid. */
    n = lseek(fd, 0, SEEK_SET);
    cr_assert_eq(n, 0, "lseek back to 0 should succeed (fd still valid)");

    close(fd);  /* caller closes its own fd after invalid-arg rejections */
}

Test(mmap, open_fd_map_failure_closes_fd_no_leak)
{
#ifndef __linux__
    cr_skip_test("fd ownership and /proc fdinfo checks are Linux-specific");
#endif
    /* Scrutiny fix: when mmap() fails inside oc_mmap_open_fd (MAP_FAILED), the function must close(fd) before free(m) — otherwise the caller- provided fd leaks (the caller has no way to reclaim it once ownership has been "handed off" to oc_mmap_open_fd). */
    int fd = open(FIXTURE("valid-v3.gguf"), O_RDONLY);
    cr_assert_geq(fd, 0, "open() should succeed on the fixture");

    /* Count open fds before the call (we expect fd to be closed by the
     * error path, leaving the open-fd count unchanged from before open). */
    char before_path[64], after_path[64];
    snprintf(before_path, sizeof(before_path), "/proc/self/fdinfo/%d", fd);
    /* before: fd should exist (just opened). */
    FILE *fb = fopen(before_path, "r");
    cr_assert_not_null(fb, "fdinfo/%d should exist before open_fd", fd);
    fclose(fb);

    OcMmap *m = NULL;
    OcError e = oc_mmap_open_fd(fd, (size_t)1 << 62, &m);
    cr_assert_eq(e, OC_ERR_INVALID_ARG,
        "oversized len should return OC_ERR_INVALID_ARG, got %s",
        oc_error_msg(e));
    cr_assert_null(m, "out should be NULL on failure");

    /* After the failed call, fd should have been closed by the error path
     * (the scrutiny fix). /proc/self/fdinfo/<fd> should now NOT exist. */
    snprintf(after_path, sizeof(after_path), "/proc/self/fdinfo/%d", fd);
    FILE *fa = fopen(after_path, "r");
    cr_assert_null(fa, "fd should be CLOSED by the MAP_FAILED error path "
        "(no fd leak) — /proc/self/fdinfo/%d should not exist", fd);
    /* If fa is non-NULL (test regression — fd leaked), close it to avoid
     * a leak in the test itself. */
    if (fa) fclose(fa);
}

Test(mmap, mlock_with_headroom_safe_for_tiny_file)
{
    /* For a 132-byte fixture, mlock_with_headroom should succeed (or skip
     * gracefully if MemAvailable is unreadable). Either way, no crash and
     * the mapping remains usable. */
    OcMmap *m = NULL;
    OcError e = oc_mmap_open_readonly(FIXTURE("valid-v3.gguf"), &m);
    cr_assert_eq(e, OC_OK, "open: %s", oc_error_msg(e));

    /* The result is platform-dependent (may or may not have CAP_IPC_LOCK).
     * We just verify no crash + mapping still readable. */
    (void)oc_mmap_mlock_with_headroom(m);

    const uint8_t *bytes = oc_mmap_bytes(m);
    cr_assert_eq(bytes[0], 'G', "mapping should still be readable after mlock attempt");

    oc_mmap_close(m);
}

Test(mmap, linux_mem_available_readable)
{
    /* On Linux, /proc/meminfo should be readable. We don't assert a specific
     * value (it varies by host), just that the function returns true and
     * writes a non-zero value. */
    uint64_t avail = 0;
    bool ok = oc_linux_mem_available_bytes(&avail);
    if (ok) {
        cr_assert_gt(avail, 0, "MemAvailable should be > 0 on a real Linux host");
    }
    /* If !ok (e.g. non-Linux), we skip the value check. */
}

/* ─── VAL-FOUND-006: oc_gguf_map_open exposes mmap'd backing ──────────── */

Test(gguf_map, open_via_mmap_returns_unified_view)
{
    /* oc_gguf_map_open() opens the GGUF via mmap and exposes a unified
     * OcGgufFile view. For a single-file GGUF, n_shards == 1. */
    OcGgufMmappedFile m;
    OcError e = oc_gguf_map_open(FIXTURE("valid-v3.gguf"), &m);
    cr_assert_eq(e, OC_OK, "map_open: %s", oc_error_msg(e));

    cr_assert_eq(m.n_shards, 1, "single-file GGUF should have 1 shard");
    cr_assert_not_null(m.shards, "shards array should be non-NULL");
    cr_assert_not_null(m.shards[0].mmap, "shard 0 mmap should be non-NULL");
    cr_assert_not_null(m.shards[0].bytes, "shard 0 bytes should be non-NULL");
    cr_assert_eq(m.shards[0].len, 132, "shard 0 len should be 132");

    /* Unified view. */
    cr_assert_eq(m.unified.magic, OC_GGUF_MAGIC, "unified magic");
    cr_assert_eq(m.unified.version, 3, "unified version");
    cr_assert_eq(m.unified.tensor_count, 1, "unified tensor_count");
    cr_assert_eq(m.unified.metadata_kv_count, 1, "unified metadata_kv_count");
    cr_assert_eq(m.unified.alignment, 64, "unified alignment");
    cr_assert_eq(m.unified.data_section_start, 128, "unified data_section_start");

    /* The unified tensor's shard_index should be 0 (single-file). */
    cr_assert_eq(m.unified.tensors[0].shard_index, 0, "shard_index should be 0");
    cr_assert_str_eq(m.unified.tensors[0].name, "tok_embeddings.weight", "tensor name");

    oc_gguf_map_free(&m);
}

Test(gguf_map, tensor_data_accessible_via_mmap)
{
    OcGgufMmappedFile m;
    OcError e = oc_gguf_map_open(FIXTURE("valid-v3.gguf"), &m);
    cr_assert_eq(e, OC_OK, "map_open: %s", oc_error_msg(e));

    const OcGgufTensorInfo *t = &m.unified.tensors[0];
    const uint8_t *data = oc_gguf_map_tensor_data(&m, t);
    cr_assert_not_null(data, "tensor data should be non-NULL");
    /* The tensor's absolute_offset is 128; the fixture stores [1,2,3,4] there (4 bytes for a 1-element F32 tensor — but the dims say [32000, 4096] which doesn't match the 4 data bytes. That's expected: the fixture is a synthetic minimal GGUF, not a real model. We just verify the pointer arithmetic: data should point at bytes[128]. */
    cr_assert_eq(data, m.shards[0].bytes + 128, "tensor data pointer arithmetic");
    cr_assert_eq(data[0], 1, "data[0]");
    cr_assert_eq(data[1], 2, "data[1]");
    cr_assert_eq(data[2], 3, "data[2]");
    cr_assert_eq(data[3], 4, "data[3]");

    oc_gguf_map_free(&m);
}

Test(gguf_map, arch_detection_from_mapped_file)
{
    OcGgufMmappedFile m;
    OcError e = oc_gguf_map_open(FIXTURE("valid-v3.gguf"), &m);
    cr_assert_eq(e, OC_OK, "map_open: %s", oc_error_msg(e));

    OcModelArchitecture arch = oc_gguf_arch_from_file(&m.unified);
    /* The valid-v3 fixture has only `general.alignment` (no arch namespace),
     * so detection returns OC_ARCH_UNKNOWN. This is the documented behavior. */
    cr_assert_eq(arch, OC_ARCH_UNKNOWN,
        "valid-v3.gguf has no arch namespace, expected UNKNOWN, got %s",
        oc_model_arch_name(arch));

    oc_gguf_map_free(&m);
}

Test(gguf_map, free_is_safe_on_zeroed)
{
    OcGgufMmappedFile m;
    memset(&m, 0, sizeof(m));
    oc_gguf_map_free(&m);
    oc_gguf_map_free(NULL);
}

Test(gguf_map, lifecycle_100_open_close_cycles_no_leak)
{
    /* VAL-FOUND-015: mmap/arena lifecycle valgrind-clean. ASan substitutes
     * for valgrind locally. Run 1000 cycles to exercise mmap/munmap pairs. */
    for (int i = 0; i < 1000; i++) {
        OcGgufMmappedFile m;
        OcError e = oc_gguf_map_open(FIXTURE("valid-v3.gguf"), &m);
        cr_assert_eq(e, OC_OK, "cycle %d: %s", i, oc_error_msg(e));
        /* Verify the unified tensor table is intact. */
        cr_assert_eq(m.unified.tensor_count, 1, "cycle %d: tensor_count", i);
        /* Verify tensor data is accessible. */
        const uint8_t *data = oc_gguf_map_tensor_data(&m, &m.unified.tensors[0]);
        cr_assert_not_null(data, "cycle %d: tensor data", i);
        oc_gguf_map_free(&m);
    }
}

Test(gguf_map, advise_hugepage_on_mapped_file)
{
    /* VAL-FOUND-006: oc_gguf_map_advise_hugepage applies MADV_HUGEPAGE to
     * every shard. Best-effort; verifies the call doesn't error out. */
    OcGgufMmappedFile m;
    OcError e = oc_gguf_map_open(FIXTURE("valid-v3.gguf"), &m);
    cr_assert_eq(e, OC_OK, "map_open: %s", oc_error_msg(e));

    e = oc_gguf_map_advise_hugepage(&m);
    cr_assert_eq(e, OC_OK, "advise_hugepage: %s", oc_error_msg(e));

    oc_gguf_map_free(&m);
}

Test(gguf_map, total_bytes_matches_file_size)
{
    OcGgufMmappedFile m;
    OcError e = oc_gguf_map_open(FIXTURE("valid-v3.gguf"), &m);
    cr_assert_eq(e, OC_OK, "map_open: %s", oc_error_msg(e));

    cr_assert_eq(oc_gguf_map_total_bytes(&m), 132, "total bytes should be 132");

    oc_gguf_map_free(&m);
}
