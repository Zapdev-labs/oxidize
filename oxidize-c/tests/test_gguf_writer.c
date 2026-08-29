/* test_gguf_writer.c — Criterion tests for the GGUF v3 writer.
 *
 * Covers:
 *   - Header initialization (magic, version, counts)
 *   - Metadata KV of each type (string, uint32, uint64, float32, array<string>)
 *   - Tensor writing (single, multiple, alignment)
 *   - Finalize (count patching)
 *   - Round-trip: write then parse back with the GGUF parser
 *   - Error handling (NULL args, finalized writer, etc.)
 */

#define _POSIX_C_SOURCE 200809L

#include "framework.h"

#include "oxidize/gguf.h"
#include "oxidize/gguf_writer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Helper: create a temp file path and return it (static buffer). */
static const char *make_temp_path(const char *suffix)
{
    static char path[256];
    snprintf(path, sizeof(path), "/tmp/oc_gguf_test_%d_%s", (int)getpid(), suffix);
    return path;
}

/* ─── Header initialization ─────────────────────────────────────────────── */

Test(gguf_writer, init_creates_valid_header)
{
    const char *path = make_temp_path("init.gguf");
    OcGgufWriter w;
    OcError e = oc_gguf_writer_init(path, "llama", &w);
    cr_assert_eq(e, OC_OK, "init failed: %s", oc_error_msg(e));
    cr_assert_not_null(w.fp, "fp should not be NULL");
    cr_assert_eq(w.tensor_count, 0, "tensor_count should be 0");
    cr_assert_eq(w.metadata_count, 1, "metadata_count should be 1 (general.architecture)");
    cr_assert_eq(w.finalized, false, "should not be finalized");
    oc_gguf_writer_free(&w);
    unlink(path);
}

Test(gguf_writer, init_writes_magic_and_version)
{
    const char *path = make_temp_path("magic.gguf");
    OcGgufWriter w;
    OcError e = oc_gguf_writer_init(path, "llama", &w);
    cr_assert_eq(e, OC_OK, "");

    /* Parse it back. */
    oc_gguf_writer_finalize(&w);
    oc_gguf_writer_free(&w);

    OcGgufFile f;
    e = oc_gguf_open(path, &f);
    cr_assert_eq(e, OC_OK, "re-parse failed: %s", oc_error_msg(e));
    cr_assert_eq(f.magic, OC_GGUF_MAGIC, "magic should be GGUF");
    cr_assert_eq(f.version, 3, "version should be 3");
    oc_gguf_free(&f);
    unlink(path);
}

Test(gguf_writer, init_null_path)
{
    OcGgufWriter w;
    OcError e = oc_gguf_writer_init(NULL, "llama", &w);
    cr_assert_neq(e, OC_OK, "NULL path should fail");
}

Test(gguf_writer, init_null_writer)
{
    OcError e = oc_gguf_writer_init("foo.gguf", "llama", NULL);
    cr_assert_neq(e, OC_OK, "NULL writer should fail");
}

Test(gguf_writer, init_null_arch_skips_arch_kv)
{
    const char *path = make_temp_path("noarch.gguf");
    OcGgufWriter w;
    OcError e = oc_gguf_writer_init(path, NULL, &w);
    cr_assert_eq(e, OC_OK, "");
    cr_assert_eq(w.metadata_count, 0, "metadata_count should be 0 when arch_name is NULL");
    oc_gguf_writer_finalize(&w);
    oc_gguf_writer_free(&w);
    unlink(path);
}

/* ─── Metadata KV: string ───────────────────────────────────────────────── */

Test(gguf_writer, add_string_metadata)
{
    const char *path = make_temp_path("meta_str.gguf");
    OcGgufWriter w;
    oc_gguf_writer_init(path, "llama", &w);
    OcError e = oc_gguf_writer_add_string(&w, "general.name", "test-model");
    cr_assert_eq(e, OC_OK, "");
    cr_assert_eq(w.metadata_count, 2, "should have 2 KVs (arch + name)");
    oc_gguf_writer_finalize(&w);
    oc_gguf_writer_free(&w);

    /* Verify round-trip. */
    OcGgufFile f;
    e = oc_gguf_open(path, &f);
    cr_assert_eq(e, OC_OK, "");
    const char *str_data;
    size_t str_len;
    cr_assert(oc_gguf_metadata_get_str(&f, "general.name", &str_data, &str_len),
              "general.name should exist");
    cr_assert_str_eq(str_data, "test-model", "");
    cr_assert_eq(f.metadata_kv_count, 2, "should have 2 KVs");
    oc_gguf_free(&f);
    unlink(path);
}

Test(gguf_writer, add_string_null_args)
{
    OcGgufWriter w;
    oc_gguf_writer_init(make_temp_path("ns.gguf"), "llama", &w);
    cr_assert_neq(oc_gguf_writer_add_string(&w, NULL, "val"), OC_OK, "");
    cr_assert_neq(oc_gguf_writer_add_string(&w, "key", NULL), OC_OK, "");
    cr_assert_neq(oc_gguf_writer_add_string(NULL, "key", "val"), OC_OK, "");
    oc_gguf_writer_free(&w);
    unlink(make_temp_path("ns.gguf"));
}

/* ─── Metadata KV: uint32 ───────────────────────────────────────────────── */

Test(gguf_writer, add_uint32_metadata)
{
    const char *path = make_temp_path("meta_u32.gguf");
    OcGgufWriter w;
    oc_gguf_writer_init(path, "llama", &w);
    OcError e = oc_gguf_writer_add_uint32(&w, "llama.context_length", 4096);
    cr_assert_eq(e, OC_OK, "");
    oc_gguf_writer_finalize(&w);
    oc_gguf_writer_free(&w);

    OcGgufFile f;
    e = oc_gguf_open(path, &f);
    cr_assert_eq(e, OC_OK, "");
    uint32_t val;
    cr_assert(oc_gguf_metadata_get_u32(&f, "llama.context_length", &val), "");
    cr_assert_eq(val, 4096, "");
    oc_gguf_free(&f);
    unlink(path);
}

Test(gguf_writer, add_uint32_null_key)
{
    OcGgufWriter w;
    oc_gguf_writer_init(make_temp_path("u32n.gguf"), "llama", &w);
    cr_assert_neq(oc_gguf_writer_add_uint32(&w, NULL, 42), OC_OK, "");
    oc_gguf_writer_free(&w);
    unlink(make_temp_path("u32n.gguf"));
}

Test(gguf_writer, add_uint32_max_value)
{
    const char *path = make_temp_path("u32max.gguf");
    OcGgufWriter w;
    oc_gguf_writer_init(path, "llama", &w);
    oc_gguf_writer_add_uint32(&w, "test.max_u32", 0xFFFFFFFF);
    oc_gguf_writer_finalize(&w);
    oc_gguf_writer_free(&w);

    OcGgufFile f;
    oc_gguf_open(path, &f);
    uint32_t val;
    cr_assert(oc_gguf_metadata_get_u32(&f, "test.max_u32", &val), "");
    cr_assert_eq(val, 0xFFFFFFFF, "");
    oc_gguf_free(&f);
    unlink(path);
}

/* ─── Metadata KV: uint64 ───────────────────────────────────────────────── */

Test(gguf_writer, add_uint64_metadata)
{
    const char *path = make_temp_path("meta_u64.gguf");
    OcGgufWriter w;
    oc_gguf_writer_init(path, "llama", &w);
    OcError e = oc_gguf_writer_add_uint64(&w, "llama.block_count", 32);
    cr_assert_eq(e, OC_OK, "");
    oc_gguf_writer_finalize(&w);
    oc_gguf_writer_free(&w);

    OcGgufFile f;
    e = oc_gguf_open(path, &f);
    cr_assert_eq(e, OC_OK, "");
    uint64_t val;
    cr_assert(oc_gguf_metadata_get_u64(&f, "llama.block_count", &val), "");
    cr_assert_eq(val, 32, "");
    oc_gguf_free(&f);
    unlink(path);
}

Test(gguf_writer, add_uint64_large_value)
{
    const char *path = make_temp_path("u64big.gguf");
    OcGgufWriter w;
    oc_gguf_writer_init(path, "llama", &w);
    oc_gguf_writer_add_uint64(&w, "test.big", 0x123456789ABCDEFULL);
    oc_gguf_writer_finalize(&w);
    oc_gguf_writer_free(&w);

    OcGgufFile f;
    oc_gguf_open(path, &f);
    uint64_t val;
    cr_assert(oc_gguf_metadata_get_u64(&f, "test.big", &val), "");
    cr_assert_eq(val, 0x123456789ABCDEFULL, "");
    oc_gguf_free(&f);
    unlink(path);
}

/* ─── Metadata KV: float32 ──────────────────────────────────────────────── */

Test(gguf_writer, add_float32_metadata)
{
    const char *path = make_temp_path("meta_f32.gguf");
    OcGgufWriter w;
    oc_gguf_writer_init(path, "llama", &w);
    OcError e = oc_gguf_writer_add_float32(&w, "test.temperature", 0.8f);
    cr_assert_eq(e, OC_OK, "");
    oc_gguf_writer_finalize(&w);
    oc_gguf_writer_free(&w);

    OcGgufFile f;
    e = oc_gguf_open(path, &f);
    cr_assert_eq(e, OC_OK, "");
    float val;
    cr_assert(oc_gguf_metadata_get_f32(&f, "test.temperature", &val), "");
    cr_assert_float_eq(val, 0.8f, 1e-6, "");
    oc_gguf_free(&f);
    unlink(path);
}

Test(gguf_writer, add_float32_zero_and_negative)
{
    const char *path = make_temp_path("f32zn.gguf");
    OcGgufWriter w;
    oc_gguf_writer_init(path, "llama", &w);
    oc_gguf_writer_add_float32(&w, "test.zero", 0.0f);
    oc_gguf_writer_add_float32(&w, "test.neg", -3.14f);
    oc_gguf_writer_finalize(&w);
    oc_gguf_writer_free(&w);

    OcGgufFile f;
    oc_gguf_open(path, &f);
    float val;
    cr_assert(oc_gguf_metadata_get_f32(&f, "test.zero", &val), "");
    cr_assert_float_eq(val, 0.0f, 1e-6, "");
    cr_assert(oc_gguf_metadata_get_f32(&f, "test.neg", &val), "");
    cr_assert_float_eq(val, -3.14f, 1e-4, "");
    oc_gguf_free(&f);
    unlink(path);
}

/* ─── Metadata KV: array<string> ────────────────────────────────────────── */

Test(gguf_writer, add_array_string_metadata)
{
    const char *path = make_temp_path("meta_arr.gguf");
    OcGgufWriter w;
    oc_gguf_writer_init(path, "llama", &w);
    const char *values[] = { "hello", "world", "foo" };
    OcError e = oc_gguf_writer_add_array_string(&w, "test.array", values, 3);
    cr_assert_eq(e, OC_OK, "");
    oc_gguf_writer_finalize(&w);
    oc_gguf_writer_free(&w);

    /* Parse back and verify the array exists. */
    OcGgufFile f;
    e = oc_gguf_open(path, &f);
    cr_assert_eq(e, OC_OK, "");
    const OcGgufMetadataValue *mv = oc_gguf_metadata_get(&f, "test.array");
    cr_assert_not_null(mv, "test.array should exist");
    cr_assert_eq(mv->type, OC_GGUF_MT_ARRAY, "should be ARRAY type");
    cr_assert_eq(mv->v.arr.len, 3, "array should have 3 elements");
    cr_assert_eq(mv->v.arr.elem_type, OC_GGUF_MT_STRING, "elem type should be STRING");
    cr_assert_str_eq(mv->v.arr.values[0].v.str.data, "hello", "");
    cr_assert_str_eq(mv->v.arr.values[1].v.str.data, "world", "");
    cr_assert_str_eq(mv->v.arr.values[2].v.str.data, "foo", "");
    oc_gguf_free(&f);
    unlink(path);
}

Test(gguf_writer, add_array_string_empty)
{
    const char *path = make_temp_path("arr_empty.gguf");
    OcGgufWriter w;
    oc_gguf_writer_init(path, "llama", &w);
    OcError e = oc_gguf_writer_add_array_string(&w, "test.empty", NULL, 0);
    cr_assert_eq(e, OC_OK, "");
    oc_gguf_writer_finalize(&w);
    oc_gguf_writer_free(&w);

    OcGgufFile f;
    e = oc_gguf_open(path, &f);
    cr_assert_eq(e, OC_OK, "");
    const OcGgufMetadataValue *mv = oc_gguf_metadata_get(&f, "test.empty");
    cr_assert_not_null(mv, "");
    cr_assert_eq(mv->type, OC_GGUF_MT_ARRAY, "");
    cr_assert_eq(mv->v.arr.len, 0, "empty array should have 0 elements");
    oc_gguf_free(&f);
    unlink(path);
}

/* ─── Tensor writing ─────────────────────────────────────────────────────── */

Test(gguf_writer, add_single_tensor)
{
    const char *path = make_temp_path("single_tensor.gguf");
    OcGgufWriter w;
    oc_gguf_writer_init(path, "llama", &w);

    uint64_t dims[] = { 4, 3 };  /* 4x3 F32 tensor = 12 floats = 48 bytes */
    float data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    OcError e = oc_gguf_writer_add_tensor(&w, "weight", 2, dims, 0 /*F32*/,
                                          data, sizeof(data));
    cr_assert_eq(e, OC_OK, "");
    cr_assert_eq(w.tensor_count, 1, "");
    oc_gguf_writer_finalize(&w);
    oc_gguf_writer_free(&w);

    /* Parse back and verify. */
    OcGgufFile f;
    e = oc_gguf_open(path, &f);
    cr_assert_eq(e, OC_OK, "");
    cr_assert_eq(f.tensor_count, 1, "");
    const OcGgufTensorInfo *ti = oc_gguf_tensor_get(&f, "weight");
    cr_assert_not_null(ti, "weight tensor should exist");
    cr_assert_str_eq(ti->name, "weight", "");
    cr_assert_eq(ti->n_dims, 2, "");
    cr_assert_eq(ti->dims[0], 4, "");
    cr_assert_eq(ti->dims[1], 3, "");
    cr_assert_eq(ti->ggml_type, 0, ""); /* F32 */
    oc_gguf_free(&f);
    unlink(path);
}

Test(gguf_writer, add_multiple_tensors)
{
    const char *path = make_temp_path("multi_tensor.gguf");
    OcGgufWriter w;
    oc_gguf_writer_init(path, "llama", &w);

    uint64_t dims1[] = { 4 };
    float data1[] = { 1, 2, 3, 4 };
    oc_gguf_writer_add_tensor(&w, "t1", 1, dims1, 0, data1, sizeof(data1));

    uint64_t dims2[] = { 2, 2 };
    float data2[] = { 10, 20, 30, 40 };
    oc_gguf_writer_add_tensor(&w, "t2", 2, dims2, 0, data2, sizeof(data2));

    cr_assert_eq(w.tensor_count, 2, "");
    oc_gguf_writer_finalize(&w);
    oc_gguf_writer_free(&w);

    OcGgufFile f;
    oc_gguf_open(path, &f);
    cr_assert_eq(f.tensor_count, 2, "");
    cr_assert_not_null(oc_gguf_tensor_get(&f, "t1"), "t1 should exist");
    cr_assert_not_null(oc_gguf_tensor_get(&f, "t2"), "t2 should exist");
    oc_gguf_free(&f);
    unlink(path);
}

Test(gguf_writer, add_tensor_null_name)
{
    OcGgufWriter w;
    oc_gguf_writer_init(make_temp_path("tn.gguf"), "llama", &w);
    uint64_t dims[] = { 1 };
    float data[] = { 0 };
    cr_assert_neq(oc_gguf_writer_add_tensor(&w, NULL, 1, dims, 0, data, 4), OC_OK, "");
    oc_gguf_writer_free(&w);
    unlink(make_temp_path("tn.gguf"));
}

Test(gguf_writer, add_tensor_zero_dims)
{
    OcGgufWriter w;
    oc_gguf_writer_init(make_temp_path("td0.gguf"), "llama", &w);
    uint64_t dims[] = { 1 };
    float data[] = { 0 };
    cr_assert_neq(oc_gguf_writer_add_tensor(&w, "t", 0, dims, 0, data, 4), OC_OK, "");
    oc_gguf_writer_free(&w);
    unlink(make_temp_path("td0.gguf"));
}

Test(gguf_writer, add_tensor_too_many_dims)
{
    OcGgufWriter w;
    oc_gguf_writer_init(make_temp_path("tdmax.gguf"), "llama", &w);
    uint64_t dims[OC_GGUF_WRITER_MAX_DIMS + 1] = {0};
    float data[] = { 0 };
    cr_assert_neq(oc_gguf_writer_add_tensor(&w, "t", OC_GGUF_WRITER_MAX_DIMS + 1,
                                            dims, 0, data, 4), OC_OK, "");
    oc_gguf_writer_free(&w);
    unlink(make_temp_path("tdmax.gguf"));
}

Test(gguf_writer, add_tensor_data_null_with_size)
{
    OcGgufWriter w;
    oc_gguf_writer_init(make_temp_path("tdn.gguf"), "llama", &w);
    uint64_t dims[] = { 1 };
    cr_assert_neq(oc_gguf_writer_add_tensor(&w, "t", 1, dims, 0, NULL, 16), OC_OK, "");
    oc_gguf_writer_free(&w);
    unlink(make_temp_path("tdn.gguf"));
}

/* ─── Finalize ───────────────────────────────────────────────────────────── */

Test(gguf_writer, finalize_patches_counts)
{
    const char *path = make_temp_path("counts.gguf");
    OcGgufWriter w;
    oc_gguf_writer_init(path, "llama", &w);
    oc_gguf_writer_add_uint32(&w, "a", 1);
    oc_gguf_writer_add_uint32(&w, "b", 2);

    uint64_t dims[] = { 4 };
    float data[] = { 1, 2, 3, 4 };
    oc_gguf_writer_add_tensor(&w, "t", 1, dims, 0, data, sizeof(data));

    OcError e = oc_gguf_writer_finalize(&w);
    cr_assert_eq(e, OC_OK, "");
    cr_assert_eq(w.finalized, true, "");
    oc_gguf_writer_free(&w);

    /* Verify counts were patched. */
    OcGgufFile f;
    e = oc_gguf_open(path, &f);
    cr_assert_eq(e, OC_OK, "");
    /* 3 KVs: general.architecture + a + b */
    cr_assert_eq(f.metadata_kv_count, 3, "metadata_kv_count should be 3");
    cr_assert_eq(f.tensor_count, 1, "tensor_count should be 1");
    oc_gguf_free(&f);
    unlink(path);
}

Test(gguf_writer, finalize_idempotent)
{
    const char *path = make_temp_path("idem.gguf");
    OcGgufWriter w;
    oc_gguf_writer_init(path, "llama", &w);
    OcError e = oc_gguf_writer_finalize(&w);
    cr_assert_eq(e, OC_OK, "");
    /* Call again — should be a no-op. */
    e = oc_gguf_writer_finalize(&w);
    cr_assert_eq(e, OC_OK, "");
    oc_gguf_writer_free(&w);
    unlink(path);
}

Test(gguf_writer, writes_after_finalize_fail)
{
    const char *path = make_temp_path("afterfin.gguf");
    OcGgufWriter w;
    oc_gguf_writer_init(path, "llama", &w);
    oc_gguf_writer_finalize(&w);
    OcError e = oc_gguf_writer_add_string(&w, "key", "val");
    cr_assert_neq(e, OC_OK, "writes after finalize should fail");
    oc_gguf_writer_free(&w);
    unlink(path);
}

Test(gguf_writer, metadata_after_tensor_fails)
{
    const char *path = make_temp_path("meta_after.gguf");
    OcGgufWriter w;
    oc_gguf_writer_init(path, "llama", &w);
    uint64_t dims[] = { 4 };
    float data[] = { 1, 2, 3, 4 };
    oc_gguf_writer_add_tensor(&w, "t", 1, dims, 0, data, sizeof(data));
    /* Adding metadata after a tensor should fail (metadata must precede tensors). */
    OcError e = oc_gguf_writer_add_string(&w, "late", "value");
    cr_assert_neq(e, OC_OK, "metadata after tensor should fail");
    oc_gguf_writer_free(&w);
    unlink(path);
}

Test(gguf_writer, full_round_trip)
{
    const char *path = make_temp_path("roundtrip.gguf");
    OcGgufWriter w;
    oc_gguf_writer_init(path, "gpt2", &w);

    /* Add various metadata types. */
    oc_gguf_writer_add_string(&w, "general.name", "round-trip-test");
    oc_gguf_writer_add_uint32(&w, "model.context_length", 2048);
    oc_gguf_writer_add_uint64(&w, "model.param_count", 1000000);
    oc_gguf_writer_add_float32(&w, "model.temperature", 1.5f);
    const char *tokens[] = { "hello", "world" };
    oc_gguf_writer_add_array_string(&w, "tokenizer.ggml.tokens", tokens, 2);

    /* Add tensors. */
    uint64_t dims1[] = { 4, 2 };
    float data1[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    oc_gguf_writer_add_tensor(&w, "layer1.weight", 2, dims1, 0, data1, sizeof(data1));

    uint64_t dims2[] = { 3 };
    float data2[] = { 0.1f, 0.2f, 0.3f };
    oc_gguf_writer_add_tensor(&w, "layer1.bias", 1, dims2, 0, data2, sizeof(data2));

    oc_gguf_writer_finalize(&w);
    oc_gguf_writer_free(&w);

    /* Parse back and verify everything. */
    OcGgufFile f;
    OcError e = oc_gguf_open(path, &f);
    cr_assert_eq(e, OC_OK, "");

    /* Header. */
    cr_assert_eq(f.magic, OC_GGUF_MAGIC, "");
    cr_assert_eq(f.version, 3, "");
    cr_assert_eq(f.tensor_count, 2, "");

    /* Metadata: arch + name + u32 + u64 + f32 + array = 6 KVs. */
    cr_assert_eq(f.metadata_kv_count, 6, "expected 6 metadata KVs");

    /* Check general.architecture. */
    const char *arch;
    size_t arch_len;
    cr_assert(oc_gguf_metadata_get_str(&f, "general.architecture", &arch, &arch_len), "");
    cr_assert_str_eq(arch, "gpt2", "");

    /* Check string. */
    const char *name;
    size_t name_len;
    cr_assert(oc_gguf_metadata_get_str(&f, "general.name", &name, &name_len), "");
    cr_assert_str_eq(name, "round-trip-test", "");

    /* Check uint32. */
    uint32_t ctx_len;
    cr_assert(oc_gguf_metadata_get_u32(&f, "model.context_length", &ctx_len), "");
    cr_assert_eq(ctx_len, 2048, "");

    /* Check uint64. */
    uint64_t pc;
    cr_assert(oc_gguf_metadata_get_u64(&f, "model.param_count", &pc), "");
    cr_assert_eq(pc, 1000000, "");

    /* Check float32. */
    float temp;
    cr_assert(oc_gguf_metadata_get_f32(&f, "model.temperature", &temp), "");
    cr_assert_float_eq(temp, 1.5f, 1e-5, "");

    /* Check array. */
    const OcGgufMetadataValue *mv = oc_gguf_metadata_get(&f, "tokenizer.ggml.tokens");
    cr_assert_not_null(mv, "");
    cr_assert_eq(mv->v.arr.len, 2, "");
    cr_assert_str_eq(mv->v.arr.values[0].v.str.data, "hello", "");
    cr_assert_str_eq(mv->v.arr.values[1].v.str.data, "world", "");

    /* Check tensors. */
    const OcGgufTensorInfo *t1 = oc_gguf_tensor_get(&f, "layer1.weight");
    cr_assert_not_null(t1, "");
    cr_assert_eq(t1->n_dims, 2, "");
    cr_assert_eq(t1->dims[0], 4, "");
    cr_assert_eq(t1->dims[1], 2, "");
    cr_assert_eq(t1->ggml_type, 0, "");

    const OcGgufTensorInfo *t2 = oc_gguf_tensor_get(&f, "layer1.bias");
    cr_assert_not_null(t2, "");
    cr_assert_eq(t2->n_dims, 1, "");
    cr_assert_eq(t2->dims[0], 3, "");

    oc_gguf_free(&f);
    unlink(path);
}

Test(gguf_writer, free_null_safe)
{
    oc_gguf_writer_free(NULL);
}

Test(gguf_writer, zero_byte_tensor_data)
{
    const char *path = make_temp_path("zero_data.gguf");
    OcGgufWriter w;
    oc_gguf_writer_init(path, "llama", &w);
    uint64_t dims[] = { 0 };
    /* data_size = 0 is a degenerate but valid case. */
    OcError e = oc_gguf_writer_add_tensor(&w, "empty", 1, dims, 0, NULL, 0);
    cr_assert_eq(e, OC_OK, "");
    oc_gguf_writer_finalize(&w);
    oc_gguf_writer_free(&w);

    OcGgufFile f;
    e = oc_gguf_open(path, &f);
    cr_assert_eq(e, OC_OK, "");
    cr_assert_eq(f.tensor_count, 1, "");
    oc_gguf_free(&f);
    unlink(path);
}
