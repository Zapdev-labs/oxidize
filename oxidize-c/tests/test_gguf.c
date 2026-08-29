/* test_gguf.c — Criterion tests for the GGUF v3/v2 parser. Covers VAL-FOUND-001 (v3 header), VAL-FOUND-002 (v2 compat), VAL-FOUND-003 (all 11 metadata KV value types round-trip), */

/* Expose POSIX helpers used by the multi-shard test (mkdtemp, rmdir, remove). */
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif

#include <criterion/criterion.h>
#include <criterion/redirect.h>

#include "oxidize/gguf.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>   /* for mkdtemp, rmdir */

/* Fixture path: oxidize-core/tests/fixtures/<name>.gguf, relative to the
 * oxidize-c/ directory where `make test` runs. */
#define FIXTURE_DIR "../oxidize-core/tests/fixtures"
#define FIXTURE(name) FIXTURE_DIR "/" name

/* ─── v3 header parse (VAL-FOUND-001, VAL-FOUND-014) ─────────────────────── */

Test(gguf, v3_header_parses_correctly)
{
    OcGgufFile f;
    OcError e = oc_gguf_open(FIXTURE("valid-v3.gguf"), &f);
    cr_assert_eq(e, OC_OK, "valid-v3.gguf should parse, got %s", oc_error_msg(e));

    /* Mirrors Rust `parses_v3_header_tensor_info_and_alignment`. */
    cr_assert_eq(f.magic, OC_GGUF_MAGIC, "magic should be GGUF (0x%08x)", OC_GGUF_MAGIC);
    cr_assert_eq(f.version, 3, "version should be 3");
    cr_assert_eq(f.tensor_count, 1, "tensor_count should be 1");
    cr_assert_eq(f.metadata_kv_count, 1, "metadata_kv_count should be 1");
    cr_assert_eq(f.alignment, 64, "alignment should be 64 (general.alignment=U32:64)");
    cr_assert_eq(f.data_section_start, 128, "data_section_start should be 128");

    oc_gguf_free(&f);
}

Test(gguf, v3_tensor_inventory_bit_exact_vs_rust)
{
    /* Rust reference expectations (from oxidize-core/src/format/gguf.rs */
    OcGgufFile f;
    OcError e = oc_gguf_open(FIXTURE("valid-v3.gguf"), &f);
    cr_assert_eq(e, OC_OK, "parse failed: %s", oc_error_msg(e));

    cr_assert_eq(f.tensor_count, 1, "expected exactly 1 tensor");
    const OcGgufTensorInfo *t = &f.tensors[0];
    cr_assert_str_eq(t->name, "tok_embeddings.weight", "tensor name mismatch");
    cr_assert_eq(t->n_dims, 2, "n_dims should be 2");
    cr_assert_eq(t->dims[0], 32000, "dims[0] should be 32000 (vocab)");
    cr_assert_eq(t->dims[1], 4096,  "dims[1] should be 4096 (hidden)");
    cr_assert_eq(t->absolute_offset, 128, "absolute_offset should be 128");
    cr_assert_eq(t->relative_offset, 0, "relative_offset should be 0");
    cr_assert_eq(t->ggml_type, 0, "ggml_type should be 0 (F32)");

    /* Metadata: general.alignment = U32:64. */
    uint32_t alignment = 0;
    cr_assert(oc_gguf_metadata_get_u32(&f, "general.alignment", &alignment),
              "general.alignment should be a U32");
    cr_assert_eq(alignment, 64, "general.alignment value should be 64");

    oc_gguf_free(&f);
}

Test(gguf, v3_data_section_first_four_bytes)
{
    /* The valid-v3 fixture stores [1, 2, 3, 4] at offset 128 (the data
     * section). Verify the parser exposes a backing buffer we can inspect. */
    OcGgufFile f;
    OcError e = oc_gguf_open(FIXTURE("valid-v3.gguf"), &f);
    cr_assert_eq(e, OC_OK, "parse failed: %s", oc_error_msg(e));

    cr_assert_not_null(f.backing_buf, "oc_gguf_open should populate backing_buf");
    cr_assert_eq(f.backing_len, 132, "fixture is 132 bytes (128 header + 4 data)");
    cr_assert_eq(f.backing_buf[128], 1, "data[0] should be 1");
    cr_assert_eq(f.backing_buf[129], 2, "data[1] should be 2");
    cr_assert_eq(f.backing_buf[130], 3, "data[2] should be 3");
    cr_assert_eq(f.backing_buf[131], 4, "data[3] should be 4");

    oc_gguf_free(&f);
}

/* ─── v2 backward compatibility (VAL-FOUND-002) ──────────────────────────── */

Test(gguf, v2_backward_compat)
{
    /* Synthesize a v2 GGUF by copying valid-v3 and rewriting the version */
    OcGgufFile v3;
    OcError e = oc_gguf_open(FIXTURE("valid-v3.gguf"), &v3);
    cr_assert_eq(e, OC_OK, "open v3: %s", oc_error_msg(e));

    /* Copy the backing buffer and rewrite the version byte. */
    uint8_t *buf = malloc(v3.backing_len);
    cr_assert_not_null(buf, "malloc");
    memcpy(buf, v3.backing_buf, v3.backing_len);
    size_t len = v3.backing_len;
    buf[4] = 2;  /* version u32 little-endian, low byte */
    oc_gguf_free(&v3);

    OcGgufFile v2;
    e = oc_gguf_parse(buf, len, &v2);
    cr_assert_eq(e, OC_OK, "v2 parse should succeed: %s", oc_error_msg(e));
    cr_assert_eq(v2.version, 2, "version should be 2");
    cr_assert_eq(v2.tensor_count, 1, "tensor_count should match v3");
    cr_assert_eq(v2.metadata_kv_count, 1, "metadata_kv_count should match v3");
    cr_assert_eq(v2.alignment, 64, "alignment should match v3");
    cr_assert_eq(v2.data_section_start, 128, "data_section_start should match v3");
    cr_assert_str_eq(v2.tensors[0].name, "tok_embeddings.weight", "tensor name should match v3");
    cr_assert_eq(v2.tensors[0].dims[0], 32000, "dims[0] should match v3");
    cr_assert_eq(v2.tensors[0].dims[1], 4096, "dims[1] should match v3");
    cr_assert_eq(v2.tensors[0].absolute_offset, 128, "absolute_offset should match v3");

    oc_gguf_free(&v2);
    free(buf);
}

/* ─── All 11 metadata KV value types round-trip (VAL-FOUND-003) ──────────── */

/* Helper: build a synthetic GGUF byte buffer in memory containing one KV per
 * metadata value type. Layout mirrors the GGUF spec. The returned buffer is
 * zero-padded to `cap` so the data_section_start alignment check passes. */
static uint8_t *build_all_types_gguf(size_t *out_len)
{
    /* We'll build the buffer in a fixed-size scratch array then copy. */
    size_t cap = 1024;
    uint8_t *buf = calloc(cap, 1);   /* zeroed: data_section_start padding is OK */
    cr_assert_not_null(buf, "calloc");
    size_t off = 0;

#define EMIT(buf, off, src, n) do { memcpy((buf) + (off), (src), (n)); (off) += (n); } while (0)
#define EMIT_U8(buf, off, v)  do { uint8_t _x = (uint8_t)(v); EMIT(buf, off, &_x, 1); } while (0)
#define EMIT_U32(buf, off, v) do { uint32_t _x = (uint32_t)(v); EMIT(buf, off, &_x, 4); } while (0)
#define EMIT_U64(buf, off, v) do { uint64_t _x = (uint64_t)(v); EMIT(buf, off, &_x, 8); } while (0)

    /* Header: magic, version=3, tensor_count=0, kv_count=11. */
    EMIT_U32(buf, off, OC_GGUF_MAGIC);
    EMIT_U32(buf, off, 3);
    EMIT_U64(buf, off, 0);   /* tensor_count */
    EMIT_U64(buf, off, 11);  /* kv_count = 11 types (incl. ARRAY) */

    /* KV helper: write a key + value_type + value. */
    /* U8 */
    {
        const char *k = "k.u8"; uint64_t kl = 4; uint32_t vt = OC_GGUF_MT_UINT8;
        EMIT_U64(buf, off, kl); EMIT(buf, off, k, kl); EMIT_U32(buf, off, vt); EMIT_U8(buf, off, 200);
    }
    /* U16 */
    {
        const char *k = "k.u16"; uint64_t kl = 5; uint32_t vt = OC_GGUF_MT_UINT16;
        uint16_t v = 50000;
        EMIT_U64(buf, off, kl); EMIT(buf, off, k, kl); EMIT_U32(buf, off, vt); EMIT(buf, off, &v, 2);
    }
    /* U32 */
    {
        const char *k = "k.u32"; uint64_t kl = 5; uint32_t vt = OC_GGUF_MT_UINT32;
        uint32_t v = 4000000000u;
        EMIT_U64(buf, off, kl); EMIT(buf, off, k, kl); EMIT_U32(buf, off, vt); EMIT(buf, off, &v, 4);
    }
    /* I8 */
    {
        const char *k = "k.i8"; uint64_t kl = 4; uint32_t vt = OC_GGUF_MT_INT8;
        int8_t v = -50;
        EMIT_U64(buf, off, kl); EMIT(buf, off, k, kl); EMIT_U32(buf, off, vt); EMIT(buf, off, &v, 1);
    }
    /* I16 */
    {
        const char *k = "k.i16"; uint64_t kl = 5; uint32_t vt = OC_GGUF_MT_INT16;
        int16_t v = -20000;
        EMIT_U64(buf, off, kl); EMIT(buf, off, k, kl); EMIT_U32(buf, off, vt); EMIT(buf, off, &v, 2);
    }
    /* I32 */
    {
        const char *k = "k.i32"; uint64_t kl = 5; uint32_t vt = OC_GGUF_MT_INT32;
        int32_t v = -1000000000;
        EMIT_U64(buf, off, kl); EMIT(buf, off, k, kl); EMIT_U32(buf, off, vt); EMIT(buf, off, &v, 4);
    }
    /* F32 */
    {
        const char *k = "k.f32"; uint64_t kl = 5; uint32_t vt = OC_GGUF_MT_FLOAT32;
        float v = 3.5f;
        EMIT_U64(buf, off, kl); EMIT(buf, off, k, kl); EMIT_U32(buf, off, vt); EMIT(buf, off, &v, 4);
    }
    /* F64 */
    {
        const char *k = "k.f64"; uint64_t kl = 5; uint32_t vt = OC_GGUF_MT_FLOAT64;
        double v = 2.718281828459045;
        EMIT_U64(buf, off, kl); EMIT(buf, off, k, kl); EMIT_U32(buf, off, vt); EMIT(buf, off, &v, 8);
    }
    /* BOOL */
    {
        const char *k = "k.bool"; uint64_t kl = 6; uint32_t vt = OC_GGUF_MT_BOOL;
        uint8_t v = 1;
        EMIT_U64(buf, off, kl); EMIT(buf, off, k, kl); EMIT_U32(buf, off, vt); EMIT(buf, off, &v, 1);
    }
    /* STRING */
    {
        const char *k = "k.str"; uint64_t kl = 5; uint32_t vt = OC_GGUF_MT_STRING;
        const char *s = "hello, gguf"; uint64_t sl = 11;
        EMIT_U64(buf, off, kl); EMIT(buf, off, k, kl); EMIT_U32(buf, off, vt);
        EMIT_U64(buf, off, sl); EMIT(buf, off, s, (size_t)sl);
    }
    /* ARRAY of U32 (3 elements) */
    {
        const char *k = "k.arr"; uint64_t kl = 5; uint32_t vt = OC_GGUF_MT_ARRAY;
        uint32_t et = OC_GGUF_MT_UINT32; uint64_t alen = 3;
        uint32_t v0 = 10, v1 = 20, v2 = 30;
        EMIT_U64(buf, off, kl); EMIT(buf, off, k, kl); EMIT_U32(buf, off, vt);
        EMIT_U32(buf, off, et); EMIT_U64(buf, off, alen);
        EMIT(buf, off, &v0, 4); EMIT(buf, off, &v1, 4); EMIT(buf, off, &v2, 4);
    }

    *out_len = cap;   /* return full cap so data_section_start fits */
    return buf;

#undef EMIT
#undef EMIT_U8
#undef EMIT_U32
#undef EMIT_U64
}

Test(gguf, all_11_metadata_value_types_round_trip)
{
    size_t len = 0;
    uint8_t *buf = build_all_types_gguf(&len);

    OcGgufFile f;
    OcError e = oc_gguf_parse(buf, len, &f);
    cr_assert_eq(e, OC_OK, "parse should succeed: %s", oc_error_msg(e));
    cr_assert_eq(f.metadata_kv_count, 11, "expected 11 KV entries");

    /* U8 */
    {
        const OcGgufMetadataValue *v = oc_gguf_metadata_get(&f, "k.u8");
        cr_assert_not_null(v, "k.u8 missing");
        cr_assert_eq(v->type, OC_GGUF_MT_UINT8, "k.u8 type");
        cr_assert_eq(v->v.u8, 200, "k.u8 value");
        uint8_t g = 0; cr_assert(oc_gguf_metadata_get_u8(&f, "k.u8", &g), "getter u8");
        cr_assert_eq(g, 200, "getter u8 value");
    }
    /* U16 */
    {
        const OcGgufMetadataValue *v = oc_gguf_metadata_get(&f, "k.u16");
        cr_assert_not_null(v, "k.u16 missing");
        cr_assert_eq(v->type, OC_GGUF_MT_UINT16, "k.u16 type");
        cr_assert_eq(v->v.u16, 50000, "k.u16 value");
    }
    /* U32 */
    {
        const OcGgufMetadataValue *v = oc_gguf_metadata_get(&f, "k.u32");
        cr_assert_not_null(v, "k.u32 missing");
        cr_assert_eq(v->type, OC_GGUF_MT_UINT32, "k.u32 type");
        cr_assert_eq(v->v.u32, 4000000000u, "k.u32 value");
    }
    /* I8 */
    {
        const OcGgufMetadataValue *v = oc_gguf_metadata_get(&f, "k.i8");
        cr_assert_not_null(v, "k.i8 missing");
        cr_assert_eq(v->type, OC_GGUF_MT_INT8, "k.i8 type");
        cr_assert_eq(v->v.i8, -50, "k.i8 value");
    }
    /* I16 */
    {
        const OcGgufMetadataValue *v = oc_gguf_metadata_get(&f, "k.i16");
        cr_assert_not_null(v, "k.i16 missing");
        cr_assert_eq(v->type, OC_GGUF_MT_INT16, "k.i16 type");
        cr_assert_eq(v->v.i16, -20000, "k.i16 value");
    }
    /* I32 */
    {
        const OcGgufMetadataValue *v = oc_gguf_metadata_get(&f, "k.i32");
        cr_assert_not_null(v, "k.i32 missing");
        cr_assert_eq(v->type, OC_GGUF_MT_INT32, "k.i32 type");
        cr_assert_eq(v->v.i32, -1000000000, "k.i32 value");
        int32_t g = 0; cr_assert(oc_gguf_metadata_get_i32(&f, "k.i32", &g), "getter i32");
        cr_assert_eq(g, -1000000000, "getter i32 value");
    }
    /* F32 */
    {
        const OcGgufMetadataValue *v = oc_gguf_metadata_get(&f, "k.f32");
        cr_assert_not_null(v, "k.f32 missing");
        cr_assert_eq(v->type, OC_GGUF_MT_FLOAT32, "k.f32 type");
        /* bit-exact compare avoids FP equality surprises */
        cr_assert_eq(v->v.f32, 3.5f, "k.f32 value");
        float g = 0; cr_assert(oc_gguf_metadata_get_f32(&f, "k.f32", &g), "getter f32");
        cr_assert_eq(g, 3.5f, "getter f32 value");
    }
    /* F64 */
    {
        const OcGgufMetadataValue *v = oc_gguf_metadata_get(&f, "k.f64");
        cr_assert_not_null(v, "k.f64 missing");
        cr_assert_eq(v->type, OC_GGUF_MT_FLOAT64, "k.f64 type");
        cr_assert_eq(v->v.f64, 2.718281828459045, "k.f64 value");
        double g = 0; cr_assert(oc_gguf_metadata_get_f64(&f, "k.f64", &g), "getter f64");
        cr_assert_eq(g, 2.718281828459045, "getter f64 value");
    }
    /* BOOL */
    {
        const OcGgufMetadataValue *v = oc_gguf_metadata_get(&f, "k.bool");
        cr_assert_not_null(v, "k.bool missing");
        cr_assert_eq(v->type, OC_GGUF_MT_BOOL, "k.bool type");
        cr_assert_eq(v->v.b, true, "k.bool value");
        bool g = false; cr_assert(oc_gguf_metadata_get_bool(&f, "k.bool", &g), "getter bool");
        cr_assert_eq(g, true, "getter bool value");
    }
    /* STRING */
    {
        const char *s = NULL; size_t sl = 0;
        cr_assert(oc_gguf_metadata_get_str(&f, "k.str", &s, &sl), "getter str");
        cr_assert_eq(sl, 11, "string length");
        cr_assert_not_null(s, "string ptr");
        cr_assert(strncmp(s, "hello, gguf", 11) == 0, "string value (NUL-safe compare)");
    }
    /* ARRAY of U32 */
    {
        const OcGgufMetadataValue *v = oc_gguf_metadata_get(&f, "k.arr");
        cr_assert_not_null(v, "k.arr missing");
        cr_assert_eq(v->type, OC_GGUF_MT_ARRAY, "k.arr type");
        cr_assert_eq(v->v.arr.elem_type, OC_GGUF_MT_UINT32, "array elem type");
        cr_assert_eq(v->v.arr.len, 3, "array length");
        cr_assert_eq(v->v.arr.values[0].v.u32, 10, "arr[0]");
        cr_assert_eq(v->v.arr.values[1].v.u32, 20, "arr[1]");
        cr_assert_eq(v->v.arr.values[2].v.u32, 30, "arr[2]");
    }

    oc_gguf_free(&f);
    free(buf);
}

Test(gguf, string_value_with_embedded_nul_preserved)
{
    /* GGUF strings are length-prefixed; embedded NULs are legal. The parser
     * must store the raw length and not truncate. */
    size_t cap = 128;  /* generous: data_section_start = align_up(53, 32) = 64 */
    uint8_t *buf = calloc(cap, 1);
    cr_assert_not_null(buf);
    size_t off = 0;
    uint32_t magic = OC_GGUF_MAGIC, ver = 3, vt = OC_GGUF_MT_STRING;
    uint64_t tc = 0, kvc = 1, kl = 5, sl = 5;
    const char *key = "k.nul";      /* 5 chars: 'k','.','n','u','l' */
    const char *payload = "ab\0cd";  /* 5 bytes with embedded NUL at index 2 */

    memcpy(buf + off, &magic, 4); off += 4;
    memcpy(buf + off, &ver,   4); off += 4;
    memcpy(buf + off, &tc,    8); off += 8;
    memcpy(buf + off, &kvc,   8); off += 8;
    memcpy(buf + off, &kl,    8); off += 8;
    memcpy(buf + off, key,   5); off += 5;
    memcpy(buf + off, &vt,    4); off += 4;
    memcpy(buf + off, &sl,    8); off += 8;
    memcpy(buf + off, payload, 5); off += 5;

    /* Use `cap` as the buffer length so data_section_start (64) fits. */
    OcGgufFile f;
    OcError e = oc_gguf_parse(buf, cap, &f);
    cr_assert_eq(e, OC_OK, "parse: %s", oc_error_msg(e));

    const char *s = NULL; size_t slen = 0;
    cr_assert(oc_gguf_metadata_get_str(&f, "k.nul", &s, &slen), "get str");
    cr_assert_eq(slen, 5, "embedded-NUL string length must be 5");
    cr_assert_eq(s[0], 'a', "s[0]");
    cr_assert_eq(s[1], 'b', "s[1]");
    cr_assert_eq(s[2], '\0', "s[2] (embedded NUL)");
    cr_assert_eq(s[3], 'c', "s[3]");
    cr_assert_eq(s[4], 'd', "s[4]");

    oc_gguf_free(&f);
    free(buf);
}

/* ─── Malformed GGUF (VAL-FOUND-013) ─────────────────────────────────────── */

Test(gguf, invalid_magic_returns_format_error)
{
    OcGgufFile f;
    OcError e = oc_gguf_open(FIXTURE("invalid-magic.gguf"), &f);
    cr_assert_eq(e, OC_ERR_FORMAT, "invalid magic must return OC_ERR_FORMAT, got %s", oc_error_msg(e));
    cr_assert_str_neq(oc_error_msg(e), "", "error message must be non-empty");
}

Test(gguf, unsupported_version_returns_format_error)
{
    OcGgufFile f;
    OcError e = oc_gguf_open(FIXTURE("unsupported-version.gguf"), &f);
    cr_assert_eq(e, OC_ERR_FORMAT, "unsupported version must return OC_ERR_FORMAT, got %s", oc_error_msg(e));
}

Test(gguf, invalid_alignment_returns_format_error)
{
    OcGgufFile f;
    OcError e = oc_gguf_open(FIXTURE("invalid-alignment.gguf"), &f);
    cr_assert_eq(e, OC_ERR_FORMAT, "invalid alignment must return OC_ERR_FORMAT, got %s", oc_error_msg(e));
}

Test(gguf, truncated_header_returns_format_error_no_segfault)
{
    /* A buffer with only the magic + version (8 bytes) — not enough for the
     * two u64 counts. Parser must return OC_ERR_FORMAT, not crash. */
    uint8_t buf[8];
    uint32_t magic = OC_GGUF_MAGIC, ver = 3;
    memcpy(buf, &magic, 4);
    memcpy(buf + 4, &ver, 4);

    OcGgufFile f;
    OcError e = oc_gguf_parse(buf, sizeof(buf), &f);
    cr_assert_eq(e, OC_ERR_FORMAT, "truncated header must return OC_ERR_FORMAT");
}

Test(gguf, zero_byte_buffer_returns_format_error)
{
    OcGgufFile f;
    OcError e = oc_gguf_parse((const uint8_t *)"", 0, &f);
    cr_assert_eq(e, OC_ERR_FORMAT, "empty buffer must return OC_ERR_FORMAT");
}

Test(gguf, truncated_metadata_returns_format_error)
{
    /* Header claims 1 KV but no KV bytes follow. */
    uint8_t buf[24];
    uint32_t magic = OC_GGUF_MAGIC, ver = 3;
    uint64_t tc = 0, kvc = 1;
    memcpy(buf, &magic, 4);
    memcpy(buf + 4, &ver, 4);
    memcpy(buf + 8, &tc, 8);
    memcpy(buf + 16, &kvc, 8);

    OcGgufFile f;
    OcError e = oc_gguf_parse(buf, sizeof(buf), &f);
    cr_assert_eq(e, OC_ERR_FORMAT, "truncated metadata must return OC_ERR_FORMAT");
}

Test(gguf, truncated_tensor_table_returns_format_error)
{
    /* Header claims 1 tensor but no tensor bytes follow. */
    uint8_t buf[24];
    uint32_t magic = OC_GGUF_MAGIC, ver = 3;
    uint64_t tc = 1, kvc = 0;
    memcpy(buf, &magic, 4);
    memcpy(buf + 4, &ver, 4);
    memcpy(buf + 8, &tc, 8);
    memcpy(buf + 16, &kvc, 8);

    OcGgufFile f;
    OcError e = oc_gguf_parse(buf, sizeof(buf), &f);
    cr_assert_eq(e, OC_ERR_FORMAT, "truncated tensor table must return OC_ERR_FORMAT");
}

Test(gguf, absurd_counts_rejected)
{
    /* Header claims 2^40 tensors in a 24-byte buffer. */
    uint8_t buf[24];
    uint32_t magic = OC_GGUF_MAGIC, ver = 3;
    uint64_t tc = (uint64_t)1 << 40, kvc = 0;
    memcpy(buf, &magic, 4);
    memcpy(buf + 4, &ver, 4);
    memcpy(buf + 8, &tc, 8);
    memcpy(buf + 16, &kvc, 8);

    OcGgufFile f;
    OcError e = oc_gguf_parse(buf, sizeof(buf), &f);
    cr_assert_eq(e, OC_ERR_FORMAT, "absurd counts must return OC_ERR_FORMAT");
}

Test(gguf, unknown_metadata_value_type_rejected)
{
    /* Build a GGUF with one KV whose value_type is 99 (unknown). */
    uint8_t buf[64];
    size_t off = 0;
    uint32_t magic = OC_GGUF_MAGIC, ver = 3, vt = 99;
    uint64_t tc = 0, kvc = 1, kl = 4;
    const char *key = "k.x";
    memcpy(buf + off, &magic, 4); off += 4;
    memcpy(buf + off, &ver,   4); off += 4;
    memcpy(buf + off, &tc,    8); off += 8;
    memcpy(buf + off, &kvc,   8); off += 8;
    memcpy(buf + off, &kl,    8); off += 8;
    memcpy(buf + off, key,   4); off += 4;
    memcpy(buf + off, &vt,    4); off += 4;

    OcGgufFile f;
    OcError e = oc_gguf_parse(buf, off, &f);
    cr_assert_eq(e, OC_ERR_FORMAT, "unknown metadata type must return OC_ERR_FORMAT");
}

Test(gguf, nested_array_metadata_rejected)
{
    /* GGUF spec forbids arrays-of-arrays (ARRAY element_type must be a scalar */
    uint8_t buf[64];
    size_t off = 0;
    uint32_t magic = OC_GGUF_MAGIC, ver = 3;
    uint32_t kv_vt = OC_GGUF_MT_ARRAY;     /* top-level value is ARRAY */
    uint32_t arr_elem_type = OC_GGUF_MT_ARRAY;  /* nested ARRAY (non-spec) */
    uint64_t arr_len = 0;                  /* 0 elements — still rejected */
    uint64_t tc = 0, kvc = 1, kl = 5;
    const char *key = "k.a2";
    memcpy(buf + off, &magic, 4); off += 4;
    memcpy(buf + off, &ver,   4); off += 4;
    memcpy(buf + off, &tc,    8); off += 8;
    memcpy(buf + off, &kvc,   8); off += 8;
    memcpy(buf + off, &kl,    8); off += 8;
    memcpy(buf + off, key,   5); off += 5;
    memcpy(buf + off, &kv_vt,        4); off += 4;  /* KV value_type = ARRAY */
    memcpy(buf + off, &arr_elem_type, 4); off += 4; /* array element_type = ARRAY */
    memcpy(buf + off, &arr_len,       8); off += 8; /* array length = 0 */

    OcGgufFile f;
    OcError e = oc_gguf_parse(buf, off, &f);
    cr_assert_eq(e, OC_ERR_FORMAT,
        "nested ARRAY element_type must return OC_ERR_FORMAT, got %s",
        oc_error_msg(e));
}

Test(gguf, missing_file_returns_io_error)
{
    OcGgufFile f;
    OcError e = oc_gguf_open("/tmp/oxidize-c-gguf-nonexistent-xxxx.gguf", &f);
    cr_assert_eq(e, OC_ERR_IO, "missing file must return OC_ERR_IO, got %s", oc_error_msg(e));
}

Test(gguf, null_args_rejected)
{
    OcGgufFile f;
    cr_assert_eq(oc_gguf_parse(NULL, 0, &f), OC_ERR_INVALID_ARG, "NULL buf");
    uint8_t b = 0;
    cr_assert_eq(oc_gguf_parse(&b, 1, NULL), OC_ERR_INVALID_ARG, "NULL out");
    cr_assert_eq(oc_gguf_open(NULL, &f), OC_ERR_INVALID_ARG, "NULL path");
    cr_assert_eq(oc_gguf_open("foo", NULL), OC_ERR_INVALID_ARG, "NULL out (open)");
    oc_gguf_free(NULL);  /* safe on NULL */
}


Test(gguf, metadata_type_round_trip)
{
    for (uint32_t i = 0; i < (uint32_t)OC_GGUF_MT__COUNT; i++) {
        OcGgufMetadataType t = oc_gguf_metadata_type_from_u32(i);
        cr_assert_eq(t, (OcGgufMetadataType)i, "type %u should round-trip", i);
        cr_assert_not_null(oc_gguf_metadata_type_name(t), "name should not be NULL");
    }
    cr_assert_eq(oc_gguf_metadata_type_from_u32(99), OC_GGUF_MT_UNKNOWN, "99 should be UNKNOWN");
    cr_assert_str_eq(oc_gguf_metadata_type_name(OC_GGUF_MT_STRING), "STRING", "STRING name");
    cr_assert_str_eq(oc_gguf_metadata_type_name(OC_GGUF_MT_ARRAY),  "ARRAY",  "ARRAY name");
    cr_assert_str_eq(oc_gguf_metadata_type_name(OC_GGUF_MT_BOOL),   "BOOL",   "BOOL name");
}

Test(gguf, tensor_lookup_by_name)
{
    OcGgufFile f;
    OcError e = oc_gguf_open(FIXTURE("valid-v3.gguf"), &f);
    cr_assert_eq(e, OC_OK, "open: %s", oc_error_msg(e));

    const OcGgufTensorInfo *t = oc_gguf_tensor_get(&f, "tok_embeddings.weight");
    cr_assert_not_null(t, "tensor should be found");
    cr_assert_str_eq(t->name, "tok_embeddings.weight", "name");

    cr_assert_null(oc_gguf_tensor_get(&f, "does.not.exist"), "missing tensor should return NULL");

    oc_gguf_free(&f);
}

Test(gguf, parse_from_caller_buffer_does_not_own_it)
{
    /* oc_gguf_parse should NOT take ownership of the caller's buffer
     * (out->backing_buf must be NULL so oc_gguf_free won't double-free). */
    OcGgufFile f;
    OcError e = oc_gguf_open(FIXTURE("valid-v3.gguf"), &f);
    cr_assert_eq(e, OC_OK, "open: %s", oc_error_msg(e));
    cr_assert_not_null(f.backing_buf, "open should own the buffer");

    /* Re-parse the same bytes via oc_gguf_parse: backing_buf must be NULL. */
    OcGgufFile f2;
    e = oc_gguf_parse(f.backing_buf, f.backing_len, &f2);
    cr_assert_eq(e, OC_OK, "parse: %s", oc_error_msg(e));
    cr_assert_null(f2.backing_buf, "parse must not own caller's buffer");

    oc_gguf_free(&f2);
    oc_gguf_free(&f);
}

/* ─── Multi-shard unified view (VAL-FOUND-005) ────────────────────────── */

/* Helper: build a minimal valid GGUF buffer with the given tensor name +
 * 4-byte data payload. Used to synthesize shard fixtures in a temp dir. */
static uint8_t *build_minimal_shard(const char *tensor_name,
                                    const uint8_t data[4],
                                    size_t *out_len)
{
    /* Layout: */
    size_t name_len = strlen(tensor_name);
    size_t header = 24;
    size_t tensor_info = 8 + name_len + 4 + 16 + 4 + 8;
    size_t pre_data = header + tensor_info;
    /* align to 32 */
    size_t data_start = (pre_data + 31) & ~(size_t)31;
    size_t total = data_start + 4;

    uint8_t *buf = calloc(total, 1);
    cr_assert_not_null(buf, "calloc");

    size_t off = 0;
    uint32_t magic = OC_GGUF_MAGIC, ver = 3;
    uint64_t tc = 1, kvc = 0;
    memcpy(buf + off, &magic, 4); off += 4;
    memcpy(buf + off, &ver,   4); off += 4;
    memcpy(buf + off, &tc,    8); off += 8;
    memcpy(buf + off, &kvc,   8); off += 8;
    /* tensor name */
    uint64_t nl = name_len;
    memcpy(buf + off, &nl, 8); off += 8;
    memcpy(buf + off, tensor_name, name_len); off += name_len;
    /* n_dims = 2 */
    uint32_t nd = 2;
    memcpy(buf + off, &nd, 4); off += 4;
    /* dims = [1, 1] (1 element, 1 row) */
    uint64_t d0 = 1, d1 = 1;
    memcpy(buf + off, &d0, 8); off += 8;
    memcpy(buf + off, &d1, 8); off += 8;
    /* ggml_type = 0 (F32) */
    uint32_t gt = 0;
    memcpy(buf + off, &gt, 4); off += 4;
    /* relative_offset = 0 (data starts at data_section_start) */
    uint64_t ro = 0;
    memcpy(buf + off, &ro, 8); off += 8;
    /* pad to data_start */
    while (off < data_start) {
        buf[off++] = 0;
    }
    /* data */
    memcpy(buf + data_start, data, 4);

    *out_len = total;
    return buf;
}

Test(gguf, multishard_unified_view)
{
    /* VAL-FOUND-005: open model-00001-of-00003.gguf and the parser exposes
     * a single unified tensor view across all 3 shards. */
    char tmpl[] = "/tmp/oxidize-c-multishard-XXXXXX";
    char *dir = mkdtemp(tmpl);
    cr_assert_not_null(dir, "mkdtemp");

    /* Build 3 shards, each containing one uniquely-named tensor. */
    const char *names[3] = {
        "tok_embeddings.weight",
        "blk.0.attn_q.weight",
        "output.weight",
    };
    uint8_t data[3][4] = {
        {10, 20, 30, 40},
        {50, 60, 70, 80},
        {90, 100, 110, 120},
    };

    char paths[3][512];
    for (int i = 0; i < 3; i++) {
        snprintf(paths[i], sizeof(paths[i]), "%s/model-%05d-of-%05d.gguf",
                 dir, i + 1, 3);
        size_t len = 0;
        uint8_t *buf = build_minimal_shard(names[i], data[i], &len);
        cr_assert_not_null(buf, "build shard %d", i);
        FILE *f = fopen(paths[i], "wb");
        cr_assert_not_null(f, "fopen shard %d", i);
        cr_assert_eq(fwrite(buf, 1, len, f), len, "fwrite shard %d", i);
        fclose(f);
        free(buf);
    }

    /* Open shard 1 via oc_gguf_map_open — should detect the multi-shard
     * pattern and load all 3 shards. */
    OcGgufMmappedFile m;
    OcError e = oc_gguf_map_open(paths[0], &m);
    cr_assert_eq(e, OC_OK, "map_open: %s", oc_error_msg(e));
    cr_assert_eq(m.n_shards, 3, "expected 3 shards");
    cr_assert_eq(m.unified.tensor_count, 3, "expected 3 tensors in unified view");

    /* Verify each tensor is accessible with the right name + data. */
    for (int i = 0; i < 3; i++) {
        const OcGgufTensorInfo *t = &m.unified.tensors[i];
        cr_assert_str_eq(t->name, names[i], "tensor %d name", i);
        cr_assert_eq(t->shard_index, (uint32_t)i, "tensor %d shard_index", i);
        const uint8_t *d = oc_gguf_map_tensor_data(&m, t);
        cr_assert_not_null(d, "tensor %d data", i);
        cr_assert_eq(d[0], data[i][0], "tensor %d data[0]", i);
        cr_assert_eq(d[1], data[i][1], "tensor %d data[1]", i);
        cr_assert_eq(d[2], data[i][2], "tensor %d data[2]", i);
        cr_assert_eq(d[3], data[i][3], "tensor %d data[3]", i);
    }

    /* Look up a tensor by name via oc_gguf_map_tensor_get. */
    const OcGgufTensorInfo *t = oc_gguf_map_tensor_get(&m, "blk.0.attn_q.weight");
    cr_assert_not_null(t, "tensor_get should find blk.0.attn_q.weight");
    cr_assert_str_eq(t->name, "blk.0.attn_q.weight", "name");

    oc_gguf_map_free(&m);

    /* Cleanup temp dir. */
    for (int i = 0; i < 3; i++) {
        remove(paths[i]);
    }
    rmdir(dir);
}

Test(gguf, multishard_falls_back_to_single_when_siblings_missing)
{
    /* If the path matches the pattern but sibling shards don't exist, fall
     * back to a single-shard load. */
    char tmpl[] = "/tmp/oxidize-c-single-XXXXXX";
    char *dir = mkdtemp(tmpl);
    cr_assert_not_null(dir, "mkdtemp");

    /* Write only shard 1 (shards 2 and 3 don't exist). */
    char path[512];
    snprintf(path, sizeof(path), "%s/model-%05d-of-%05d.gguf", dir, 1, 3);
    uint8_t data[4] = {1, 2, 3, 4};
    size_t len = 0;
    uint8_t *buf = build_minimal_shard("tok_embeddings.weight", data, &len);
    FILE *f = fopen(path, "wb");
    cr_assert_not_null(f, "fopen");
    cr_assert_eq(fwrite(buf, 1, len, f), len, "fwrite");
    fclose(f);
    free(buf);

    /* Should load as single shard (siblings missing). */
    OcGgufMmappedFile m;
    OcError e = oc_gguf_map_open(path, &m);
    cr_assert_eq(e, OC_OK, "map_open: %s", oc_error_msg(e));
    cr_assert_eq(m.n_shards, 1, "should fall back to single shard");
    cr_assert_eq(m.unified.tensor_count, 1, "tensor_count");

    oc_gguf_map_free(&m);
    remove(path);
    rmdir(dir);
}

Test(gguf, mapped_tensor_infos_returns_mapped_names)
{
    /* oc_gguf_map_mapped_tensor_infos() returns a fresh array with mapped */
    OcGgufMmappedFile m;
    OcError e = oc_gguf_map_open(FIXTURE("valid-v3.gguf"), &m);
    cr_assert_eq(e, OC_OK, "map_open: %s", oc_error_msg(e));

    OcArena *arena = oc_arena_new(0);
    cr_assert_not_null(arena, "arena");

    OcGgufTensorInfo *infos = NULL;
    size_t count = 0;
    e = oc_gguf_map_mapped_tensor_infos(&m, arena, &infos, &count);
    cr_assert_eq(e, OC_OK, "mapped_tensor_infos: %s", oc_error_msg(e));
    cr_assert_eq(count, 1, "count should be 1");
    cr_assert_not_null(infos, "infos should be non-NULL");
    /* Unknown arch → name passes through unchanged. */
    cr_assert_str_eq(infos[0].name, "tok_embeddings.weight", "name");
    cr_assert_eq(infos[0].shard_index, 0, "shard_index");
    cr_assert_eq(infos[0].absolute_offset, 128, "absolute_offset");

    oc_arena_free(arena);
    oc_gguf_map_free(&m);
}

Test(gguf, arch_from_file_detects_general_architecture_key)
{
    /* Build a synthetic GGUF with `general.architecture = "llama"` and verify
     * oc_gguf_arch_from_file() returns OC_ARCH_LLAMA. */
    size_t cap = 256;
    uint8_t *buf = calloc(cap, 1);
    cr_assert_not_null(buf);
    size_t off = 0;
    uint32_t magic = OC_GGUF_MAGIC, ver = 3, vt = OC_GGUF_MT_STRING;
    uint64_t tc = 0, kvc = 1;
    const char *key = "general.architecture";   /* 20 chars */
    size_t kl = strlen(key);
    uint64_t kl64 = kl;
    const char *val = "llama";                    /* 5 chars */
    size_t sl = strlen(val);
    uint64_t sl64 = sl;

    memcpy(buf + off, &magic, 4); off += 4;
    memcpy(buf + off, &ver,   4); off += 4;
    memcpy(buf + off, &tc,    8); off += 8;
    memcpy(buf + off, &kvc,   8); off += 8;
    memcpy(buf + off, &kl64,  8); off += 8;
    memcpy(buf + off, key,   kl); off += kl;
    memcpy(buf + off, &vt,    4); off += 4;
    memcpy(buf + off, &sl64,  8); off += 8;
    memcpy(buf + off, val,    sl); off += sl;

    OcGgufFile f;
    OcError e = oc_gguf_parse(buf, cap, &f);
    cr_assert_eq(e, OC_OK, "parse: %s", oc_error_msg(e));
    cr_assert_eq(oc_gguf_arch_from_file(&f), OC_ARCH_LLAMA, "arch should be Llama");

    oc_gguf_free(&f);
    free(buf);
}

Test(gguf, arch_from_file_detects_namespace_fallback)
{
    /* If `general.architecture` is absent, scan metadata keys for an
     * `<arch>.*` namespace. Build a GGUF with `qwen2.block_count = 32`. */
    size_t cap = 256;
    uint8_t *buf = calloc(cap, 1);
    cr_assert_not_null(buf);
    size_t off = 0;
    uint32_t magic = OC_GGUF_MAGIC, ver = 3, vt = OC_GGUF_MT_UINT32;
    uint64_t tc = 0, kvc = 1;
    const char *key = "qwen2.block_count";
    size_t kl = strlen(key);
    uint64_t kl64 = kl;
    uint32_t val = 32;

    memcpy(buf + off, &magic, 4); off += 4;
    memcpy(buf + off, &ver,   4); off += 4;
    memcpy(buf + off, &tc,    8); off += 8;
    memcpy(buf + off, &kvc,   8); off += 8;
    memcpy(buf + off, &kl64,  8); off += 8;
    memcpy(buf + off, key,   kl); off += kl;
    memcpy(buf + off, &vt,    4); off += 4;
    memcpy(buf + off, &val,   4); off += 4;

    OcGgufFile f;
    OcError e = oc_gguf_parse(buf, cap, &f);
    cr_assert_eq(e, OC_OK, "parse: %s", oc_error_msg(e));
    cr_assert_eq(oc_gguf_arch_from_file(&f), OC_ARCH_QWEN, "arch should be Qwen");

    oc_gguf_free(&f);
    free(buf);
}

Test(gguf, arch_from_file_returns_unknown_when_no_metadata)
{
    /* A GGUF with no architecture-relevant metadata returns OC_ARCH_UNKNOWN. */
    size_t cap = 256;
    uint8_t *buf = calloc(cap, 1);
    cr_assert_not_null(buf);
    size_t off = 0;
    uint32_t magic = OC_GGUF_MAGIC, ver = 3, vt = OC_GGUF_MT_STRING;
    uint64_t tc = 0, kvc = 1;
    const char *key = "general.name";
    size_t kl = strlen(key);
    uint64_t kl64 = kl;
    const char *val = "test";
    size_t sl = strlen(val);
    uint64_t sl64 = sl;

    memcpy(buf + off, &magic, 4); off += 4;
    memcpy(buf + off, &ver,   4); off += 4;
    memcpy(buf + off, &tc,    8); off += 8;
    memcpy(buf + off, &kvc,   8); off += 8;
    memcpy(buf + off, &kl64,  8); off += 8;
    memcpy(buf + off, key,   kl); off += kl;
    memcpy(buf + off, &vt,    4); off += 4;
    memcpy(buf + off, &sl64,  8); off += 8;
    memcpy(buf + off, val,    sl); off += sl;

    OcGgufFile f;
    OcError e = oc_gguf_parse(buf, cap, &f);
    cr_assert_eq(e, OC_OK, "parse: %s", oc_error_msg(e));
    cr_assert_eq(oc_gguf_arch_from_file(&f), OC_ARCH_UNKNOWN,
        "arch should be UNKNOWN for non-arch metadata");

    oc_gguf_free(&f);
    free(buf);
}
