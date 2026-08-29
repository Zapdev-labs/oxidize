/*
 * gguf_emitter.h — shared GGUF byte-emitter macros for tests that build
 * minimal GGUF buffers by hand (tokenizer + gguf parser suites).
 *
 * EMIT appends raw bytes and advances the offset; the typed variants
 * serialize host-endian scalars (tests run on both endiannesses in CI, so
 * only use these when the GGUF field happens to match host order — the
 * suites that need LE use the explicit put_* helpers).
 */
#ifndef OXIDIZE_C_TESTS_GGUF_EMITTER_H
#define OXIDIZE_C_TESTS_GGUF_EMITTER_H

#include <stdint.h>
#include <string.h>

#define EMIT(buf, off, src, n) \
    do { memcpy((buf) + (off), (src), (n)); (off) += (n); } while (0)

#define EMIT_U8(buf, off, v) \
    do { uint8_t _x = (uint8_t)(v); EMIT(buf, off, &_x, 1); } while (0)

#define EMIT_U32(buf, off, v) \
    do { uint32_t _x = (uint32_t)(v); EMIT(buf, off, &_x, 4); } while (0)

#define EMIT_U64(buf, off, v) \
    do { uint64_t _x = (uint64_t)(v); EMIT(buf, off, &_x, 8); } while (0)

#define EMIT_F32(buf, off, v) \
    do { float _x = (float)(v); EMIT(buf, off, &_x, 4); } while (0)

/* Append a GGUF string-KV key (u64 length + bytes). */
#define EMIT_KV_STR_KEY(buf, off, key_str) \
    do { \
        const char *k = (key_str); \
        uint64_t kl = strlen(k); \
        EMIT_U64(buf, off, kl); \
        EMIT(buf, off, k, kl); \
    } while (0)

#endif /* OXIDIZE_C_TESTS_GGUF_EMITTER_H */
