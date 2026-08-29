/* test_bytes_buf.c — OcBytes byte buffer tests. */
#include "framework.h"
#include "oxidize/bytes.h"

#include <stdint.h>
#include <string.h>

Test(bytes_buf, init)
{
    OcBytes b;
    cr_assert_eq(oc_bytes_init(&b), OC_OK, "");
    cr_assert_null(b.data, "");
    cr_assert_eq(b.size, 0, "");
    cr_assert_eq(b.capacity, 0, "");
    cr_assert_not(b.owned, "");
    oc_bytes_free(&b);
}

Test(bytes_buf, init_null)
{
    cr_assert_eq(oc_bytes_init(NULL), OC_ERR_INVALID_ARG, "");
}

Test(bytes_buf, append_u8)
{
    OcBytes b;
    oc_bytes_init(&b);
    cr_assert_eq(oc_bytes_append_u8(&b, 0x42), OC_OK, "");
    cr_assert_eq(b.size, 1, "");
    cr_assert_eq(b.data[0], 0x42, "");
    oc_bytes_free(&b);
}

Test(bytes_buf, append_u16_le)
{
    OcBytes b;
    oc_bytes_init(&b);
    cr_assert_eq(oc_bytes_append_u16_le(&b, 0x0102), OC_OK, "");
    cr_assert_eq(b.size, 2, "");
    cr_assert_eq(b.data[0], 0x02, "low byte first");
    cr_assert_eq(b.data[1], 0x01, "high byte second");
    oc_bytes_free(&b);
}

Test(bytes_buf, append_u32_le)
{
    OcBytes b;
    oc_bytes_init(&b);
    cr_assert_eq(oc_bytes_append_u32_le(&b, 0x04030201u), OC_OK, "");
    cr_assert_eq(b.size, 4, "");
    cr_assert_eq(b.data[0], 0x01, "");
    cr_assert_eq(b.data[1], 0x02, "");
    cr_assert_eq(b.data[2], 0x03, "");
    cr_assert_eq(b.data[3], 0x04, "");
    oc_bytes_free(&b);
}

Test(bytes_buf, append_u64_le)
{
    OcBytes b;
    oc_bytes_init(&b);
    cr_assert_eq(oc_bytes_append_u64_le(&b, 0x0807060504030201ULL), OC_OK, "");
    cr_assert_eq(b.size, 8, "");
    cr_assert_eq(b.data[0], 0x01, "");
    cr_assert_eq(b.data[7], 0x08, "");
    oc_bytes_free(&b);
}

Test(bytes_buf, append_str)
{
    OcBytes b;
    oc_bytes_init(&b);
    cr_assert_eq(oc_bytes_append_str(&b, "hello"), OC_OK, "");
    /* 5 chars + NUL = 6 bytes */
    cr_assert_eq(b.size, 6, "");
    cr_assert_str_eq((const char *)b.data, "hello", "");
    oc_bytes_free(&b);
}

Test(bytes_buf, read_u8)
{
    OcBytes b;
    oc_bytes_init(&b);
    oc_bytes_append_u8(&b, 0xAB);
    uint8_t v;
    cr_assert_eq(oc_bytes_read_u8(&b, 0, &v), OC_OK, "");
    cr_assert_eq(v, 0xAB, "");
    /* Out of range. */
    cr_assert_neq(oc_bytes_read_u8(&b, 1, &v), OC_OK, "");
    oc_bytes_free(&b);
}

Test(bytes_buf, read_u32_le)
{
    OcBytes b;
    oc_bytes_init(&b);
    oc_bytes_append_u32_le(&b, 0xdeadbeefu);
    uint32_t v;
    cr_assert_eq(oc_bytes_read_u32_le(&b, 0, &v), OC_OK, "");
    cr_assert_eq(v, 0xdeadbeefu, "");
    /* Out of range. */
    cr_assert_neq(oc_bytes_read_u32_le(&b, 1, &v), OC_OK, "");
    oc_bytes_free(&b);
}

Test(bytes_buf, read_u64_le)
{
    OcBytes b;
    oc_bytes_init(&b);
    oc_bytes_append_u64_le(&b, 0x0123456789abcdefULL);
    uint64_t v;
    cr_assert_eq(oc_bytes_read_u64_le(&b, 0, &v), OC_OK, "");
    cr_assert_eq(v, 0x0123456789abcdefULL, "");
    /* Out of range. */
    cr_assert_neq(oc_bytes_read_u64_le(&b, 4, &v), OC_OK, "");
    oc_bytes_free(&b);
}

Test(bytes_buf, from_data_copy)
{
    uint8_t src[] = { 0x10, 0x20, 0x30 };
    OcBytes b;
    cr_assert_eq(oc_bytes_from_data(&b, src, 3, true), OC_OK, "");
    cr_assert(b.owned, "should be owned");
    cr_assert_eq(b.size, 3, "");
    cr_assert_eq(b.data[0], 0x10, "");
    /* Mutating source should not affect the copy. */
    src[0] = 0xFF;
    cr_assert_eq(b.data[0], 0x10, "");
    oc_bytes_free(&b);
}

Test(bytes_buf, from_data_borrow)
{
    uint8_t src[] = { 0x10, 0x20, 0x30 };
    OcBytes b;
    cr_assert_eq(oc_bytes_from_data(&b, src, 3, false), OC_OK, "");
    cr_assert_not(b.owned, "should be borrowed");
    cr_assert_eq(b.size, 3, "");
    cr_assert_eq(b.data[0], 0x10, "");
    /* Borrow shares memory. */
    cr_assert_eq(b.data, src, "");
    /* free should NOT free borrowed memory. */
    oc_bytes_free(&b);
    cr_assert_null(b.data, "");
    /* Source is still valid. */
    cr_assert_eq(src[0], 0x10, "");
}

Test(bytes_buf, reserve)
{
    OcBytes b;
    oc_bytes_init(&b);
    cr_assert_eq(oc_bytes_reserve(&b, 100), OC_OK, "");
    cr_assert(b.capacity >= 100, "");
    cr_assert(b.owned, "after reserve, buffer should be owned");
    oc_bytes_free(&b);
}

Test(bytes_buf, reserve_grows)
{
    OcBytes b;
    oc_bytes_init(&b);
    oc_bytes_append_u8(&b, 0x01);
    size_t cap1 = b.capacity;
    cr_assert_eq(oc_bytes_reserve(&b, cap1 * 10), OC_OK, "");
    cr_assert(b.capacity >= cap1 * 10, "");
    cr_assert_eq(b.data[0], 0x01, "data preserved after reserve");
    oc_bytes_free(&b);
}

Test(bytes_buf, append_grows_geometric)
{
    OcBytes b;
    oc_bytes_init(&b);
    for (int i = 0; i < 1000; i++) {
        cr_assert_eq(oc_bytes_append_u8(&b, (uint8_t)(i & 0xFF)), OC_OK, "");
    }
    cr_assert_eq(b.size, 1000, "");
    cr_assert(b.capacity >= 1000, "");
    for (int i = 0; i < 1000; i++) {
        cr_assert_eq(b.data[i], (uint8_t)(i & 0xFF), "byte %d", i);
    }
    oc_bytes_free(&b);
}

Test(bytes_buf, clear)
{
    OcBytes b;
    oc_bytes_init(&b);
    oc_bytes_append_u32_le(&b, 0x12345678u);
    size_t cap = b.capacity;
    cr_assert_eq(oc_bytes_clear(&b), OC_OK, "");
    cr_assert_eq(b.size, 0, "");
    cr_assert_eq(b.capacity, cap, "capacity preserved");
    /* Can still append after clear. */
    cr_assert_eq(oc_bytes_append_u8(&b, 0x99), OC_OK, "");
    cr_assert_eq(b.size, 1, "");
    oc_bytes_free(&b);
}

Test(bytes_buf, size_and_data_accessors)
{
    OcBytes b;
    oc_bytes_init(&b);
    cr_assert_eq(oc_bytes_size(&b), 0, "");
    cr_assert_null(oc_bytes_data(&b), "");
    oc_bytes_append_u8(&b, 0x42);
    cr_assert_eq(oc_bytes_size(&b), 1, "");
    cr_assert_not_null(oc_bytes_data(&b), "");
    cr_assert_eq(oc_bytes_data(&b)[0], 0x42, "");
    /* NULL safety. */
    cr_assert_eq(oc_bytes_size(NULL), 0, "");
    cr_assert_null(oc_bytes_data(NULL), "");
    oc_bytes_free(&b);
}

Test(bytes_buf, append_bulk)
{
    OcBytes b;
    oc_bytes_init(&b);
    uint8_t data[256];
    for (int i = 0; i < 256; i++) data[i] = (uint8_t)i;
    cr_assert_eq(oc_bytes_append(&b, data, 256), OC_OK, "");
    cr_assert_eq(b.size, 256, "");
    cr_assert_arr_eq(b.data, data, 256, "");
    oc_bytes_free(&b);
}

Test(bytes_buf, append_zero_size_ok)
{
    OcBytes b;
    oc_bytes_init(&b);
    uint8_t dummy = 0;
    cr_assert_eq(oc_bytes_append(&b, &dummy, 0), OC_OK, "");
    cr_assert_eq(b.size, 0, "");
    oc_bytes_free(&b);
}

Test(bytes_buf, append_null_data_error)
{
    OcBytes b;
    oc_bytes_init(&b);
    cr_assert_eq(oc_bytes_append(&b, NULL, 5), OC_ERR_INVALID_ARG, "");
    oc_bytes_free(&b);
}

Test(bytes_buf, from_data_null_args)
{
    OcBytes b;
    cr_assert_eq(oc_bytes_from_data(NULL, NULL, 0, true), OC_ERR_INVALID_ARG, "");
    cr_assert_eq(oc_bytes_from_data(&b, NULL, 5, true), OC_ERR_INVALID_ARG, "");
    oc_bytes_free(&b);
}

Test(bytes_buf, free_safe_on_uninit)
{
    OcBytes b;
    oc_bytes_init(&b);
    oc_bytes_free(&b);  /* should not crash */
    oc_bytes_free(&b);  /* double free safe */
    cr_assert_null(b.data, "");
}
