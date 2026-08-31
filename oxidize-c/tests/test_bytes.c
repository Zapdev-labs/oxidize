/* test_bytes.c — byte-level read primitive tests. */
#include <criterion/criterion.h>
#include "oxidize/util/bytes.h"

#include <string.h>

Test(bytes, read_u16_u32_u64_le)
{
    uint8_t buf[] = { 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08 };
    cr_assert_eq(oc_read_u16(buf, sizeof(buf), 0), 0x0201u, "");
    cr_assert_eq(oc_read_u32(buf, sizeof(buf), 0), 0x04030201u, "");
    cr_assert_eq(oc_read_u64(buf, sizeof(buf), 0), 0x0807060504030201ULL, "");
}

Test(bytes, read_signed)
{
    uint8_t buf[] = { 0xff, 0xff, 0xff, 0xff };
    cr_assert_eq(oc_read_i8(buf, 1, 0), -1, "");
    cr_assert_eq(oc_read_i16(buf, 2, 0), -1, "");
    cr_assert_eq(oc_read_i32(buf, 4, 0), -1, "");
}

Test(bytes, read_float)
{
    uint32_t f_bits = 0x4048f5c3u;  /* 3.14f */
    uint8_t buf[4];
    memcpy(buf, &f_bits, 4);
    float f = oc_read_f32(buf, 4, 0);
    cr_assert(f > 3.13f && f < 3.15f, "expected ~3.14, got %f", f);
}

Test(bytes, out_of_bounds_returns_zero)
{
    uint8_t buf[] = { 0x01, 0x02 };
    cr_assert_eq(oc_read_u16(buf, 2, 1), 0, "offset+2 > len");
    cr_assert_eq(oc_read_u32(buf, 2, 0), 0, "len too small");
    cr_assert_eq(oc_read_u8(NULL, 0, 0), 0, "NULL buf");
    cr_assert_eq(oc_read_u8(buf, 2, 5), 0, "offset >= len");
}

Test(bytes, write_and_read_back)
{
    uint8_t buf[8] = {0};
    cr_assert(oc_write_u32(buf, 8, 0, 0xdeadbeefu), "");
    cr_assert_eq(oc_read_u32(buf, 8, 0), 0xdeadbeefu, "");

    cr_assert(oc_write_u64(buf, 8, 0, 0x0123456789abcdefULL), "");
    cr_assert_eq(oc_read_u64(buf, 8, 0), 0x0123456789abcdefULL, "");

    cr_assert(!oc_write_u32(buf, 4, 2, 0), "out-of-bounds write should fail");
}

Test(bytes, volatile_read)
{
    uint8_t buf[] = { 0x42, 0x99 };
    cr_assert_eq(oc_read_volatile_byte(buf, 2, 0), 0x42, "");
    cr_assert_eq(oc_read_volatile_byte(buf, 2, 1), 0x99, "");
    cr_assert_eq(oc_read_volatile_byte(buf, 2, 5), 0, "oob");
    cr_assert_eq(oc_read_volatile_byte(NULL, 0, 0), 0, "NULL");
}
