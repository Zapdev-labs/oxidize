import struct


def f16_bits_to_f32(bits: int) -> float:
    sign = (bits >> 15) & 1
    exp = (bits >> 10) & 0x1F
    mant = bits & 0x3FF
    if exp == 0:
        if mant == 0:
            out = sign << 31
        else:
            while mant & 0x400 == 0:
                mant <<= 1
                exp -= 1
            exp += 1
            mant &= 0x3FF
            out = (sign << 31) | ((exp + 112) << 23) | (mant << 13)
    elif exp == 0x1F:
        out = (sign << 31) | (0xFF << 23) | (mant << 13)
    else:
        out = (sign << 31) | ((exp + 112) << 23) | (mant << 13)
    return struct.unpack("<f", struct.pack("<I", out))[0]
