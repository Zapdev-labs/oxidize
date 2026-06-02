package gguf

import (
	"encoding/binary"
	"math"
)

type reader struct {
	bytes  []byte
	cursor int
}

func newReader(raw []byte) *reader {
	return &reader{bytes: raw}
}

func (r *reader) position() int {
	return r.cursor
}

func (r *reader) readExact(length int) ([]byte, error) {
	end := r.cursor + length
	if end < r.cursor {
		return nil, errIntegerOverflow()
	}
	if end > len(r.bytes) {
		return nil, errUnexpectedEOF()
	}
	out := r.bytes[r.cursor:end]
	r.cursor = end
	return out, nil
}

func (r *reader) readU8() (uint8, error) {
	raw, err := r.readExact(1)
	if err != nil {
		return 0, err
	}
	return raw[0], nil
}

func (r *reader) readU16() (uint16, error) { return r.readUint16() }
func (r *reader) readU32() (uint32, error) { return r.readUint32() }
func (r *reader) readU64() (uint64, error) { return r.readUint64() }

func (r *reader) readUint16() (uint16, error) {
	raw, err := r.readExact(2)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint16(raw), nil
}

func (r *reader) readUint32() (uint32, error) {
	raw, err := r.readExact(4)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(raw), nil
}

func (r *reader) readUint64() (uint64, error) {
	raw, err := r.readExact(8)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(raw), nil
}

func (r *reader) readI8() (int8, error) {
	value, err := r.readU8()
	return int8(value), err
}

func (r *reader) readI16() (int16, error) {
	value, err := r.readU16()
	return int16(value), err
}

func (r *reader) readI32() (int32, error) {
	value, err := r.readU32()
	return int32(value), err
}

func (r *reader) readI64() (int64, error) {
	value, err := r.readU64()
	return int64(value), err
}

func (r *reader) readF32() (float32, error) {
	value, err := r.readU32()
	if err != nil {
		return 0, err
	}
	return math.Float32frombits(value), nil
}

func (r *reader) readF64() (float64, error) {
	value, err := r.readU64()
	if err != nil {
		return 0, err
	}
	return math.Float64frombits(value), nil
}

func (r *reader) readString() (string, error) {
	length, err := r.readU64()
	if err != nil {
		return "", err
	}
	if length > uint64(len(r.bytes)) {
		return "", errIntegerOverflow()
	}
	raw, err := r.readExact(int(length))
	if err != nil {
		return "", err
	}
	return string(raw), nil
}
