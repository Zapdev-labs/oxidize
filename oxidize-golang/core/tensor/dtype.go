// Package tensor mirrors oxidize_core::compute::tensor. It provides DType,
package tensor

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
)

// DType identifies the element type of a tensor.
type DType uint8

const (
	DTypeF32 DType = iota
	DTypeF16
	DTypeI8
	DTypeI16
	DTypeI32
	DTypeI64
)

func (d DType) String() string {
	switch d {
	case DTypeF32:
		return "f32"
	case DTypeF16:
		return "f16"
	case DTypeI8:
		return "i8"
	case DTypeI16:
		return "i16"
	case DTypeI32:
		return "i32"
	case DTypeI64:
		return "i64"
	default:
		return fmt.Sprintf("dtype(%d)", uint8(d))
	}
}

// SizeInBytes returns the byte width of a single element of the dtype.
func (d DType) SizeInBytes() int {
	switch d {
	case DTypeF32, DTypeI32:
		return 4
	case DTypeF16, DTypeI16:
		return 2
	case DTypeI8:
		return 1
	case DTypeI64:
		return 8
	default:
		return 0
	}
}

// ActivationFn identifies a hidden activation function for linear layers.
type ActivationFn uint8

const (
	ActivationNone ActivationFn = iota
	ActivationSilu
	ActivationGelu
	ActivationRelu
)

func (a ActivationFn) String() string {
	switch a {
	case ActivationSilu:
		return "silu"
	case ActivationGelu:
		return "gelu"
	case ActivationRelu:
		return "relu"
	default:
		return "none"
	}
}

// Kernel-related constants.
const (
	FlashAttentionBlockTokens  = 64
	ParallelGemvMinOps         = 4096
	TransposedGemvColChunk     = 64
)

// Common block sizes re-exported for convenience.
const (
	QK8_0     = 32
	QK4_0     = 32
	QK4_1     = 32
	QK5_0     = 32
	QK5_1     = 32
	QK_K      = 256
	QK_NVFP4  = 64
	QK_NVFP4_SUB = 16
)

// Tensor is a row-major, dense, n-dimensional array. The implementation
// mirrors the bottom-up Tensor struct in oxidize-core/src/compute/tensor.rs
// (constants, errors, kernels, struct) using a single struct with helpers.
type Tensor struct {
	mu       sync.RWMutex
	data     []float32
	bytes    []byte
	shape    []int
	strides  []int
	dtype    DType
	owned    bool
}

// New constructs a Tensor from a float32 slice and shape. The data is copied.
func New(data []float32, shape []int) *Tensor {
	t := &Tensor{shape: append([]int(nil), shape...), dtype: DTypeF32, owned: true}
	total := 1
	for _, s := range shape {
		if s <= 0 {
			return nil
		}
		total *= s
	}
	if total != len(data) {
		return nil
	}
	t.data = append([]float32(nil), data...)
	t.strides = defaultStrides(shape)
	return t
}

// FromBytes constructs a Tensor from a raw byte buffer with the given shape
// and dtype. The bytes are referenced (not copied) when copyBytes is false.
func FromBytes(data []byte, shape []int, dt DType) (*Tensor, error) {
	total := 1
	for _, s := range shape {
		if s <= 0 {
			return nil, fmt.Errorf("invalid shape dimension: %d", s)
		}
		total *= s
	}
	if total*dt.SizeInBytes() != len(data) {
		return nil, fmt.Errorf("shape/dtype mismatch: shape=%v dtype=%v bytes=%d", shape, dt, len(data))
	}
	t := &Tensor{bytes: data, shape: append([]int(nil), shape...), dtype: dt, owned: false, strides: defaultStrides(shape)}
	if dt == DTypeF32 {
		t.data = bytesAsF32(data)
		t.owned = true
	}
	return t, nil
}

func defaultStrides(shape []int) []int {
	strides := make([]int, len(shape))
	if len(shape) == 0 {
		return strides
	}
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}

// Data returns a copy of the underlying float32 data.
func (t *Tensor) Data() []float32 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	if t.dtype == DTypeF32 {
		return append([]float32(nil), t.data...)
	}
	out := make([]float32, t.numElements())
	for i := range out {
		out[i] = t.f32At(i)
	}
	return out
}

// Bytes returns a copy of the underlying raw bytes.
func (t *Tensor) Bytes() []byte {
	t.mu.RLock()
	defer t.mu.RUnlock()
	if t.dtype == DTypeF32 {
		buf := make([]byte, len(t.data)*4)
		for i, v := range t.data {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
		}
		return buf
	}
	return append([]byte(nil), t.bytes...)
}

// Shape returns the shape of the tensor.
func (t *Tensor) Shape() []int { return append([]int(nil), t.shape...) }

// DType returns the dtype of the tensor.
func (t *Tensor) DType() DType { return t.dtype }

// NumElements returns the total element count.
func (t *Tensor) NumElements() int { return t.numElements() }

func (t *Tensor) numElements() int {
	n := 1
	for _, s := range t.shape {
		n *= s
	}
	return n
}

// At returns the float32 value at the given linear offset. For f16/i8 tensors
// it converts on the fly.
func (t *Tensor) At(i int) float32 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.f32At(i)
}

func (t *Tensor) f32At(i int) float32 {
	switch t.dtype {
	case DTypeF32:
		return t.data[i]
	case DTypeF16:
		return F16BitsToF32(uint16(t.bytes[i*2]) | uint16(t.bytes[i*2+1])<<8)
	case DTypeI8:
		return float32(int8(t.bytes[i]))
	case DTypeI16:
		v := int16(binary.LittleEndian.Uint16(t.bytes[i*2:]))
		return float32(v)
	case DTypeI32:
		v := int32(binary.LittleEndian.Uint32(t.bytes[i*4:]))
		return float32(v)
	case DTypeI64:
		v := int64(binary.LittleEndian.Uint64(t.bytes[i*8:]))
		return float32(v)
	default:
		return 0
	}
}

// Set writes a float32 value at the given linear offset. Conversion to the
// underlying dtype is performed as needed.
func (t *Tensor) Set(i int, v float32) {
	t.mu.Lock()
	defer t.mu.Unlock()
	switch t.dtype {
	case DTypeF32:
		t.data[i] = v
	case DTypeF16:
		bits := F32ToF16Bits(v)
		t.bytes[i*2] = byte(bits)
		t.bytes[i*2+1] = byte(bits >> 8)
	case DTypeI8:
		t.bytes[i] = byte(int8(clampInt(v, -128, 127)))
	case DTypeI16:
		binary.LittleEndian.PutUint16(t.bytes[i*2:], uint16(int16(clampInt(v, -32768, 32767))))
	case DTypeI32:
		binary.LittleEndian.PutUint32(t.bytes[i*4:], uint32(int32(v)))
	case DTypeI64:
		binary.LittleEndian.PutUint64(t.bytes[i*8:], uint64(int64(v)))
	}
}

func bytesAsF32(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

// F16LEToF32 converts a little-endian f16 byte pair to f32.
func F16LEToF32(b [2]byte) float32 {
	bits := uint16(b[0]) | uint16(b[1])<<8
	return F16BitsToF32(bits)
}

// F16BitsToF32 converts a 16-bit IEEE-754 half-precision value to float32.
func F16BitsToF32(bits uint16) float32 {
	sign := uint32(bits>>15) & 0x1
	exp := uint32(bits>>10) & 0x1F
	mant := uint32(bits) & 0x3FF
	var out uint32
	switch {
	case exp == 0:
		if mant == 0 {
			out = sign << 31
		} else {
			// subnormal
			for mant&0x400 == 0 {
				mant <<= 1
				exp--
			}
			exp++
			mant &= 0x3FF
			out = (sign << 31) | ((exp + 112) << 23) | (mant << 13)
		}
	case exp == 0x1F:
		out = (sign << 31) | (0xFF << 23) | (mant << 13)
	default:
		out = (sign << 31) | ((exp + 112) << 23) | (mant << 13)
	}
	return math.Float32frombits(out)
}

// F32ToF16Bits converts a float32 to a 16-bit IEEE-754 half-precision value.
func F32ToF16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 31) & 0x1)
	exp := int32((bits>>23)&0xFF) - 127
	mant := bits & 0x7FFFFF
	var out uint16
	switch {
	case exp == 128:
		// inf/nan
		if mant == 0 {
			out = (sign << 15) | 0x7C00
		} else {
			out = (sign << 15) | 0x7C00 | uint16(mant>>13)
		}
	case exp > 15:
		out = (sign << 15) | 0x7C00
	case exp < -24:
		out = sign << 15
	default:
		if exp < -13 {
			mant |= 0x800000
			shift := uint(-10 - exp)
			mant = (mant + (1 << (shift - 1))) >> shift
			out = (sign << 15) | uint16(mant)
		} else {
			out = (sign << 15) | uint16(exp+15)<<10 | uint16(mant>>13)
		}
	}
	return out
}

// ExtractBits pulls a `bits`-wide field out of `bitstream` starting at the
// given bit index.
func ExtractBits(bitstream []byte, index int, bits int) uint32 {
	if bits == 0 || bits > 32 {
		return 0
	}
	bitPos := uint(index * bits)
	bytePos := bitPos / 8
	bitOffset := bitPos % 8
	var value uint32
	remaining := bits
	shift := 0
	for remaining > 0 {
		available := 8 - int(bitOffset)
		take := available
		if remaining < take {
			take = remaining
		}
		mask := uint32((1 << take) - 1)
		chunk := uint32(bitstream[bytePos]) >> bitOffset
		value |= (chunk & mask) << shift
		remaining -= take
		shift += take
		bitOffset = 0
		bytePos++
	}
	return value
}

func clampInt(v float32, lo, hi int64) int64 {
	iv := int64(v)
	if iv < lo {
		return lo
	}
	if iv > hi {
		return hi
	}
	return iv
}
