// Package oxk mirrors oxidize-kernels: custom CPU kernels for quantized GEMV.
//
// Phase 1 scope: Q4_K × Q8_K row dots (scalar reference) and a contiguous-range
// GEMV helper. The per-row math is bit-identical to the legacy kernels in
// oxidize-core/src/compute/tensor.rs and the Rust oxidize-kernels crate.
//
// This package is self-contained (no deps on other oxidize packages) so it can
// be benchmarked and tested in isolation.
package oxk

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"sync"
)

// ---------------------------------------------------------------------------
// Constants (match GGUF K-quants)
// ---------------------------------------------------------------------------

// QK_K is values per super-block.
const QK_K = 256

// BLOCK_Q4_K_SIZE is bytes per Q4_K block: f16 d + f16 dmin + 12 scale bytes + 128 nibbles.
const BLOCK_Q4_K_SIZE = 144

// BLOCK_Q8_K_BYTES is bytes per Q8_K block: f32 d + 256 int8 + 16 i16 bsums.
const BLOCK_Q8_K_BYTES = 4 + 256 + 32

// ---------------------------------------------------------------------------
// CPU vendor / ISA detection and tuning
// ---------------------------------------------------------------------------

// CpuVendor identifies the CPU manufacturer.
type CpuVendor int

const (
	VendorIntel CpuVendor = iota
	VendorAmd
	VendorOther
)

// CpuInfo is a snapshot of the CPU we are running on.
type CpuInfo struct {
	Vendor     CpuVendor
	Family     uint32
	Model      uint32
	Stepping   uint32
	HasAvx2    bool
	HasFma     bool
	HasAvx512f bool
	HasAvx512bw bool
	HasAvx512vnni bool
	HasAvxvnni    bool
	// UseAvx512 is the kernel-selected default: use AVX-512F/BW path when available.
	UseAvx512 bool
}

// OxkTune is memory-pipeline tuning consumed by the kernels.
type OxkTune struct {
	// PfBytes is prefetch distance in bytes ahead of the current weight block
	// pointer (multiple of BLOCK_Q4_K_SIZE; 0 disables software prefetch).
	PfBytes int
	// PfNta is prefetch with NTA instead of T0.
	PfNta bool
}

var (
	cpuInfoOnce sync.Once
	cpuInfoVal  CpuInfo
	tuneOnce    sync.Once
	tuneVal     OxkTune
)

// GetCpuInfo returns the detected CPU info (resolved once per process).
func GetCpuInfo() CpuInfo {
	cpuInfoOnce.Do(func() {
		cpuInfoVal = detectCpuinfo()
	})
	return cpuInfoVal
}

// Tune returns the tuning profile for this process (resolved once).
func Tune() OxkTune {
	tuneOnce.Do(func() {
		info := GetCpuInfo()
		defaultBlocks := 2
		switch info.Vendor {
		case VendorIntel:
			defaultBlocks = 1
		case VendorAmd:
			defaultBlocks = 2
		}
		blocks := defaultBlocks
		if v := os.Getenv("OXIDIZE_OXK_PF"); v != "" {
			if b, err := strconv.Atoi(v); err == nil {
				blocks = b
			}
		}
		pfNta := false
		switch os.Getenv("OXIDIZE_OXK_PF_HINT") {
		case "nta":
			pfNta = true
		case "t0", "":
			pfNta = false
		default:
			fmt.Fprintf(os.Stderr, "OXIDIZE_OXK_PF_HINT=%s unknown (use t0|nta); using t0\n", os.Getenv("OXIDIZE_OXK_PF_HINT"))
		}
		tuneVal = OxkTune{
			PfBytes: blocks * BLOCK_Q4_K_SIZE,
			PfNta:   pfNta,
		}
	})
	return tuneVal
}

// OxkCpuSummary returns a one-line human-readable summary of detected CPU + tuning.
func OxkCpuSummary() string {
	info := GetCpuInfo()
	vendor := "other"
	switch info.Vendor {
	case VendorIntel:
		vendor = "intel"
	case VendorAmd:
		vendor = "amd"
	}
	t := Tune()
	pfHint := "t0"
	if t.PfNta {
		pfHint = "nta"
	}
	return fmt.Sprintf(
		"vendor=%s fam=%d model=%d step=%d avx2=%v fma=%v avx512f=%v avx512bw=%v avx512vnni=%v avxvnni=%v use_avx512=%v pf_blocks=%d pf_hint=%s",
		vendor, info.Family, info.Model, info.Stepping,
		info.HasAvx2, info.HasFma, info.HasAvx512f, info.HasAvx512bw,
		info.HasAvx512vnni, info.HasAvxvnni, info.UseAvx512,
		t.PfBytes/BLOCK_Q4_K_SIZE, pfHint,
	)
}

func detectCpuinfo() CpuInfo {
	// Go has no direct cpuid access without asm or cgo. We conservatively
	// return "other" with all flags false. Build tags or cgo can override.
	return CpuInfo{
		Vendor:   VendorOther,
		Family:   0,
		Model:    0,
		Stepping: 0,
	}
}

// ---------------------------------------------------------------------------
// ISA availability stubs (always false in pure Go; build tags can override)
// ---------------------------------------------------------------------------

// OxkAvx2Available returns whether AVX2 kernels can run.
func OxkAvx2Available() bool { return false }

// OxkAvx512Available returns whether AVX-512F+BW kernels can run.
func OxkAvx512Available() bool { return false }

// OxkAvx512vnniAvailable returns whether AVX-512 VNNI kernels can run.
func OxkAvx512vnniAvailable() bool { return false }

// OxkAvxvnniAvailable returns whether AVX-VNNI (256-bit) kernels can run.
func OxkAvxvnniAvailable() bool { return false }

// ---------------------------------------------------------------------------
// Tile width selection (default 16 = widest, enabled by default)
// ---------------------------------------------------------------------------

var (
	maxTileOnce sync.Once
	maxTileVal  int
)

// MaxTile returns the lead multi-row tile width for range GEMV.
// Default is 16 (widest) on every vendor. OXIDIZE_OXK_TILE={1,4,8,16} overrides.
func MaxTile() int {
	maxTileOnce.Do(func() {
		maxTileVal = 16
		if v := os.Getenv("OXIDIZE_OXK_TILE"); v != "" {
			if t, err := strconv.Atoi(v); err == nil {
				switch t {
				case 1, 4, 8, 16:
					maxTileVal = t
				}
			}
		}
	})
	return maxTileVal
}

// ---------------------------------------------------------------------------
// f16 helpers
// ---------------------------------------------------------------------------

// f16LeToF32 converts little-endian f16 bytes to f32, no half dependency.
func f16LeToF32(bytes [2]byte) float32 {
	bits := uint16(bytes[0]) | uint16(bytes[1])<<8
	sign := uint32((bits >> 15) & 1)
	exp := uint32((bits >> 10) & 0x1f)
	frac := uint32(bits & 0x03ff)
	var f32Bits uint32
	if exp == 0 {
		if frac == 0 {
			f32Bits = sign << 31
		} else {
			fracNorm := frac
			e := -14
			for (fracNorm & 0x0400) == 0 {
				fracNorm <<= 1
				e--
			}
			fracNorm &= 0x03ff
			f32Bits = (sign << 31) | (uint32(e+127) << 23) | (fracNorm << 13)
		}
	} else if exp == 0x1f {
		f32Bits = (sign << 31) | (0xff << 23) | (frac << 13)
	} else {
		f32Bits = (sign << 31) | ((exp + 112) << 23) | (frac << 13)
	}
	return math.Float32frombits(f32Bits)
}

// ---------------------------------------------------------------------------
// Scale/min decoding
// ---------------------------------------------------------------------------

// GetScaleMinK4 decodes the (scale, min) pair for sub-group j from a Q4_K
// 12-byte scale field (identical to llama.cpp's get_scale_min_k4).
func GetScaleMinK4(j int, scales []byte) (uint8, uint8) {
	if j < 4 {
		return scales[j] & 63, scales[j+4] & 63
	}
	return (scales[j+4] & 0x0f) | ((scales[j-4] >> 6) << 4),
		(scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
}

// readQ8kBsum reads a bsum from Q8_K block bsums area.
func readQ8kBsum(bsums []byte, index int) int16 {
	return int16(bsums[index*2]) | int16(bsums[index*2+1])<<8
}

// ---------------------------------------------------------------------------
// Q8_K activation quantization
// ---------------------------------------------------------------------------

// QuantizeQ8KInto quantizes vector (length nBlocks*256) into nBlocks Q8_K blocks.
func QuantizeQ8KInto(vector []float32, nBlocks int, out []byte) {
	if len(vector) != nBlocks*QK_K {
		panic("vector length mismatch")
	}
	if len(out) < nBlocks*BLOCK_Q8_K_BYTES {
		panic("output buffer too small")
	}
	for b, blockIn := range chunksExact(vector, QK_K) {
		blockOut := out[b*BLOCK_Q8_K_BYTES : (b+1)*BLOCK_Q8_K_BYTES]
		quantizeQ8KBlock(blockIn, blockOut)
	}
}

func quantizeQ8KBlock(blockIn []float32, blockOut []byte) {
	amax := 0.0
	max := 0.0
	for _, v := range blockIn {
		av := math.Abs(float64(v))
		if av > amax {
			amax = av
			max = float64(v)
		}
	}
	if amax == 0.0 {
		copy(blockOut[:4], []byte{0, 0, 0, 0})
		for i := range blockOut[4:] {
			blockOut[4+i] = 0
		}
		return
	}
	iscale := -128.0 / max
	d := 1.0 / iscale
	bits := math.Float32bits(float32(d))
	blockOut[0] = byte(bits)
	blockOut[1] = byte(bits >> 8)
	blockOut[2] = byte(bits >> 16)
	blockOut[3] = byte(bits >> 24)
	qsOff := 4
	for i, v := range blockIn {
		q := int(iscale * float64(v))
		if q < -128 {
			q = -128
		}
		if q > 127 {
			q = 127
		}
		blockOut[qsOff+i] = byte(int8(q))
	}
	bsumsOff := qsOff + QK_K
	for g := 0; g < QK_K/16; g++ {
		sum := 0
		for i := 0; i < 16; i++ {
			sum += int(int8(blockOut[qsOff+g*16+i]))
		}
		if sum < math.MinInt16 {
			sum = math.MinInt16
		}
		if sum > math.MaxInt16 {
			sum = math.MaxInt16
		}
		sum16 := int16(sum)
		blockOut[bsumsOff+g*2] = byte(sum16)
		blockOut[bsumsOff+g*2+1] = byte(sum16 >> 8)
	}
}

// ---------------------------------------------------------------------------
// Q4_K × Q8_K scalar row dot
// ---------------------------------------------------------------------------

// Q4kQ8kRowDotScalar dots one Q4_K row (blocksPerRow blocks) against a Q8_K vector.
func Q4kQ8kRowDotScalar(row []byte, blocksPerRow int, q8k []byte) float32 {
	if len(row) < blocksPerRow*BLOCK_Q4_K_SIZE {
		panic("row too small")
	}
	if len(q8k) < blocksPerRow*BLOCK_Q8_K_BYTES {
		panic("q8k too small")
	}
	var acc float32
	for blockIdx := 0; blockIdx < blocksPerRow; blockIdx++ {
		w := row[blockIdx*BLOCK_Q4_K_SIZE : (blockIdx+1)*BLOCK_Q4_K_SIZE]
		q8b := q8k[blockIdx*BLOCK_Q8_K_BYTES : (blockIdx+1)*BLOCK_Q8_K_BYTES]
		dW := f16LeToF32([2]byte{w[0], w[1]})
		dminW := f16LeToF32([2]byte{w[2], w[3]})
		dQ8 := math.Float32frombits(uint32(q8b[0]) | uint32(q8b[1])<<8 | uint32(q8b[2])<<16 | uint32(q8b[3])<<24)
		scales := w[4:16]
		qs := w[16 : 16+QK_K/2]
		q8 := q8b[4 : 4+QK_K]
		bsums := q8b[4+QK_K:]

		var pos int32
		var minAcc int32
		for gp := 0; gp < 4; gp++ {
			g1 := gp * 2
			g2 := g1 + 1
			s1, ms1 := GetScaleMinK4(g1, scales)
			s2, ms2 := GetScaleMinK4(g2, scales)
			var sum1, sum2 int32
			for i := 0; i < 32; i++ {
				byteVal := qs[gp*32+i]
				sum1 += int32(byteVal&0x0f) * int32(int8(q8[g1*32+i]))
				sum2 += int32(byteVal>>4) * int32(int8(q8[g2*32+i]))
			}
			pos += int32(s1) * sum1 + int32(s2) * sum2
			bs1 := int32(readQ8kBsum(bsums, g1*2)) + int32(readQ8kBsum(bsums, g1*2+1))
			bs2 := int32(readQ8kBsum(bsums, g2*2)) + int32(readQ8kBsum(bsums, g2*2+1))
			minAcc += int32(ms1) * bs1
			minAcc += int32(ms2) * bs2
		}
		acc += dW*dQ8*float32(pos) - dminW*dQ8*float32(minAcc)
	}
	return acc
}

// ---------------------------------------------------------------------------
// Tile runners (multi-row variants) — enabled by default, scalar fallback
// ---------------------------------------------------------------------------

// Q4kQ8kRowDotX1Scalar is the single-row scalar dot (alias for the scalar function).
func Q4kQ8kRowDotX1Scalar(row []byte, blocksPerRow int, q8k []byte) float32 {
	return Q4kQ8kRowDotScalar(row, blocksPerRow, q8k)
}

// Q4kQ8kRowDotX4Scalar dots 4 consecutive rows against one q8k vector.
func Q4kQ8kRowDotX4Scalar(rows []byte, rowBytes int, blocksPerRow int, q8k []byte, out []float32) {
	if len(out) < 4 {
		panic("out too small")
	}
	for r := 0; r < 4; r++ {
		row := rows[r*rowBytes : (r+1)*rowBytes]
		out[r] = Q4kQ8kRowDotScalar(row, blocksPerRow, q8k)
	}
}

// Q4kQ8kRowDotX8Scalar dots 8 consecutive rows against one q8k vector.
func Q4kQ8kRowDotX8Scalar(rows []byte, rowBytes int, blocksPerRow int, q8k []byte, out []float32) {
	if len(out) < 8 {
		panic("out too small")
	}
	for r := 0; r < 8; r++ {
		row := rows[r*rowBytes : (r+1)*rowBytes]
		out[r] = Q4kQ8kRowDotScalar(row, blocksPerRow, q8k)
	}
}

// Q4kQ8kRowDotX16Scalar dots 16 consecutive rows against one q8k vector.
func Q4kQ8kRowDotX16Scalar(rows []byte, rowBytes int, blocksPerRow int, q8k []byte, out []float32) {
	if len(out) < 16 {
		panic("out too small")
	}
	for r := 0; r < 16; r++ {
		row := rows[r*rowBytes : (r+1)*rowBytes]
		out[r] = Q4kQ8kRowDotScalar(row, blocksPerRow, q8k)
	}
}

// ---------------------------------------------------------------------------
// Range GEMV — main entry point with tile runner enabled by default
// ---------------------------------------------------------------------------

// GemvQ4kRange dots a contiguous range of Q4_K rows against one pre-quantized Q8_K vector.
//
// rows must point at out.len() rows of blocksPerRow Q4_K blocks laid out back-to-back
// (rowBytes = blocksPerRow * BLOCK_Q4_K_SIZE apart); q8k holds blocksPerRow Q8_K blocks.
// Uses the widest available tile width (default 16) with scalar fallback.
func GemvQ4kRange(rows []byte, blocksPerRow int, q8k []byte, out []float32) {
	rowBytes := blocksPerRow * BLOCK_Q4_K_SIZE
	if len(rows) < len(out)*rowBytes {
		panic("rows buffer too small")
	}
	if len(q8k) < blocksPerRow*BLOCK_Q8_K_BYTES {
		panic("q8k buffer too small")
	}

	tile := MaxTile()
	n := len(out)
	r := 0

	// x16 tile (widest, enabled by default)
	for tile >= 16 && r+16 <= n {
		base := rows[r*rowBytes:]
		var hex [16]float32
		Q4kQ8kRowDotX16Scalar(base, rowBytes, blocksPerRow, q8k, hex[:])
		copy(out[r:r+16], hex[:])
		r += 16
	}
	// x8 tile
	for tile >= 8 && r+8 <= n {
		base := rows[r*rowBytes:]
		var octet [8]float32
		Q4kQ8kRowDotX8Scalar(base, rowBytes, blocksPerRow, q8k, octet[:])
		copy(out[r:r+8], octet[:])
		r += 8
	}
	// x4 tile
	for tile >= 4 && r+4 <= n {
		base := rows[r*rowBytes:]
		var quad [4]float32
		Q4kQ8kRowDotX4Scalar(base, rowBytes, blocksPerRow, q8k, quad[:])
		copy(out[r:r+4], quad[:])
		r += 4
	}
	// x1 tail
	for r < n {
		row := rows[r*rowBytes : (r+1)*rowBytes]
		out[r] = Q4kQ8kRowDotScalar(row, blocksPerRow, q8k)
		r++
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func chunksExact[T any](s []T, size int) [][]T {
	if len(s)%size != 0 {
		panic("slice length not divisible by chunk size")
	}
	var out [][]T
	for i := 0; i < len(s); i += size {
		out = append(out, s[i:i+size])
	}
	return out
}
