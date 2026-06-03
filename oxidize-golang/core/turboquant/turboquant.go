// Package turboquant mirrors oxidize_core::compute::turboquant. It implements
// blockwise 4-bit and 8-bit quantization that pairs with the kv_cache module.
package turboquant

import "math"

// Type mirrors TurboQuantType.
type Type uint8

const (
	Int4 Type = iota
	Int8
)

// Block mirrors TurboQuantBlock.
type Block struct {
	Scale  float32
	Values []uint8
}

// Data mirrors TurboQuantData.
type Data struct {
	QType Type
	Cols  int
	Rows  int
	Blocks []Block
}

// QuantizeF32 quantizes a 2D float32 source (row-major, rows x cols) using
// TurboQuant blockwise quantization. The block size is fixed to 32.
func (d *Data) QuantizeF32(src []float32, rows, cols int, qtype Type) {
	d.QType = qtype
	d.Rows = rows
	d.Cols = cols
	if rows*cols == 0 {
		d.Blocks = nil
		return
	}
	const block = 32
	stride := block
	if cols < stride {
		stride = cols
	}
	numBlocks := (cols + stride - 1) / stride
	d.Blocks = make([]Block, 0, rows*numBlocks)
	for r := 0; r < rows; r++ {
		for b := 0; b < numBlocks; b++ {
			start := b * stride
			end := start + stride
			if end > cols {
				end = cols
			}
			values := src[r*cols+start : r*cols+end]
			blk := quantizeBlock(values, qtype)
			d.Blocks = append(d.Blocks, blk)
		}
	}
}

// DequantizeF32 writes the dequantized data into `out` (rows x cols).
func (d *Data) DequantizeF32(out []float32) {
	if len(out) < d.Rows*d.Cols {
		return
	}
	const block = 32
	stride := block
	if d.Cols < stride {
		stride = d.Cols
	}
	numBlocks := (d.Cols + stride - 1) / stride
	idx := 0
	for r := 0; r < d.Rows; r++ {
		for b := 0; b < numBlocks; b++ {
			if idx >= len(d.Blocks) {
				return
			}
			start := b * stride
			end := start + stride
			if end > d.Cols {
				end = d.Cols
			}
			blk := d.Blocks[idx]
			idx++
			dequantizeBlock(blk, out[r*d.Cols+start:r*d.Cols+end], d.QType)
		}
	}
}

// GEMV computes output = dequant(data) * vector.
func (d *Data) GEMV(vector, output []float32) {
	if len(vector) < d.Cols {
		return
	}
	if len(output) < d.Rows {
		return
	}
	row := make([]float32, d.Rows*d.Cols)
	d.DequantizeF32(row)
	for r := 0; r < d.Rows; r++ {
		var sum float32
		for c := 0; c < d.Cols; c++ {
			sum += row[r*d.Cols+c] * vector[c]
		}
		output[r] = sum
	}
}

func quantizeBlock(values []float32, qtype Type) Block {
	if len(values) == 0 {
		return Block{}
	}
	var max float32
	for _, v := range values {
		if a := float32(math.Abs(float64(v))); a > max {
			max = a
		}
	}
	var scale float32
	var levels int
	switch qtype {
	case Int4:
		levels = 7
		scale = max / float32(levels)
	case Int8:
		levels = 127
		scale = max / float32(levels)
	}
	if scale == 0 {
		scale = 1
	}
	inv := 1 / scale
	out := make([]uint8, len(values))
	for i, v := range values {
		switch qtype {
		case Int4:
			q := int8(clampInt(v*inv, -8, 7))
			out[i] = nibblePack(q)
		case Int8:
			q := int8(clampInt(v*inv, -128, 127))
			out[i] = uint8(q)
		}
	}
	return Block{Scale: scale, Values: out}
}

func dequantizeBlock(blk Block, dst []float32, qtype Type) {
	if len(dst) == 0 {
		return
	}
	for i, v := range blk.Values {
		switch qtype {
		case Int4:
			q := nibbleUnpack(v)
			dst[i] = float32(int8(q)) * blk.Scale
		case Int8:
			dst[i] = float32(int8(v)) * blk.Scale
		}
	}
	if len(dst) > len(blk.Values) {
		for i := len(blk.Values); i < len(dst); i++ {
			dst[i] = 0
		}
	}
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

func nibblePack(v int8) uint8 {
	// Two's-complement 4-bit
	u := uint8(int8(v) & 0x0F)
	return u
}

func nibbleUnpack(v uint8) int8 {
	// High 4 bits are the actual value; the low 4 bits are metadata in some
	// variants. For our simple format, the byte IS the value.
	if v >= 8 {
		return int8(v) - 16
	}
	return int8(v)
}
