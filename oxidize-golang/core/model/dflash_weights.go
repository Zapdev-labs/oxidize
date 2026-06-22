package model

import (
	"fmt"

	"github.com/Zapdev-labs/oxidize/golang/core/quantization"
	"github.com/Zapdev-labs/oxidize/golang/core/tensor"
)

// QuantWeight holds raw GGUF-quantized bytes in row-major out_dim × in_dim layout.
type QuantWeight struct {
	Bytes  []byte
	QType  quantization.Type
	OutDim int
	InDim  int
}

// F32Weight is a projection stored as f32 (transposed GEMV layout) or quantized bytes.
type F32Weight struct {
	Data  []float32
	Rows  int
	Cols  int
	Quant *QuantWeight
}

// NewF32WeightFromSlice builds a dense f32 weight (rows = output, cols = input).
func NewF32WeightFromSlice(data []float32, rows, cols int) F32Weight {
	return F32Weight{Data: data, Rows: rows, Cols: cols}
}

// NewF32WeightFromQuantized builds a weight backed by on-the-fly dequant GEMV.
func NewF32WeightFromQuantized(bytes []byte, qtype quantization.Type, outDim, inDim int) F32Weight {
	return F32Weight{
		Rows: inDim,
		Cols: outDim,
		Quant: &QuantWeight{
			Bytes:  bytes,
			QType:  qtype,
			OutDim: outDim,
			InDim:  inDim,
		},
	}
}

// IsLoaded reports whether the weight has f32 or quantized data.
func (w F32Weight) IsLoaded() bool {
	return len(w.Data) > 0 || w.Quant != nil
}

func (w F32Weight) inputDim() int {
	if w.Quant != nil {
		return w.Quant.InDim
	}
	return w.Cols
}

func (w F32Weight) outputDim() int {
	if w.Quant != nil {
		return w.Quant.OutDim
	}
	return w.Rows
}

func (w F32Weight) dequantFn() func([]byte, []float32) error {
	if w.Quant == nil {
		return nil
	}
	qt := w.Quant.QType
	return func(row []byte, out []float32) error {
		return quantization.DequantizeScalar(qt, row, out)
	}
}

// Gemv computes output = W * input.
func (w F32Weight) Gemv(input, output []float32) error {
	if w.Quant != nil {
		q := w.Quant
		// Fast path 1: C++ OXK fused kernels (liboxk).
		ok, err := quantization.GemvOxk(q.Bytes, q.QType, q.OutDim, q.InDim, input, output)
		if err != nil {
			return err
		}
		if ok {
			return nil
		}
		// Fast path 2: Rust AVX2+FMA kernel via CGo (oxidize-ffi).
		ok, err = quantization.GemvRust(q.Bytes, q.QType, q.OutDim, q.InDim, input, output)
		if err != nil {
			return err
		}
		if ok {
			return nil
		}
		// Fallback: pure-Go dequant+dot for any type native kernels don't handle.
		return tensor.GemvQuantizedF32(q.Bytes, w.dequantFn(), q.OutDim, q.InDim, input, output, nil)
	}
	return tensor.GemvF32Transposed(w.Data, w.Cols, w.Rows, input, output)
}

// Gemm computes batched matmul; batch=1 delegates to Gemv.
func (w F32Weight) Gemm(inputs, outputs []float32, batch int) error {
	if batch <= 1 {
		return w.Gemv(inputs, outputs)
	}
	if w.Quant != nil {
		q := w.Quant
		return tensor.GemmQuantizedF32(q.Bytes, w.dequantFn(), q.OutDim, q.InDim, 1, batch, inputs, outputs, nil)
	}
	return tensor.GemmF32(inputs, w.Data, batch, w.Cols, w.Rows, outputs)
}

// Row copies one output row (e.g. embedding vector) into output.
func (w F32Weight) Row(rowIdx int, output []float32) error {
	if w.Quant != nil {
		q := w.Quant
		if rowIdx >= q.OutDim || len(output) != q.InDim {
			return fmt.Errorf("row bounds mismatch: row=%d out=%d want in=%d", rowIdx, q.OutDim, q.InDim)
		}
		rowBytes, err := quantization.QuantizedSize(q.QType, q.InDim)
		if err != nil {
			return err
		}
		start := rowIdx * rowBytes
		end := start + rowBytes
		if end > len(q.Bytes) {
			return fmt.Errorf("quantized row extends past tensor data")
		}
		return quantization.DequantizeScalar(q.QType, q.Bytes[start:end], output)
	}
	if rowIdx >= w.Rows || len(output) != w.Cols {
		return fmt.Errorf("row bounds mismatch: row=%d rows=%d out=%d cols=%d", rowIdx, w.Rows, len(output), w.Cols)
	}
	copy(output, w.Data[rowIdx*w.Cols:(rowIdx+1)*w.Cols])
	return nil
}

func transposeF32(data []float32, ggufRows, ggufCols int) []float32 {
	result := make([]float32, len(data))
	for r := 0; r < ggufRows; r++ {
		for c := 0; c < ggufCols; c++ {
			result[c*ggufRows+r] = data[r*ggufCols+c]
		}
	}
	return result
}

func quantizedGemvSupported(qtype quantization.Type, inDim int) bool {
	switch qtype {
	case quantization.TypeQ4_K_S, quantization.TypeQ4_K_M, quantization.TypeQ2_K, quantization.TypeQ6_K:
		return inDim%256 == 0
	case quantization.TypeQ8_0:
		return inDim%32 == 0
	default:
		return false
	}
}
