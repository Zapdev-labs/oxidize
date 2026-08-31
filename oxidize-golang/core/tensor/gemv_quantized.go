package tensor

import (
	"github.com/Zapdev-labs/oxidize/golang/core/quantization"
)

// GemvQuantizedDispatch computes output = dequant(matrix) * vector, dispatching
func GemvQuantizedDispatch(
	qbytes []byte,
	qtype quantization.Type,
	rows, cols int,
	vector, output []float32,
) error {
	if rows <= 0 || cols <= 0 {
		return &GemvError{Message: "invalid dims"}
	}
	if len(vector) < cols {
		return &GemvError{Message: "vector buffer too small"}
	}
	if len(output) < rows {
		return &GemvError{Message: "output buffer too small"}
	}

	switch qtype {
	case quantization.TypeQ4_K_S, quantization.TypeQ4_K_M:
		if cols%quantization.QK_K != 0 {
			break // fall through to scalar dequant path below
		}
		return gemvQ4KQ8K(qbytes, rows, cols, vector, output)
	}

	// Scalar fallback: dequant each row, then dot.
	dequant, err := dequantFor(qtype)
	if err != nil {
		return err
	}
	return GemvQuantizedF32(qbytes, dequant, rows, cols, vector, output, nil)
}

func dequantFor(qtype quantization.Type) (func([]byte, []float32) error, error) {
	switch qtype {
	case quantization.TypeQ4_K_S, quantization.TypeQ4_K_M:
		return quantization.DequantQ4_K, nil
	case quantization.TypeQ5_K_S, quantization.TypeQ5_K_M:
		return quantization.DequantQ5_K, nil
	case quantization.TypeQ6_K:
		return quantization.DequantQ6_K, nil
	case quantization.TypeNVFP4:
		return quantization.DequantNVFP4, nil
	default:
		return nil, &GemvError{Message: "unsupported quantization type for gemv dispatch: " + qtype.String()}
	}
}

// gemvQ4KQ8K runs the fused Q4_K × Q8_K integer GEMV. The input vector is
// quantized to Q8_K once and shared across all rows.
func gemvQ4KQ8K(qbytes []byte, rows, cols int, vector, output []float32) error {
	blocksPerRow := cols / quantization.QK_K
	bytesPerRow := blocksPerRow * quantization.BLOCK_Q4_K_SIZE
	if len(qbytes) < rows*bytesPerRow {
		return &GemvError{Message: "qbytes buffer too small for q4_k gemv"}
	}

	// Quantize the activation vector once.
	q8 := make([]byte, blocksPerRow*quantization.BLOCK_Q8_K_SIZE)
	if err := quantization.QuantizeVectorQ8KInto(vector[:cols], blocksPerRow, q8); err != nil {
		return err
	}

	parallelizeRows(rows, func(start, end int) {
		r := start
		// 4-row unrolled kernel for the bulk of the range.
		for ; r+4 <= end; r += 4 {
			base := r * bytesPerRow
			var out4 [4]float32
			quantization.Q4KQ8KRowDotX4(
				qbytes[base:base+4*bytesPerRow],
				bytesPerRow, blocksPerRow, q8, &out4,
			)
			output[r] = out4[0]
			output[r+1] = out4[1]
			output[r+2] = out4[2]
			output[r+3] = out4[3]
		}
		// Scalar tail.
		for ; r < end; r++ {
			base := r * bytesPerRow
			output[r] = quantization.Q4KQ8KRowDot(qbytes[base:base+bytesPerRow], blocksPerRow, q8)
		}
	})
	return nil
}
