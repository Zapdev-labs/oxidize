package tensor

import (
	"fmt"
	"math"
	"sync"
)

// GemvF32 computes output = matrix * vector, where matrix is row-major
// (rows x cols) and vector/output have length `cols` and `rows` respectively.
func GemvF32(matrix []float32, rows, cols int, vector, output []float32) error {
	if rows <= 0 || cols <= 0 {
		return &GemvError{Message: fmt.Sprintf("invalid dims rows=%d cols=%d", rows, cols)}
	}
	if len(matrix) < rows*cols {
		return &GemvError{Message: "matrix buffer too small"}
	}
	if len(vector) < cols {
		return &GemvError{Message: "vector buffer too small"}
	}
	if len(output) < rows {
		return &GemvError{Message: "output buffer too small"}
	}
	parallelizeRows(rows, func(start, end int) {
		for r := start; r < end; r++ {
			row := matrix[r*cols : (r+1)*cols]
			var sum float32
			for c, v := range vector[:cols] {
				sum += row[c] * v
			}
			output[r] = sum
		}
	})
	return nil
}

// GemvF32Transposed computes output = matrix^T * vector, where matrix is
// stored row-major (rows x cols) but logically transposed.
func GemvF32Transposed(matrix []float32, rows, cols int, vector, output []float32) error {
	if len(matrix) < rows*cols {
		return &GemvError{Message: "matrix buffer too small"}
	}
	if len(vector) < rows {
		return &GemvError{Message: "vector buffer too small"}
	}
	if len(output) < cols {
		return &GemvError{Message: "output buffer too small"}
	}
	for c := 0; c < cols; c += TransposedGemvColChunk {
		end := c + TransposedGemvColChunk
		if end > cols {
			end = cols
		}
		parallelizeRange(end-c, func(start, stop int) {
			for k := start; k < stop; k++ {
				col := c + k
				var sum float32
				for r := 0; r < rows; r++ {
					sum += matrix[r*cols+col] * vector[r]
				}
				output[col] = sum
			}
		})
	}
	return nil
}

// GemvQuantizedF32 computes output = dequant(matrix) * vector for a quantized
// weight matrix stored in `qbytes`. The dequantization scheme is selected by
// the provided `dequant` function.
func GemvQuantizedF32(qbytes []byte, dequant func([]byte, []float32) error, rows, cols int, vector, output []float32, scratch []float32) error {
	if scratch == nil {
		scratch = make([]float32, cols)
	} else if len(scratch) < cols {
		return &GemvError{Message: "scratch too small"}
	}
	bytesPerRow := len(qbytes) / rows
	if bytesPerRow*rows != len(qbytes) {
		return &GemvError{Message: "qbytes not aligned to row boundary"}
	}
	parallelizeRows(rows, func(start, end int) {
		buf := scratch[:cols]
		for r := start; r < end; r++ {
			if err := dequant(qbytes[r*bytesPerRow:(r+1)*bytesPerRow], buf); err != nil {
				continue
			}
			var sum float32
			for c := 0; c < cols; c++ {
				sum += buf[c] * vector[c]
			}
			output[r] = sum
		}
	})
	return nil
}

// GemvQuantizedF32Transposed computes output = dequant(matrix)^T * vector.
func GemvQuantizedF32Transposed(qbytes []byte, dequant func([]byte, []float32) error, rows, cols int, vector, output []float32, scratch []float32) error {
	if scratch == nil || len(scratch) < cols {
		scratch = make([]float32, cols)
	}
	bytesPerRow := len(qbytes) / rows
	if bytesPerRow*rows != len(qbytes) {
		return &GemvError{Message: "qbytes not aligned to row boundary"}
	}
	row := make([]float32, cols)
	for r := 0; r < rows; r++ {
		if err := dequant(qbytes[r*bytesPerRow:(r+1)*bytesPerRow], row); err != nil {
			return err
		}
		v := vector[r]
		for c := 0; c < cols; c++ {
			output[c] += row[c] * v
		}
	}
	return nil
}

// GemmF32 computes output (rows x cols) = left (rows x shared) * right (shared x cols).
func GemmF32(left, right []float32, rows, shared, cols int, output []float32) error {
	if len(left) < rows*shared {
		return &GemmError{Message: "left buffer too small"}
	}
	if len(right) < shared*cols {
		return &GemmError{Message: "right buffer too small"}
	}
	if len(output) < rows*cols {
		return &GemmError{Message: "output buffer too small"}
	}
	parallelizeRows(rows, func(start, end int) {
		for r := start; r < end; r++ {
			for c := 0; c < cols; c++ {
				var sum float32
				for k := 0; k < shared; k++ {
					sum += left[r*shared+k] * right[k*cols+c]
				}
				output[r*cols+c] = sum
			}
		}
	})
	return nil
}

// GemmQuantizedF32 computes the GEMM of a dequantized left factor with right.
func GemmQuantizedF32(qbytes []byte, dequant func([]byte, []float32) error, rows, shared, cols int, right, output []float32, scratch []float32) error {
	if scratch == nil || len(scratch) < shared {
		scratch = make([]float32, shared)
	}
	bytesPerRow := len(qbytes) / rows
	if bytesPerRow*rows != len(qbytes) {
		return &GemmError{Message: "qbytes not aligned to row boundary"}
	}
	parallelizeRows(rows, func(start, end int) {
		row := scratch[:shared]
		for r := start; r < end; r++ {
			if err := dequant(qbytes[r*bytesPerRow:(r+1)*bytesPerRow], row); err != nil {
				continue
			}
			for c := 0; c < cols; c++ {
				var sum float32
				for k := 0; k < shared; k++ {
					sum += row[k] * right[k*cols+c]
				}
				output[r*cols+c] = sum
			}
		}
	})
	return nil
}

// GemmF32TensorParallel performs a tensor-parallel all-reduce after the GEMM.
func GemmF32TensorParallel(left, right []float32, rows, shared, cols int, output, reduceBuf []float32) error {
	if err := GemmF32(left, right, rows, shared, cols, output); err != nil {
		return err
	}
	if reduceBuf != nil && len(reduceBuf) >= len(output) {
		for i := range output {
			output[i] += reduceBuf[i]
		}
	}
	return nil
}

// GemmI8 computes int8 GEMM. The inputs are interpreted as int8; the output
// is f32 (or a widened equivalent).
func GemmI8(left, right []int8, rows, shared, cols int, output []int32) error {
	if len(left) < rows*shared {
		return &GemmError{Message: "left buffer too small"}
	}
	if len(right) < shared*cols {
		return &GemmError{Message: "right buffer too small"}
	}
	if len(output) < rows*cols {
		return &GemmError{Message: "output buffer too small"}
	}
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			var sum int32
			for k := 0; k < shared; k++ {
				sum += int32(left[r*shared+k]) * int32(right[k*cols+c])
			}
			output[r*cols+c] = sum
		}
	}
	return nil
}

// GemmI4 computes a 4-bit GEMM. The right factor is nibble-packed.
func GemmI4(left []int8, right []uint8, rows, shared, cols int, output []int32) error {
	if len(left) < rows*shared {
		return &GemmError{Message: "left buffer too small"}
	}
	if len(right) < (shared*cols)/2 {
		return &GemmError{Message: "right buffer too small"}
	}
	if len(output) < rows*cols {
		return &GemmError{Message: "output buffer too small"}
	}
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			var sum int32
			for k := 0; k < shared; k++ {
				packed := right[(k*cols+c)/2]
				var nibble int8
				if (k*cols+c)%2 == 0 {
					nibble = unpackNibble(packed & 0x0F)
				} else {
					nibble = unpackNibble((packed >> 4) & 0x0F)
				}
				sum += int32(left[r*shared+k]) * int32(nibble)
			}
			output[r*cols+c] = sum
		}
	}
	return nil
}

func unpackNibble(n uint8) int8 {
	if n >= 8 {
		return int8(n) - 16
	}
	return int8(n)
}

func parallelizeRows(rows int, fn func(start, end int)) {
	if rows <= ParallelGemvMinOps/4 || rows < 32 {
		fn(0, rows)
		return
	}
	workers := parallelWorkers(rows)
	if workers <= 1 {
		fn(0, rows)
		return
	}
	var wg sync.WaitGroup
	chunk := (rows + workers - 1) / workers
	for w := 0; w < workers; w++ {
		start := w * chunk
		if start >= rows {
			break
		}
		end := start + chunk
		if end > rows {
			end = rows
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			fn(s, e)
		}(start, end)
	}
	wg.Wait()
}

func parallelizeRange(n int, fn func(start, end int)) {
	parallelizeRows(n, fn)
}

func parallelWorkers(n int) int {
	if n < 1 {
		return 1
	}
	if n < 256 {
		return 1
	}
	return 8
}

// Sigmoid computes element-wise sigmoid.
func Sigmoid(x, out []float32) error {
	if len(out) < len(x) {
		return &GemvError{Message: "out too small"}
	}
	for i, v := range x {
		out[i] = 1 / (1 + float32(math.Exp(float64(-v))))
	}
	return nil
}
