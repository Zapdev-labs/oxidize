package tensor

import "math"

// ApplyRopeHeadF32 applies RoPE to a single head vector (len headDim), matching
// oxidize-core apply_rope_f32.
func ApplyRopeHeadF32(input, output []float32, position, headDim int, theta float32) error {
	if headDim <= 0 || headDim%2 != 0 {
		return &RopeError{Message: "headDim must be positive and even"}
	}
	if len(input) < headDim || len(output) < headDim {
		return &RopeError{Message: "buffer too small for headDim"}
	}
	if position == 0 {
		copy(output[:headDim], input[:headDim])
		return nil
	}
	half := headDim / 2
	invHead := 1.0 / float32(headDim)
	freqMul := float32(math.Pow(float64(theta), -2.0*float64(invHead)))
	freq := float32(1)
	for i := 0; i < half; i++ {
		angle := float32(float64(position) * float64(freq))
		cos := float32(math.Cos(float64(angle)))
		sin := float32(math.Sin(float64(angle)))
		x0 := input[i]
		x1 := input[half+i]
		output[i] = x0*cos - x1*sin
		output[half+i] = x0*sin + x1*cos
		freq *= freqMul
	}
	return nil
}

// ApplyRopeF32 applies rotary position embedding to `input` (size headDim*2)
// starting at the given `position`.
func ApplyRopeF32(input, output []float32, position, headDim int, theta float32) error {
	if headDim <= 0 || headDim%2 != 0 {
		return &RopeError{Message: "headDim must be positive and even"}
	}
	if len(input) < headDim*2 {
		return &RopeError{Message: "input buffer too small"}
	}
	if len(output) < headDim*2 {
		return &RopeError{Message: "output buffer too small"}
	}
	for i := 0; i < headDim; i += 2 {
		freq := float32(1) / float32(math.Pow(float64(theta), float64(2*i)/float64(headDim)))
		angle := float32(float64(position) * float64(freq))
		cos := float32(math.Cos(float64(angle)))
		sin := float32(math.Sin(float64(angle)))
		x0 := input[i]
		x1 := input[i+1]
		output[i] = x0*cos - x1*sin
		output[i+1] = x0*sin + x1*cos
	}
	// copy the second half (paired) untouched
	copy(output[headDim:], input[headDim:headDim*2])
	return nil
}

// RMSNormF32 computes output = input / sqrt(mean(input^2) + eps) * weight.
func RMSNormF32(input, weight, output []float32, eps float32) error {
	if len(input) == 0 {
		return &RmsNormError{Message: "empty input"}
	}
	if len(weight) < len(input) {
		return &RmsNormError{Message: "weight too small"}
	}
	if len(output) < len(input) {
		return &RmsNormError{Message: "output too small"}
	}
	var sumSq float32
	for _, v := range input {
		sumSq += v * v
	}
	mean := sumSq / float32(len(input))
	inv := 1 / float32(math.Sqrt(float64(mean+eps)))
	for i, v := range input {
		output[i] = v * inv * weight[i]
	}
	return nil
}

// RmsNormGemvF32Transposed fuses RMSNorm with a transposed GEMV. The norm
// is applied to `input` (length `cols`) producing a normalized vector, then
// multiplied by `matrix` (rows x cols, transposed).
func RmsNormGemvF32Transposed(input, normWeight, matrix, output []float32, rows, cols int, eps float32) error {
	if len(input) < cols {
		return &RmsNormError{Message: "input too small"}
	}
	if len(normWeight) < cols {
		return &RmsNormError{Message: "norm weight too small"}
	}
	if len(matrix) < rows*cols {
		return &RmsNormError{Message: "matrix too small"}
	}
	if len(output) < rows {
		return &RmsNormError{Message: "output too small"}
	}
	normalized := make([]float32, cols)
	if err := RMSNormF32(input[:cols], normWeight[:cols], normalized, eps); err != nil {
		return err
	}
	return GemvF32Transposed(matrix, rows, cols, normalized, output)
}

// LayerNormF32 computes output = (input - mean(input)) / sqrt(var(input)+eps) * weight + bias.
func LayerNormF32(input, weight, bias, output []float32, eps float32) error {
	if len(input) == 0 {
		return &LayerNormError{Message: "empty input"}
	}
	if len(weight) < len(input) {
		return &LayerNormError{Message: "weight too small"}
	}
	if len(bias) < len(input) {
		return &LayerNormError{Message: "bias too small"}
	}
	if len(output) < len(input) {
		return &LayerNormError{Message: "output too small"}
	}
	var mean float32
	for _, v := range input {
		mean += v
	}
	mean /= float32(len(input))
	var variance float32
	for _, v := range input {
		d := v - mean
		variance += d * d
	}
	variance /= float32(len(input))
	inv := 1 / float32(math.Sqrt(float64(variance + eps)))
	for i, v := range input {
		output[i] = (v-mean)*inv*weight[i] + bias[i]
	}
	return nil
}

// SoftmaxF32 computes softmax along the last axis. `input` is shape (batch x dim).
func SoftmaxF32(input, output []float32, dim int) error {
	if dim <= 0 || len(input)%dim != 0 {
		return &SoftmaxError{Message: "dim must divide input length"}
	}
	batch := len(input) / dim
	for b := 0; b < batch; b++ {
		base := b * dim
		maxVal := input[base]
		for i := 1; i < dim; i++ {
			if input[base+i] > maxVal {
				maxVal = input[base+i]
			}
		}
		var sum float32
		for i := 0; i < dim; i++ {
			output[base+i] = float32(math.Exp(float64(input[base+i] - maxVal)))
			sum += output[base+i]
		}
		inv := 1 / sum
		for i := 0; i < dim; i++ {
			output[base+i] *= inv
		}
	}
	return nil
}

// ApplySwiGLUF32 computes output = silu(gate) * up. The gate, up, and output
// slices are all length `dim`.
func ApplySwiGLUF32(gate, up, output []float32) error {
	if len(gate) != len(up) {
		return &SwiGluError{Message: "gate/up length mismatch"}
	}
	if len(output) < len(gate) {
		return &SwiGluError{Message: "output too small"}
	}
	for i, g := range gate {
		silu := g / (1 + float32(math.Exp(float64(-g))))
		output[i] = silu * up[i]
	}
	return nil
}

// LinearActivationF32 computes output = activation(input @ weight^T + bias).
func LinearActivationF32(input, weight []float32, rows, shared, cols int, bias, output []float32, act ActivationFn) error {
	gemm := make([]float32, rows*cols)
	if err := GemmF32(weight, nil, rows, shared, cols, gemm); err != nil {
		return &LinearActivationError{Message: err.Error()}
	}
	// Apply bias and activation
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			v := gemm[r*cols+c]
			if bias != nil {
				v += bias[r*cols+c]
			}
			switch act {
			case ActivationSilu:
				v = v / (1 + float32(math.Exp(float64(-v))))
			case ActivationGelu:
				v = 0.5 * v * (1 + float32(math.Erf(float64(v)/math.Sqrt2)))
			case ActivationRelu:
				if v < 0 {
					v = 0
				}
			}
			output[r*cols+c] = v
		}
	}
	return nil
}

// ScaledDotProductAttentionF32 computes attention for a single query attending
// to a sequence of keys/values. This is the non-flash, non-paged baseline.
func ScaledDotProductAttentionF32(query, keys, values, output []float32, seqLen, headDim int, scale float32) error {
	if len(query) < headDim {
		return &AttentionError{Message: "query too small"}
	}
	if len(keys) < seqLen*headDim {
		return &AttentionError{Message: "keys too small"}
	}
	if len(values) < seqLen*headDim {
		return &AttentionError{Message: "values too small"}
	}
	if len(output) < headDim {
		return &AttentionError{Message: "output too small"}
	}
	scores := make([]float32, seqLen)
	for s := 0; s < seqLen; s++ {
		var sum float32
		for d := 0; d < headDim; d++ {
			sum += query[d] * keys[s*headDim+d]
		}
		scores[s] = sum * scale
	}
	maxScore := scores[0]
	for _, s := range scores {
		if s > maxScore {
			maxScore = s
		}
	}
	var total float32
	for i, s := range scores {
		scores[i] = float32(math.Exp(float64(s - maxScore)))
		total += scores[i]
	}
	inv := 1 / total
	for i := range scores {
		scores[i] *= inv
	}
	for d := 0; d < headDim; d++ {
		var acc float32
		for s := 0; s < seqLen; s++ {
			acc += scores[s] * values[s*headDim+d]
		}
		output[d] = acc
	}
	return nil
}
