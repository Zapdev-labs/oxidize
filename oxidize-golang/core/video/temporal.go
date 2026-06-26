package video

import (
	"fmt"
	"math"

	"github.com/Zapdev-labs/oxidize/golang/core/tensor"
)

// TemporalLayerWeights holds the weights of one temporal self-attention layer.
// Mirrors temporal.rs:TemporalLayerWeights.
type TemporalLayerWeights struct {
	AttnNorm []float32
	QProj    []float32
	KProj    []float32
	VProj    []float32
	OProj    []float32
	FfnNorm  []float32
	FfnGate  []float32
	FfnUp    []float32
	FfnDown  []float32
}

// ZeroTemporalLayerWeights builds zero-initialized layer weights.
func ZeroTemporalLayerWeights(cfg TemporalConfig) TemporalLayerWeights {
	h := cfg.HiddenSize
	inter := cfg.IntermediateSize
	ones := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = 1.0
		}
		return s
	}
	return TemporalLayerWeights{
		AttnNorm: ones(h),
		QProj:    make([]float32, h*h),
		KProj:    make([]float32, h*h),
		VProj:    make([]float32, h*h),
		OProj:    make([]float32, h*h),
		FfnNorm:  ones(h),
		FfnGate:  make([]float32, h*inter),
		FfnUp:    make([]float32, h*inter),
		FfnDown:  make([]float32, inter*h),
	}
}

// TemporalWeights holds the full temporal encoder weights.
// Mirrors temporal.rs:TemporalWeights.
type TemporalWeights struct {
	Layers    []TemporalLayerWeights
	FinalNorm []float32
	// ClsToken is the optional learnable token (len hidden_size). Empty when
	// UseClsToken is false.
	ClsToken []float32
}

// ZeroTemporalWeights builds zero-initialized weights for cfg.
func ZeroTemporalWeights(cfg TemporalConfig) TemporalWeights {
	layers := make([]TemporalLayerWeights, cfg.NumLayers)
	for i := range layers {
		layers[i] = ZeroTemporalLayerWeights(cfg)
	}
	finalNorm := make([]float32, cfg.HiddenSize)
	for i := range finalNorm {
		finalNorm[i] = 1.0
	}
	var cls []float32
	if cfg.UseClsToken {
		cls = make([]float32, cfg.HiddenSize)
	}
	return TemporalWeights{Layers: layers, FinalNorm: finalNorm, ClsToken: cls}
}

// TemporalWorkspace holds persistent scratch buffers for the temporal encoder.
// Allocate once via NewTemporalWorkspace and reuse across calls. Mirrors
// temporal.rs:TemporalWorkspace.
type TemporalWorkspace struct {
	Hidden      []float32
	Residual    []float32
	QKV         []float32
	Attn        []float32
	AttnSoftmax []float32
	AttnOut     []float32
	FfnGate     []float32
	FfnUp       []float32
	FfnSilu     []float32
	FfnDown     []float32
	// normed / proj scratch reused per layer to avoid per-layer allocation.
	normed  []float32
	projOut []float32
	rope    []float32
}

// NewTemporalWorkspace allocates all scratch buffers for cfg's worst case.
func NewTemporalWorkspace(cfg TemporalConfig) *TemporalWorkspace {
	h := cfg.HiddenSize
	inter := cfg.IntermediateSize
	seq := cfg.MaxFrames + 1
	attnSize := cfg.NumHeads * seq * seq
	return &TemporalWorkspace{
		Hidden:      make([]float32, seq*h),
		Residual:    make([]float32, seq*h),
		QKV:         make([]float32, seq*3*h),
		Attn:        make([]float32, attnSize),
		AttnSoftmax: make([]float32, attnSize),
		AttnOut:     make([]float32, seq*h),
		FfnGate:     make([]float32, seq*inter),
		FfnUp:       make([]float32, seq*inter),
		FfnSilu:     make([]float32, seq*inter),
		FfnDown:     make([]float32, seq*h),
		normed:      make([]float32, seq*h),
		projOut:     make([]float32, seq*h),
		rope:        make([]float32, h),
	}
}

// ForwardTemporal runs the temporal encoder over a [inputSeqLen, hidden] input
// matrix and returns the normalized [seqLen, hidden] output (seqLen includes
// the cls token when enabled). Mirrors temporal.rs:forward_temporal.
func ForwardTemporal(cfg TemporalConfig, weights *TemporalWeights, input []float32, inputSeqLen int, ws *TemporalWorkspace) ([]float32, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if inputSeqLen == 0 || inputSeqLen > cfg.MaxFrames {
		return nil, &Error{Message: fmt.Sprintf(
			"frame count %d out of range [1, %d]", inputSeqLen, cfg.MaxFrames)}
	}
	h := cfg.HiddenSize
	if len(input) != inputSeqLen*h {
		return nil, &Error{Message: fmt.Sprintf(
			"input buffer length %d does not match seq_len*hidden (%d*%d)",
			len(input), inputSeqLen, h)}
	}
	if len(weights.Layers) != cfg.NumLayers {
		return nil, &Error{Message: fmt.Sprintf(
			"temporal_layers shape mismatch: expected %d got %d", cfg.NumLayers, len(weights.Layers))}
	}

	useCls := len(weights.ClsToken) != 0
	seqLen := inputSeqLen
	if useCls {
		seqLen++
	}
	if ws == nil {
		ws = NewTemporalWorkspace(cfg)
	}

	hidden := ws.Hidden[:seqLen*h]
	if useCls {
		copy(hidden[:h], weights.ClsToken)
		copy(hidden[h:h+inputSeqLen*h], input)
	} else {
		copy(hidden, input)
	}

	for layerIdx := range weights.Layers {
		if err := forwardTemporalLayer(cfg, &weights.Layers[layerIdx], hidden, seqLen, ws); err != nil {
			return nil, &Error{Message: fmt.Sprintf("layer %d: %v", layerIdx, err)}
		}
	}

	// Final norm.
	out := make([]float32, seqLen*h)
	for row := 0; row < seqLen; row++ {
		if err := tensor.RMSNormF32(hidden[row*h:(row+1)*h], weights.FinalNorm, out[row*h:(row+1)*h], cfg.RmsNormEps); err != nil {
			return nil, &Error{Message: fmt.Sprintf("final rms norm: %v", err)}
		}
	}
	return out, nil
}

func forwardTemporalLayer(cfg TemporalConfig, layer *TemporalLayerWeights, hidden []float32, seqLen int, ws *TemporalWorkspace) error {
	h := cfg.HiddenSize
	inter := cfg.IntermediateSize
	headDim := cfg.HeadDim()
	if headDim == 0 {
		return &Error{Message: "head_dim must be non-zero"}
	}
	if len(layer.QProj) != h*h || len(layer.KProj) != h*h || len(layer.VProj) != h*h || len(layer.OProj) != h*h {
		return &Error{Message: "QKV/O projection shape mismatch"}
	}
	if len(layer.FfnGate) != h*inter || len(layer.FfnUp) != h*inter || len(layer.FfnDown) != inter*h {
		return &Error{Message: "FFN projection shape mismatch"}
	}

	// ---- Pre-norm + QKV + attention ----
	residual := ws.Residual[:seqLen*h]
	copy(residual, hidden[:seqLen*h])

	normed := ws.normed[:seqLen*h]
	for row := 0; row < seqLen; row++ {
		if err := tensor.RMSNormF32(hidden[row*h:(row+1)*h], layer.AttnNorm, normed[row*h:(row+1)*h], cfg.RmsNormEps); err != nil {
			return fmt.Errorf("attn rms norm: %w", err)
		}
	}

	qkv := ws.QKV[:seqLen*3*h]
	qPart := qkv[:seqLen*h]
	kPart := qkv[seqLen*h : 2*seqLen*h]
	vPart := qkv[2*seqLen*h : 3*seqLen*h]
	if err := tensor.GemmF32(normed, layer.QProj, seqLen, h, h, qPart); err != nil {
		return fmt.Errorf("q_proj: %w", err)
	}
	if err := tensor.GemmF32(normed, layer.KProj, seqLen, h, h, kPart); err != nil {
		return fmt.Errorf("k_proj: %w", err)
	}
	if err := tensor.GemmF32(normed, layer.VProj, seqLen, h, h, vPart); err != nil {
		return fmt.Errorf("v_proj: %w", err)
	}

	// Apply RoPE to Q and K along the time axis (position = row index).
	rope := ws.rope[:headDim]
	for pos := 0; pos < seqLen; pos++ {
		for head := 0; head < cfg.NumHeads; head++ {
			start := pos*h + head*headDim
			end := start + headDim
			if err := tensor.ApplyRopeHeadF32(qPart[start:end], rope, pos, headDim, cfg.RopeTheta); err != nil {
				return fmt.Errorf("q rope: %w", err)
			}
			copy(qPart[start:end], rope)
			if err := tensor.ApplyRopeHeadF32(kPart[start:end], rope, pos, headDim, cfg.RopeTheta); err != nil {
				return fmt.Errorf("k rope: %w", err)
			}
			copy(kPart[start:end], rope)
		}
	}

	attnSize := cfg.NumHeads * seqLen * seqLen
	scores := ws.Attn[:attnSize]
	softmax := ws.AttnSoftmax[:attnSize]
	attnOut := ws.AttnOut[:seqLen*h]
	computeCausalAttention(qPart, kPart, vPart, scores, softmax, attnOut, seqLen, h, cfg.NumHeads, headDim)

	// Output projection.
	projOut := ws.projOut[:seqLen*h]
	if err := tensor.GemmF32(attnOut, layer.OProj, seqLen, h, h, projOut); err != nil {
		return fmt.Errorf("o_proj: %w", err)
	}
	for i := 0; i < seqLen*h; i++ {
		hidden[i] = residual[i] + projOut[i]
	}

	// ---- FFN block ----
	copy(residual, hidden[:seqLen*h])
	for row := 0; row < seqLen; row++ {
		if err := tensor.RMSNormF32(hidden[row*h:(row+1)*h], layer.FfnNorm, normed[row*h:(row+1)*h], cfg.RmsNormEps); err != nil {
			return fmt.Errorf("ffn rms norm: %w", err)
		}
	}

	gate := ws.FfnGate[:seqLen*inter]
	up := ws.FfnUp[:seqLen*inter]
	silu := ws.FfnSilu[:seqLen*inter]
	down := ws.FfnDown[:seqLen*h]
	if err := tensor.GemmF32(normed, layer.FfnGate, seqLen, h, inter, gate); err != nil {
		return fmt.Errorf("ffn_gate: %w", err)
	}
	if err := tensor.GemmF32(normed, layer.FfnUp, seqLen, h, inter, up); err != nil {
		return fmt.Errorf("ffn_up: %w", err)
	}
	for i := 0; i < seqLen*inter; i++ {
		g := gate[i]
		silu[i] = g * sigmoid(g) * up[i]
	}
	if err := tensor.GemmF32(silu, layer.FfnDown, seqLen, inter, h, down); err != nil {
		return fmt.Errorf("ffn_down: %w", err)
	}
	for i := 0; i < seqLen*h; i++ {
		hidden[i] = residual[i] + down[i]
	}
	return nil
}

// computeCausalAttention runs per-head causal self-attention over a
// [seq, hidden] matrix. Mirrors temporal.rs:compute_causal_attention.
func computeCausalAttention(q, k, v, scores, softmaxOut, output []float32, seqLen, hidden, numHeads, headDim int) {
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	for hIdx := 0; hIdx < numHeads; hIdx++ {
		for qPos := 0; qPos < seqLen; qPos++ {
			qOff := qPos*hidden + hIdx*headDim
			qRow := q[qOff : qOff+headDim]
			maxScore := float32(math.Inf(-1))
			scoreRow := scores[hIdx*seqLen*seqLen+qPos*seqLen:]
			for kPos := 0; kPos <= qPos; kPos++ {
				kOff := kPos*hidden + hIdx*headDim
				kRow := k[kOff : kOff+headDim]
				var dot float32
				for d := 0; d < headDim; d++ {
					dot += qRow[d] * kRow[d]
				}
				s := dot * scale
				scoreRow[kPos] = s
				if s > maxScore {
					maxScore = s
				}
			}
			for kPos := qPos + 1; kPos < seqLen; kPos++ {
				scoreRow[kPos] = float32(math.Inf(-1))
			}
			if math.IsInf(float64(maxScore), 0) || math.IsNaN(float64(maxScore)) {
				maxScore = 0.0
			}
			softmaxRow := softmaxOut[hIdx*seqLen*seqLen+qPos*seqLen:]
			var sum float32
			for kPos := 0; kPos <= qPos; kPos++ {
				p := float32(math.Exp(float64(scoreRow[kPos] - maxScore)))
				softmaxRow[kPos] = p
				sum += p
			}
			for kPos := qPos + 1; kPos < seqLen; kPos++ {
				softmaxRow[kPos] = 0.0
			}
			invSum := float32(1.0)
			if sum > 0.0 {
				invSum = 1.0 / sum
			}
			for kPos := 0; kPos < seqLen; kPos++ {
				softmaxRow[kPos] *= invSum
			}

			outOff := qPos*hidden + hIdx*headDim
			outRow := output[outOff : outOff+headDim]
			for d := 0; d < headDim; d++ {
				outRow[d] = 0.0
			}
			for kPos := 0; kPos <= qPos; kPos++ {
				vOff := kPos*hidden + hIdx*headDim
				vRow := v[vOff : vOff+headDim]
				a := softmaxRow[kPos]
				for d := 0; d < headDim; d++ {
					outRow[d] += a * vRow[d]
				}
			}
		}
	}
}

func sigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(float64(-x))))
}
