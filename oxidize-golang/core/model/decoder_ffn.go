package model

import (
	"math"

	"github.com/Zapdev-labs/oxidize/golang/core/tensor"
)

func (s *LlamaDecoderStack) runDenseOrMoEFFN(layer *DecoderLayer, normed, out []float32) error {
	h := s.Config.HiddenSize
	inter := s.Config.IntermediateSize
	hasDense := layer.MLPGate.IsLoaded() && layer.MLPUp.IsLoaded() && layer.MLPDown.IsLoaded()
	if layer.MoE.Loaded() {
		return moeFFNForward(layer.MoE, h, inter, s.Config.NumExpertsPerToken, normed, out)
	}
	if !hasDense {
		for i := range out {
			out[i] = 0
		}
		return nil
	}
	gate := make([]float32, inter)
	up := make([]float32, inter)
	if err := layer.MLPGate.Gemv(normed, gate); err != nil {
		return err
	}
	if err := layer.MLPUp.Gemv(normed, up); err != nil {
		return err
	}
	applyGLU(gate, up, s.Config.GeluFFN)
	return layer.MLPDown.Gemv(gate, out)
}

// applyGLU computes gate[i] = activation(gate[i]) * up[i] in place, using
// GeGLU (tanh-GELU) when gelu is true, otherwise SwiGLU (SiLU).
func applyGLU(gate, up []float32, gelu bool) {
	n := len(gate)
	if len(up) < n {
		n = len(up)
	}
	if gelu {
		const k = 0.7978845608 // sqrt(2/pi)
		for i := 0; i < n; i++ {
			g := gate[i]
			act := 0.5 * g * (1 + float32(math.Tanh(float64(k*(g+0.044715*g*g*g)))))
			gate[i] = act * up[i]
		}
		return
	}
	for i := 0; i < n; i++ {
		g := gate[i]
		gate[i] = g * (1 / (1 + float32(math.Exp(float64(-g))))) * up[i]
	}
}

// mlpNorm returns the pre-MLP normalization weights: PreFFNLayernorm for Gemma
// sandwich models, otherwise PostAttentionLayernorm.
func (s *LlamaDecoderStack) mlpNorm(layer *DecoderLayer) []float32 {
	if s.Config.SandwichNorm && len(layer.PreFFNLayernorm) > 0 {
		return layer.PreFFNLayernorm
	}
	return layer.PostAttentionLayernorm
}

func (s *LlamaDecoderStack) runPostAttentionFFN(layer *DecoderLayer, hidden, out []float32) error {
	normed := make([]float32, len(hidden))
	if err := tensor.RMSNormF32(hidden, s.mlpNorm(layer), normed, s.Config.RMSNormEps); err != nil {
		return err
	}
	if err := s.runDenseOrMoEFFN(layer, normed, out); err != nil {
		return err
	}
	// Gemma sandwich norm: normalize the FFN output before its residual add.
	// Reuse the pre-MLP `normed` buffer as scratch (RMSNorm needs distinct
	// in/out buffers) to avoid a per-call allocation in this hot path.
	if s.Config.SandwichNorm && len(layer.PostFFNLayernorm) > 0 {
		if err := tensor.RMSNormF32(out, layer.PostFFNLayernorm, normed, s.Config.RMSNormEps); err != nil {
			return err
		}
		copy(out, normed)
	}
	return nil
}
