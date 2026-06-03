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
	for i := 0; i < inter; i++ {
		g := gate[i]
		gate[i] = g * (1 / (1 + float32(math.Exp(float64(-g))))) * up[i]
	}
	return layer.MLPDown.Gemv(gate, out)
}

func (s *LlamaDecoderStack) runPostAttentionFFN(layer *DecoderLayer, hidden, out []float32) error {
	normed := make([]float32, len(hidden))
	if err := tensor.RMSNormF32(hidden, layer.PostAttentionLayernorm, normed, s.Config.RMSNormEps); err != nil {
		return err
	}
	return s.runDenseOrMoEFFN(layer, normed, out)
}
