package model

import (
	"fmt"

	"github.com/Zapdev-labs/oxidize/golang/core/flash_attention"
	"github.com/Zapdev-labs/oxidize/golang/core/tensor"
)

// ForwardToken runs one decode step and returns the final hidden state (pre-lm_head).
func (s *LlamaDecoderStack) ForwardToken(token uint32) ([]float32, error) {
	return s.forwardTokenWithContext(token, nil, nil)
}

// forwardTokenWithContext runs decode with optional precomputed K/V context vectors per layer
// (used by DFlash target-hidden fusion).
func (s *LlamaDecoderStack) forwardTokenWithContext(
	token uint32,
	targetContext []float32,
	kvContext func(layerIdx int, targetCtx []float32) (kCtx, vCtx []float32, err error),
) ([]float32, error) {
	h := s.Config.HiddenSize
	hidden := make([]float32, h)

	if err := s.fillTokenEmbedding(token, hidden); err != nil {
		return nil, err
	}
	s.scaleEmbedding(hidden)

	for layerIdx := range s.Layers {
		layer := &s.Layers[layerIdx]
		headDim := s.Config.KVHeadDim()
		if len(layer.Attention.QNormWeight) > 0 {
			headDim = len(layer.Attention.QNormWeight)
		}
		numHeads := s.Config.NumAttentionHeads
		numKV := s.Config.NumKeyValueHeads
		qSize := numHeads * headDim
		kvLen := numKV * headDim

		attnOut := make([]float32, qSize)
		mlpOut := make([]float32, h)

		{
			normed := make([]float32, h)
			if err := tensor.RMSNormF32(hidden, layer.InputLayernorm, normed, s.Config.RMSNormEps); err != nil {
				return nil, err
			}

			q := make([]float32, qSize)
			k := make([]float32, kvLen)
			v := make([]float32, kvLen)
			if layer.Attention.QProj.IsLoaded() {
				if err := layer.Attention.QProj.Gemv(normed, q); err != nil {
					return nil, err
				}
			}
			if layer.Attention.KProj.IsLoaded() {
				if err := layer.Attention.KProj.Gemv(normed, k); err != nil {
					return nil, err
				}
			}
			if layer.Attention.VProj.IsLoaded() {
				if err := layer.Attention.VProj.Gemv(normed, v); err != nil {
					return nil, err
				}
			}

			var kCtx, vCtx []float32
			if kvContext != nil && targetContext != nil {
				var err error
				kCtx, vCtx, err = kvContext(layerIdx, targetContext)
				if err != nil {
					return nil, err
				}
			}

			headScratch := make([]float32, headDim)
			pos := s.PositionOffset

			applyQKNorm := func(vec []float32, weight []float32, heads int) error {
				if len(weight) != headDim {
					return nil
				}
				for hi := 0; hi < heads; hi++ {
					start := hi * headDim
					if err := tensor.RMSNormF32(vec[start:start+headDim], weight, headScratch, s.Config.RMSNormEps); err != nil {
						return err
					}
					copy(vec[start:start+headDim], headScratch)
				}
				return nil
			}
			if err := applyQKNorm(q, layer.Attention.QNormWeight, numHeads); err != nil {
				return nil, err
			}
			if err := applyQKNorm(k, layer.Attention.KNormWeight, numKV); err != nil {
				return nil, err
			}
			if kCtx != nil {
				if err := applyQKNorm(kCtx, layer.Attention.KNormWeight, numKV); err != nil {
					return nil, err
				}
			}

			layerRope := s.Config.LayerRopeTheta(layerIdx)
			if !s.Config.UseAlibi {
				applyRope := func(vec []float32, heads int) error {
					for hi := 0; hi < heads; hi++ {
						start := hi * headDim
						if err := tensor.ApplyRopeHeadF32(vec[start:start+headDim], headScratch, pos, headDim, layerRope); err != nil {
							return err
						}
						copy(vec[start:start+headDim], headScratch)
					}
					return nil
				}
				if err := applyRope(q, numHeads); err != nil {
					return nil, err
				}
				if err := applyRope(k, numKV); err != nil {
					return nil, err
				}
				if kCtx != nil {
					if err := applyRope(kCtx, numKV); err != nil {
						return nil, err
					}
				}
			}

			cache := &s.KVCache[layerIdx]
			if kCtx != nil {
				cache.Keys = append(cache.Keys, kCtx...)
				cache.Values = append(cache.Values, vCtx...)
				cache.SeqLen++
			}
			cache.Keys = append(cache.Keys, k...)
			cache.Values = append(cache.Values, v...)
			cache.SeqLen++

			window := s.Config.LayerSlidingWindow(layerIdx)
			var attnErr error
			switch {
			case s.Config.UseAlibi && len(s.AlibiSlopes) > 0:
				attnErr = flash_attention.FlashAttentionDecodeHeadsGQAAlibi(
					q, cache.Keys, cache.Values, attnOut,
					cache.SeqLen, headDim, kvLen, numHeads, numKV,
					s.AlibiSlopes, pos, window,
				)
			case window > 0:
				attnErr = flash_attention.FlashAttentionDecodeHeadsGQAWindow(
					q, cache.Keys, cache.Values, attnOut,
					cache.SeqLen, headDim, kvLen, numHeads, numKV, window,
				)
			default:
				attnErr = flash_attention.FlashAttentionDecodeHeadsGQA(
					q, cache.Keys, cache.Values, attnOut,
					cache.SeqLen, headDim, kvLen, numHeads, numKV,
				)
			}
			if attnErr != nil {
				return nil, attnErr
			}

			oResult := make([]float32, h)
			if layer.Attention.OProj.IsLoaded() {
				if err := layer.Attention.OProj.Gemv(attnOut, oResult); err != nil {
					return nil, err
				}
			} else {
				copy(oResult, attnOut)
				if qSize < h {
					for i := qSize; i < h; i++ {
						oResult[i] = 0
					}
				}
			}
			attnOut = oResult

			if s.Config.ParallelAttnFFN {
				if err := s.runPostAttentionFFN(layer, hidden, mlpOut); err != nil {
					return nil, err
				}
			}
		}

		if s.Config.ParallelAttnFFN {
			for i := 0; i < h; i++ {
				hidden[i] += attnOut[i] + mlpOut[i]
			}
		} else {
			// Gemma sandwich norm: normalize the attention output before residual.
			if s.Config.SandwichNorm && len(layer.PostAttentionLayernorm) == h {
				normedAttn := make([]float32, h)
				if err := tensor.RMSNormF32(attnOut, layer.PostAttentionLayernorm, normedAttn, s.Config.RMSNormEps); err != nil {
					return nil, err
				}
				attnOut = normedAttn
			}
			for i := 0; i < h; i++ {
				hidden[i] += attnOut[i]
			}
			if err := s.runPostAttentionFFN(layer, hidden, mlpOut); err != nil {
				return nil, err
			}
			for i := 0; i < h; i++ {
				hidden[i] += mlpOut[i]
			}
		}
	}

	if len(s.Norm) > 0 {
		out := make([]float32, h)
		if err := tensor.RMSNormF32(hidden, s.Norm, out, s.Config.RMSNormEps); err != nil {
			return nil, err
		}
		hidden = out
	}

	s.PositionOffset++
	return hidden, nil
}

// ForwardBatch prefills multiple tokens; returns the last token hidden state.
func (s *LlamaDecoderStack) ForwardBatch(tokens []uint32) ([]float32, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("empty token batch")
	}
	if len(tokens) == 1 {
		return s.ForwardToken(tokens[0])
	}

	b := len(tokens)
	h := s.Config.HiddenSize
	hidden := make([]float32, b*h)

	if s.TokEmbeddings.IsLoaded() {
		for t, tok := range tokens {
			if err := s.fillTokenEmbedding(tok, hidden[t*h:(t+1)*h]); err != nil {
				return nil, err
			}
			s.scaleEmbedding(hidden[t*h : (t+1)*h])
		}
	}

	for layerIdx := range s.Layers {
		layer := &s.Layers[layerIdx]
		headDim := s.Config.KVHeadDim()
		if len(layer.Attention.QNormWeight) > 0 {
			headDim = len(layer.Attention.QNormWeight)
		}
		numHeads := s.Config.NumAttentionHeads
		numKV := s.Config.NumKeyValueHeads
		qSize := numHeads * headDim
		kvLen := numKV * headDim
		layerRope := s.Config.LayerRopeTheta(layerIdx)
		layerWindow := s.Config.LayerSlidingWindow(layerIdx)

		normed := make([]float32, b*h)
		for t := 0; t < b; t++ {
			if err := tensor.RMSNormF32(hidden[t*h:(t+1)*h], layer.InputLayernorm, normed[t*h:(t+1)*h], s.Config.RMSNormEps); err != nil {
				return nil, err
			}
		}

		qAll := make([]float32, b*qSize)
		kAll := make([]float32, b*kvLen)
		vAll := make([]float32, b*kvLen)
		if layer.Attention.QProj.IsLoaded() {
			if err := layer.Attention.QProj.Gemm(normed, qAll, b); err != nil {
				return nil, err
			}
		}
		if layer.Attention.KProj.IsLoaded() {
			if err := layer.Attention.KProj.Gemm(normed, kAll, b); err != nil {
				return nil, err
			}
		}
		if layer.Attention.VProj.IsLoaded() {
			if err := layer.Attention.VProj.Gemm(normed, vAll, b); err != nil {
				return nil, err
			}
		}

		headScratch := make([]float32, headDim)
		for t := 0; t < b; t++ {
			pos := s.PositionOffset + t
			q := qAll[t*qSize : (t+1)*qSize]
			k := kAll[t*kvLen : (t+1)*kvLen]
			if len(layer.Attention.QNormWeight) == headDim {
				for hi := 0; hi < numHeads; hi++ {
					start := hi * headDim
					if err := tensor.RMSNormF32(q[start:start+headDim], layer.Attention.QNormWeight, headScratch, s.Config.RMSNormEps); err != nil {
						return nil, err
					}
					copy(q[start:start+headDim], headScratch)
				}
			}
			if len(layer.Attention.KNormWeight) == headDim {
				for hi := 0; hi < numKV; hi++ {
					start := hi * headDim
					if err := tensor.RMSNormF32(k[start:start+headDim], layer.Attention.KNormWeight, headScratch, s.Config.RMSNormEps); err != nil {
						return nil, err
					}
					copy(k[start:start+headDim], headScratch)
				}
			}
			if !s.Config.UseAlibi {
				for hi := 0; hi < numHeads; hi++ {
					start := hi * headDim
					if err := tensor.ApplyRopeHeadF32(q[start:start+headDim], headScratch, pos, headDim, layerRope); err != nil {
						return nil, err
					}
					copy(q[start:start+headDim], headScratch)
				}
				for hi := 0; hi < numKV; hi++ {
					start := hi * headDim
					if err := tensor.ApplyRopeHeadF32(k[start:start+headDim], headScratch, pos, headDim, layerRope); err != nil {
						return nil, err
					}
					copy(k[start:start+headDim], headScratch)
				}
			}
		}

		attnPreO := make([]float32, b*qSize)
		cache := &s.KVCache[layerIdx]
		for t := 0; t < b; t++ {
			k := kAll[t*kvLen : (t+1)*kvLen]
			v := vAll[t*kvLen : (t+1)*kvLen]
			cache.Keys = append(cache.Keys, k...)
			cache.Values = append(cache.Values, v...)
			cache.SeqLen++
			q := qAll[t*qSize : (t+1)*qSize]
			out := attnPreO[t*qSize : (t+1)*qSize]
			pos := s.PositionOffset + t
			switch {
			case s.Config.UseAlibi && len(s.AlibiSlopes) > 0:
				if err := flash_attention.FlashAttentionDecodeHeadsGQAAlibi(
					q, cache.Keys, cache.Values, out,
					cache.SeqLen, headDim, kvLen, numHeads, numKV,
					s.AlibiSlopes, pos, layerWindow,
				); err != nil {
					return nil, err
				}
			case layerWindow > 0:
				if err := flash_attention.FlashAttentionDecodeHeadsGQAWindow(
					q, cache.Keys, cache.Values, out,
					cache.SeqLen, headDim, kvLen, numHeads, numKV, layerWindow,
				); err != nil {
					return nil, err
				}
			default:
				if err := flash_attention.FlashAttentionDecodeHeadsGQA(
					q, cache.Keys, cache.Values, out,
					cache.SeqLen, headDim, kvLen, numHeads, numKV,
				); err != nil {
					return nil, err
				}
			}
		}

		attnOutAll := make([]float32, b*h)
		if layer.Attention.OProj.IsLoaded() {
			if err := layer.Attention.OProj.Gemm(attnPreO, attnOutAll, b); err != nil {
				return nil, err
			}
		} else if qSize == h {
			copy(attnOutAll, attnPreO)
		}
		// Gemma sandwich norm: normalize attention output before residual.
		if s.Config.SandwichNorm && len(layer.PostAttentionLayernorm) == h {
			scratch := make([]float32, h)
			for t := 0; t < b; t++ {
				slice := attnOutAll[t*h : (t+1)*h]
				if err := tensor.RMSNormF32(slice, layer.PostAttentionLayernorm, scratch, s.Config.RMSNormEps); err != nil {
					return nil, err
				}
				copy(slice, scratch)
			}
		}
		for i := range hidden {
			hidden[i] += attnOutAll[i]
		}

		mlpNorm := s.mlpNorm(layer)
		normedMLP := make([]float32, b*h)
		for t := 0; t < b; t++ {
			if err := tensor.RMSNormF32(hidden[t*h:(t+1)*h], mlpNorm, normedMLP[t*h:(t+1)*h], s.Config.RMSNormEps); err != nil {
				return nil, err
			}
		}
		inter := s.Config.IntermediateSize
		gate := make([]float32, b*inter)
		up := make([]float32, b*inter)
		if layer.MLPGate.IsLoaded() {
			if err := layer.MLPGate.Gemm(normedMLP, gate, b); err != nil {
				return nil, err
			}
		}
		if layer.MLPUp.IsLoaded() {
			if err := layer.MLPUp.Gemm(normedMLP, up, b); err != nil {
				return nil, err
			}
		}
		applyGLU(gate, up, s.Config.GeluFFN)
		mlpOutAll := make([]float32, b*h)
		if layer.MLPDown.IsLoaded() {
			if err := layer.MLPDown.Gemm(gate, mlpOutAll, b); err != nil {
				return nil, err
			}
		}
		// Gemma sandwich norm: normalize FFN output before residual.
		if s.Config.SandwichNorm && len(layer.PostFFNLayernorm) == h {
			scratch := make([]float32, h)
			for t := 0; t < b; t++ {
				slice := mlpOutAll[t*h : (t+1)*h]
				if err := tensor.RMSNormF32(slice, layer.PostFFNLayernorm, scratch, s.Config.RMSNormEps); err != nil {
					return nil, err
				}
				copy(slice, scratch)
			}
		}
		for i := range hidden {
			hidden[i] += mlpOutAll[i]
		}
	}

	outHidden := make([]float32, b*h)
	for t := 0; t < b; t++ {
		if err := tensor.RMSNormF32(hidden[t*h:(t+1)*h], s.Norm, outHidden[t*h:(t+1)*h], s.Config.RMSNormEps); err != nil {
			return nil, err
		}
	}
	s.PositionOffset += b
	last := outHidden[(b-1)*h : b*h]
	return append([]float32(nil), last...), nil
}

// Logits projects hidden state to vocabulary logits.
func (s *LlamaDecoderStack) Logits(hidden []float32) (Logits, error) {
	if !s.Output.IsLoaded() {
		return nil, fmt.Errorf("decoder stack is missing output projection")
	}
	vocab := s.Output.outputDim()
	inDim := s.Output.inputDim()
	if len(hidden) < inDim {
		return nil, fmt.Errorf("hidden width %d smaller than output input width %d", len(hidden), inDim)
	}
	logits := make(Logits, vocab)
	if err := s.Output.Gemv(hidden[:inDim], logits); err != nil {
		return nil, err
	}
	return logits, nil
}
