package model

import (
	"fmt"

	"github.com/Zapdev-labs/oxidize/golang/core/tensor"
)

// CacheTargetHidden appends fused target hidden states for speculative decode.
func (m *DFlashDraftModel) CacheTargetHidden(hidden []float32) error {
	want := m.Config.TargetHiddenWidth()
	if len(hidden) != want {
		return fmt.Errorf("target hidden width mismatch: expected %d, actual %d", want, len(hidden))
	}
	m.TargetHiddenCache = append(m.TargetHiddenCache, append([]float32(nil), hidden...))
	return nil
}

// ClearSpeculativeCaches resets KV, target hidden cache, and position.
func (m *DFlashDraftModel) ClearSpeculativeCaches() {
	if m.Stack != nil {
		m.Stack.ResetCache()
	}
	m.TargetHiddenCache = m.TargetHiddenCache[:0]
}

func (m *DFlashDraftModel) dflashTargetContext(targetHidden []float32) ([]float32, error) {
	h := m.Config.HiddenSize
	if len(targetHidden) == 0 || !m.FC.IsLoaded() {
		return nil, nil
	}
	fused := make([]float32, h)
	if err := m.FC.Gemv(targetHidden, fused); err != nil {
		return nil, err
	}
	for i := 0; i < h && i < len(m.FCBias); i++ {
		fused[i] += m.FCBias[i]
	}
	if len(m.HiddenNorm) == 0 {
		return fused, nil
	}
	ctx := make([]float32, h)
	if err := tensor.RMSNormF32(fused, m.HiddenNorm, ctx, m.Config.RMSNormEps); err != nil {
		return nil, err
	}
	return ctx, nil
}

// ForwardToken runs one token with optional target_hidden fusion.
func (m *DFlashDraftModel) ForwardToken(token uint32, targetHidden []float32) ([]float32, error) {
	if m.Stack == nil {
		return nil, fmt.Errorf("DFlash draft model has no decoder stack")
	}
	targetContext, err := m.dflashTargetContext(targetHidden)
	if err != nil {
		return nil, err
	}
	kvContext := func(layerIdx int, ctx []float32) (kCtx, vCtx []float32, err error) {
		if ctx == nil {
			return nil, nil, nil
		}
		layer := &m.Stack.Layers[layerIdx]
		kvLen := m.Config.NumKeyValueHeads * m.Config.HeadDim()
		if !layer.Attention.KProj.IsLoaded() || !layer.Attention.VProj.IsLoaded() ||
			layer.Attention.KProj.inputDim() != len(ctx) ||
			layer.Attention.VProj.inputDim() != len(ctx) {
			return nil, nil, nil
		}
		kCtx = make([]float32, kvLen)
		vCtx = make([]float32, kvLen)
		if err := layer.Attention.KProj.Gemv(ctx, kCtx); err != nil {
			return nil, nil, err
		}
		if err := layer.Attention.VProj.Gemv(ctx, vCtx); err != nil {
			return nil, nil, err
		}
		return kCtx, vCtx, nil
	}
	return m.Stack.forwardTokenWithContext(token, targetContext, kvContext)
}

// ForwardBatch prefills multiple tokens; returns the last token hidden state.
func (m *DFlashDraftModel) ForwardBatch(tokens []uint32) ([]float32, error) {
	if m.Stack == nil {
		return nil, fmt.Errorf("DFlash draft model has no decoder stack")
	}
	if len(tokens) == 0 {
		return nil, fmt.Errorf("empty token batch")
	}
	if len(tokens) == 1 {
		return m.ForwardToken(tokens[0], nil)
	}
	if len(m.HiddenNorm) > 0 {
		var last []float32
		var err error
		for _, tok := range tokens {
			last, err = m.ForwardToken(tok, nil)
			if err != nil {
				return nil, err
			}
		}
		return last, nil
	}
	return m.Stack.ForwardBatch(tokens)
}

// Logits projects hidden state to vocabulary logits.
func (m *DFlashDraftModel) Logits(hidden []float32) (Logits, error) {
	if m.Stack == nil {
		return nil, fmt.Errorf("DFlash draft model has no decoder stack")
	}
	return m.Stack.Logits(hidden)
}

// ResetCache clears KV cache and position.
func (m *DFlashDraftModel) ResetCache() {
	if m.Stack != nil {
		m.Stack.ResetCache()
	}
	m.TargetHiddenCache = m.TargetHiddenCache[:0]
}

// ReserveCacheTokens pre-allocates KV cache capacity.
func (m *DFlashDraftModel) ReserveCacheTokens(tokens int) {
	if m.Stack != nil {
		m.Stack.ReserveCacheTokens(tokens)
	}
}
