package model

import (
	"math"
	"sort"
)

// MoELayer holds mixture-of-experts FFN weights (Mixtral / MiniMax style).
type MoELayer struct {
	GateInp    F32Weight
	GateExps   F32Weight
	UpExps     F32Weight
	DownExps   F32Weight
	NumExperts int
}

// Loaded reports whether MoE tensors are present.
func (m MoELayer) Loaded() bool {
	return m.NumExperts > 0 &&
		m.GateInp.IsLoaded() &&
		m.GateExps.IsLoaded() &&
		m.UpExps.IsLoaded() &&
		m.DownExps.IsLoaded()
}

func moeExpertGemv(w F32Weight, expertIdx, numExperts, outDim, inDim int, input, output []float32) error {
	if !w.IsLoaded() {
		return nil
	}
	if w.Quant != nil {
		return w.Gemv(input, output)
	}
	if w.Cols != inDim || numExperts <= 0 {
		return w.Gemv(input, output)
	}
	rowsPerExpert := w.Rows / numExperts
	if rowsPerExpert <= 0 {
		rowsPerExpert = outDim
	}
	rowOff := expertIdx * rowsPerExpert
	if (rowOff+rowsPerExpert)*w.Cols > len(w.Data) {
		return w.Gemv(input, output)
	}
	sub := F32Weight{Data: w.Data[rowOff*w.Cols : (rowOff+rowsPerExpert)*w.Cols], Rows: rowsPerExpert, Cols: inDim}
	return sub.Gemv(input, output)
}

// moeFFNForward implements single-token MoE FFN (oxidize-core moe_ffn_forward_single).
func moeFFNForward(
	moe MoELayer,
	hiddenSize, intermediateSize, expertsPerTok int,
	normed, ffnOut []float32,
) error {
	nExperts := moe.NumExperts
	if nExperts == 0 {
		return nil
	}
	if expertsPerTok <= 0 {
		expertsPerTok = 1
	}
	if expertsPerTok > nExperts {
		expertsPerTok = nExperts
	}

	router := make([]float32, nExperts)
	if err := moe.GateInp.Gemv(normed, router); err != nil {
		return err
	}
	maxLogit := float32(math.Inf(-1))
	for _, v := range router {
		if v > maxLogit {
			maxLogit = v
		}
	}
	var sumExp float32
	for i := range router {
		router[i] = float32(math.Exp(float64(router[i] - maxLogit)))
		sumExp += router[i]
	}
	if sumExp > 0 {
		for i := range router {
			router[i] /= sumExp
		}
	}

	type scored struct {
		idx   int
		score float32
	}
	ranked := make([]scored, nExperts)
	for i, s := range router {
		ranked[i] = scored{i, s}
	}
	sort.Slice(ranked, func(i, j int) bool { return ranked[i].score > ranked[j].score })

	gate := make([]float32, intermediateSize)
	up := make([]float32, intermediateSize)
	expertOut := make([]float32, hiddenSize)
	for i := range ffnOut {
		ffnOut[i] = 0
	}

	for _, pick := range ranked[:expertsPerTok] {
		ex := pick.idx
		weight := pick.score
		clearF32(gate)
		clearF32(up)
		clearF32(expertOut)

		if err := moeExpertGemv(moe.GateExps, ex, nExperts, intermediateSize, hiddenSize, normed, gate); err != nil {
			return err
		}
		if err := moeExpertGemv(moe.UpExps, ex, nExperts, intermediateSize, hiddenSize, normed, up); err != nil {
			return err
		}
		for j := 0; j < intermediateSize; j++ {
			g := gate[j]
			gate[j] = g * (1 / (1 + float32(math.Exp(float64(-g))))) * up[j]
		}
		if err := moeExpertGemv(moe.DownExps, ex, nExperts, hiddenSize, intermediateSize, gate, expertOut); err != nil {
			return err
		}
		for j := 0; j < hiddenSize; j++ {
			ffnOut[j] += weight * expertOut[j]
		}
	}
	return nil
}

func clearF32(s []float32) {
	for i := range s {
		s[i] = 0
	}
}
