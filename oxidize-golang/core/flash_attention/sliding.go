package flash_attention

import "math"

// FlashAttentionDecodeGQAWindow is like FlashAttentionDecodeGQA but only attends
// to keys in [seqLen-windowSize, seqLen). windowSize <= 0 means full attention.
func FlashAttentionDecodeGQAWindow(
	query, keyLayer, valueLayer, output []float32,
	seqLen, headDim, kvLen, kvHead int,
	windowSize int,
) error {
	if windowSize <= 0 || windowSize >= seqLen {
		return FlashAttentionDecodeGQA(query, keyLayer, valueLayer, output, seqLen, headDim, kvLen, kvHead)
	}
	start := seqLen - windowSize
	if len(query) < headDim {
		return &Error{Message: "query too small"}
	}
	expected := seqLen * kvLen
	if len(keyLayer) < expected || len(valueLayer) < expected {
		return &Error{Message: "kv cache too small"}
	}
	if len(output) < headDim {
		return &Error{Message: "output too small"}
	}
	if headDim == 0 || kvLen%headDim != 0 {
		return &Error{Message: "invalid kv_len for head_dim"}
	}
	kvHeads := kvLen / headDim
	if kvHead < 0 || kvHead >= kvHeads {
		return &Error{Message: "invalid kv_head"}
	}
	scale := float32(1) / float32(math.Sqrt(float64(headDim)))
	kvOff := kvHead * headDim
	m := float32(math.Inf(-1))
	var l float32
	for d := range output[:headDim] {
		output[d] = 0
	}
	for t := start; t < seqLen; t++ {
		row := t*kvLen + kvOff
		key := keyLayer[row : row+headDim]
		score := DotProductF32(query[:headDim], key) * scale
		newMax := m
		if score > newMax {
			newMax = score
		}
		alpha := float32(math.Exp(float64(m - newMax)))
		beta := float32(math.Exp(float64(score - newMax)))
		m = newMax
		l = l*alpha + beta
		val := valueLayer[row : row+headDim]
		for d := 0; d < headDim; d++ {
			output[d] = output[d]*alpha + beta*val[d]
		}
	}
	if l > 0 {
		inv := 1 / l
		for d := 0; d < headDim; d++ {
			output[d] *= inv
		}
	}
	return nil
}

// FlashAttentionDecodeHeadsGQAWindow runs grouped-query decode with an optional window.
func FlashAttentionDecodeHeadsGQAWindow(
	queryHeads, keyLayer, valueLayer, output []float32,
	seqLen, headDim, kvLen, numHeads, kvHeads int,
	windowSize int,
) error {
	qLen := numHeads * headDim
	if len(queryHeads) < qLen || len(output) < qLen {
		return &Error{Message: "query/output too small"}
	}
	if kvHeads <= 0 || numHeads%kvHeads != 0 {
		return &Error{Message: "invalid head grouping"}
	}
	group := numHeads / kvHeads
	for h := 0; h < numHeads; h++ {
		kvH := h / group
		q := queryHeads[h*headDim : (h+1)*headDim]
		out := output[h*headDim : (h+1)*headDim]
		if err := FlashAttentionDecodeGQAWindow(q, keyLayer, valueLayer, out, seqLen, headDim, kvLen, kvH, windowSize); err != nil {
			return err
		}
	}
	return nil
}
