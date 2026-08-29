package flash_attention

import "math"

// FlashAttentionDecodeF16 computes single-head decode attention against an
func FlashAttentionDecodeF16(
	query []float32,
	keyLayer, valueLayer []uint16,
	output []float32,
	seqLen, headDim, kvLen, kvHead int,
) error {
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
	if seqLen == 0 {
		for i := range output[:headDim] {
			output[i] = 0
		}
		return nil
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
	for i := range output[:headDim] {
		output[i] = 0
	}
	for t := 0; t < seqLen; t++ {
		row := t*kvLen + kvOff
		key := keyLayer[row : row+headDim]
		score := DotProductF32F16(query[:headDim], key) * scale
		newMax := m
		if score > newMax {
			newMax = score
		}
		alpha := float32(math.Exp(float64(m - newMax)))
		beta := float32(math.Exp(float64(score - newMax)))
		m = newMax
		l = l*alpha + beta
		// out = out*alpha + beta*value (online softmax), value is f16.
		if alpha != 1 {
			for d := 0; d < headDim; d++ {
				output[d] *= alpha
			}
		}
		AxpyF32F16(output[:headDim], beta, valueLayer[row:row+headDim])
	}
	if l > 0 {
		inv := 1 / l
		for d := 0; d < headDim; d++ {
			output[d] *= inv
		}
	}
	return nil
}

// FlashAttentionDecodeHeadsF16 runs grouped-query decode attention over an
// f16 KV cache. KV is shared across query heads via
// kv_head = query_head / (num_heads / kv_heads).
func FlashAttentionDecodeHeadsF16(
	queryHeads []float32,
	keyLayer, valueLayer []uint16,
	output []float32,
	seqLen, headDim, kvLen, numHeads, kvHeads int,
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
		if err := FlashAttentionDecodeF16(q, keyLayer, valueLayer, out, seqLen, headDim, kvLen, kvH); err != nil {
			return err
		}
	}
	return nil
}
