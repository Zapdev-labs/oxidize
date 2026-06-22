// Package flash_attention mirrors oxidize_core::compute::flash_attention. It
// provides blocked implementations of single-query decode attention and
// multi-token prefill attention, plus a runtime-dispatched dot product.
package flash_attention

import (
	"math"
	"sync"

	"github.com/Zapdev-labs/oxidize/golang/core/quantization"
	"github.com/Zapdev-labs/oxidize/golang/core/simd"
)

// Error mirrors FlashAttentionError.
type Error struct{ Message string }

func (e *Error) Error() string { return "flash_attention: " + e.Message }

// DotProductF32 returns the dot product of two float32 slices. The
// implementation dispatches to OXK SIMD kernels when linked, otherwise
// chunked scalar with auto-vectorization hints.
func DotProductF32(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	if quantization.OxkHasAVX2() {
		return quantization.OxkDotF32(a, b)
	}
	width := simd.Preferred().LaneWidthF32()
	if width <= 1 {
		return dotScalar(a, b)
	}
	var sum float32
	for i := 0; i < len(a); i += width {
		end := i + width
		if end > len(a) {
			end = len(a)
		}
		for j := i; j < end; j++ {
			sum += a[j] * b[j]
		}
	}
	return sum
}

func dotScalar(a, b []float32) float32 {
	var sum float32
	for i, v := range a {
		sum += v * b[i]
	}
	return sum
}

// FlashAttentionDecodeF32 computes attention for a single query against
// cached keys/values. The implementation uses online softmax with the
// blocking constant from tensor.FlashAttentionBlockTokens.
func FlashAttentionDecodeF32(query, keyCache, valueCache, output []float32, seqLen, headDim int, scale float32) error {
	if len(query) < headDim {
		return &Error{Message: "query too small"}
	}
	if len(keyCache) < seqLen*headDim {
		return &Error{Message: "key cache too small"}
	}
	if len(valueCache) < seqLen*headDim {
		return &Error{Message: "value cache too small"}
	}
	if len(output) < headDim {
		return &Error{Message: "output too small"}
	}
	for d := range output[:headDim] {
		output[d] = 0
	}
	m := float32(math.Inf(-1))
	var l float32
	for s := 0; s < seqLen; s++ {
		k := keyCache[s*headDim : (s+1)*headDim]
		score := DotProductF32(query[:headDim], k) * scale
		newMax := m
		if score > newMax {
			newMax = score
		}
		alpha := float32(math.Exp(float64(m - newMax)))
		beta := float32(math.Exp(float64(score - newMax)))
		m = newMax
		l = l*alpha + beta
		v := valueCache[s*headDim : (s+1)*headDim]
		for d := 0; d < headDim; d++ {
			output[d] = output[d]*alpha + beta*v[d]
		}
	}
	inv := 1 / l
	for d := 0; d < headDim; d++ {
		output[d] *= inv
	}
	return nil
}

// FlashAttentionDecodeGQA computes single-head decode attention against a
// layer KV cache laid out as [seq_len][kv_len] row-major (oxidize-core layout).
func FlashAttentionDecodeGQA(
	query, keyLayer, valueLayer, output []float32,
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

// FlashAttentionDecodeHeadsGQA runs grouped-query decode attention. KV cache
// is shared across query heads: kv_head = query_head / (num_heads / kv_heads).
func FlashAttentionDecodeHeadsGQA(
	queryHeads, keyLayer, valueLayer, output []float32,
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
		if err := FlashAttentionDecodeGQA(q, keyLayer, valueLayer, out, seqLen, headDim, kvLen, kvH); err != nil {
			return err
		}
	}
	return nil
}

// FlashAttentionDecodeHeadsF32 runs FlashAttentionDecodeF32 in parallel over
// multiple heads.
func FlashAttentionDecodeHeadsF32(queries, keyCache, valueCache, output []float32, headCount, seqLen, headDim int, scale float32) error {
	if len(queries) < headCount*headDim {
		return &Error{Message: "queries too small"}
	}
	if len(keyCache) < headCount*seqLen*headDim {
		return &Error{Message: "key cache too small"}
	}
	if len(valueCache) < headCount*seqLen*headDim {
		return &Error{Message: "value cache too small"}
	}
	if len(output) < headCount*headDim {
		return &Error{Message: "output too small"}
	}
	var wg sync.WaitGroup
	for h := 0; h < headCount; h++ {
		wg.Add(1)
		go func(h int) {
			defer wg.Done()
			q := queries[h*headDim : (h+1)*headDim]
			kc := keyCache[h*seqLen*headDim : (h+1)*seqLen*headDim]
			vc := valueCache[h*seqLen*headDim : (h+1)*seqLen*headDim]
			o := output[h*headDim : (h+1)*headDim]
			_ = FlashAttentionDecodeF32(q, kc, vc, o, seqLen, headDim, scale)
		}(h)
	}
	wg.Wait()
	return nil
}

// FlashAttentionPrefillF32 computes attention for a multi-token query against
// itself (causal mask). The implementation processes the query in blocks of
// FlashAttentionBlockTokens.
func FlashAttentionPrefillF32(queries, keys, values, output []float32, seqLen, headDim int, scale float32) error {
	if len(queries) < seqLen*headDim {
		return &Error{Message: "queries too small"}
	}
	if len(keys) < seqLen*headDim {
		return &Error{Message: "keys too small"}
	}
	if len(values) < seqLen*headDim {
		return &Error{Message: "values too small"}
	}
	if len(output) < seqLen*headDim {
		return &Error{Message: "output too small"}
	}
	const block = 64
	for blockStart := 0; blockStart < seqLen; blockStart += block {
		end := blockStart + block
		if end > seqLen {
			end = seqLen
		}
		for i := blockStart; i < end; i++ {
			q := queries[i*headDim : (i+1)*headDim]
			for d := 0; d < headDim; d++ {
				output[i*headDim+d] = 0
			}
			scores := make([]float32, i+1)
			for j := 0; j <= i; j++ {
				k := keys[j*headDim : (j+1)*headDim]
				scores[j] = DotProductF32(q, k) * scale
			}
			maxScore := float32(math.Inf(-1))
			for _, s := range scores {
				if s > maxScore {
					maxScore = s
				}
			}
			var total float32
			for j, s := range scores {
				scores[j] = float32(math.Exp(float64(s - maxScore)))
				total += scores[j]
			}
			inv := 1 / total
			for j := range scores {
				scores[j] *= inv
			}
			for d := 0; d < headDim; d++ {
				var acc float32
				for j := 0; j <= i; j++ {
					acc += scores[j] * values[j*headDim+d]
				}
				output[i*headDim+d] = acc
			}
		}
	}
	return nil
}
