package tokenizer

import (
	"sort"
	"strings"
)

// BpeTokenizer is a byte-pair encoding tokenizer (GPT-2 / Llama BPE).
type BpeTokenizer struct {
	vocab   map[string]uint32
	reverse map[uint32]string
	merges  []string
	scores  map[string]float32
	special SpecialTokens
}

// NewBpeTokenizer constructs a BPE tokenizer from explicit pieces and merge
// rules.
func NewBpeTokenizer(pieces []string, merges []string, special SpecialTokens) *BpeTokenizer {
	rev := make(map[uint32]string, len(pieces))
	vocab := make(map[string]uint32, len(pieces))
	for i, p := range pieces {
		vocab[p] = uint32(i)
		rev[uint32(i)] = p
	}
	return &BpeTokenizer{vocab: vocab, reverse: rev, merges: merges, scores: map[string]float32{}, special: special}
}

func (b *BpeTokenizer) Name() string { return "bpe" }

func (b *BpeTokenizer) SpecialTokens() SpecialTokens { return b.special }
func (b *BpeTokenizer) VocabSize() int               { return len(b.vocab) }

func (b *BpeTokenizer) Encode(text string, opts EncodeOptions) ([]uint32, error) {
	if text == "" {
		return []uint32{}, nil
	}
	words := splitOnWhitespace(text)
	out := make([]uint32, 0, len(words))
	for _, w := range words {
		runes := []byte("▁" + strings.TrimPrefix(w, " "))
		tokens := b.bpe(runes)
		out = append(out, tokens...)
	}
	return out, nil
}

func (b *BpeTokenizer) bpe(word []byte) []uint32 {
	if len(word) == 1 {
		if id, ok := b.vocab[string(word)]; ok {
			return []uint32{id}
		}
		return []uint32{b.special.Unknown}
	}
	parts := []string{string(word)}
	for {
		pairs := make([][2]string, 0, len(parts)-1)
		for i := 0; i < len(parts)-1; i++ {
			pairs = append(pairs, [2]string{parts[i], parts[i+1]})
		}
		bestIdx := -1
		bestRank := -1
		for i, p := range pairs {
			if rank, ok := b.scoreFor(p[0] + p[1]); ok {
				if bestIdx < 0 || rank < bestRank {
					bestIdx = i
					bestRank = rank
				}
			}
		}
		if bestIdx < 0 {
			break
		}
		merged := make([]string, 0, len(parts)-1)
		merged = append(merged, parts[:bestIdx]...)
		merged = append(merged, parts[bestIdx]+parts[bestIdx+1])
		merged = append(merged, parts[bestIdx+2:]...)
		parts = merged
	}
	out := make([]uint32, 0, len(parts))
	for _, p := range parts {
		if id, ok := b.vocab[p]; ok {
			out = append(out, id)
		} else {
			out = append(out, b.special.Unknown)
		}
	}
	return out
}

func (b *BpeTokenizer) scoreFor(merge string) (int, bool) {
	for i, m := range b.merges {
		if m == merge {
			return i, true
		}
	}
	if rank, ok := b.scores[merge]; ok {
		return int(rank), true
	}
	return 0, false
}

func (b *BpeTokenizer) Decode(tokens []uint32) (string, error) {
	var sb strings.Builder
	for _, id := range tokens {
		if id == b.special.BOS || id == b.special.EOS || id == b.special.Pad {
			continue
		}
		sb.WriteString(b.reverse[id])
	}
	return strings.ReplaceAll(sb.String(), "▁", " "), nil
}

func (b *BpeTokenizer) DecodeSkipSpecial(tokens []uint32) (string, error) {
	return b.Decode(tokens)
}

func splitOnWhitespace(text string) []string {
	return strings.Fields(text)
}

// sortImports is unused; we just keep the import set stable.
var _ = sort.Stable
