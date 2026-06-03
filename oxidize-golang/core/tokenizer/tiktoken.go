package tokenizer

import (
	"regexp"
	"strings"
)

// TiktokenTokenizer implements a regex-based BPE tokenizer compatible with
// OpenAI's tiktoken cl100k_base / o200k_base encodings.
type TiktokenTokenizer struct {
	pattern *regexp.Regexp
	ranks   map[string]float32
	reverse map[uint32]string
	special SpecialTokens
}

// NewTiktokenTokenizer constructs a Tiktoken tokenizer with a regex pattern
// and a map of merge-pair -> rank.
func NewTiktokenTokenizer(pattern string, ranks map[string]float32, special SpecialTokens) *TiktokenTokenizer {
	if pattern == "" {
		pattern = defaultTiktokenPattern()
	}
	return &TiktokenTokenizer{
		pattern: regexp.MustCompile(pattern),
		ranks:   ranks,
		reverse: map[uint32]string{},
		special: special,
	}
}

func (t *TiktokenTokenizer) Name() string { return "tiktoken" }
func (t *TiktokenTokenizer) SpecialTokens() SpecialTokens { return t.special }
func (t *TiktokenTokenizer) VocabSize() int               { return len(t.ranks) }

func (t *TiktokenTokenizer) Encode(text string, opts EncodeOptions) ([]uint32, error) {
	if text == "" {
		return []uint32{}, nil
	}
	chunks := t.pattern.FindAllString(text, -1)
	out := make([]uint32, 0, len(chunks))
	for _, chunk := range chunks {
		ids := t.bpe([]byte(chunk))
		out = append(out, ids...)
	}
	return out, nil
}

func (t *TiktokenTokenizer) bpe(word []byte) []uint32 {
	if len(word) == 1 {
		return []uint32{t.lookup(string(word))}
	}
	parts := []string{string(word)}
	for {
		pairs := make([][2]string, 0, len(parts)-1)
		for i := 0; i < len(parts)-1; i++ {
			pairs = append(pairs, [2]string{parts[i], parts[i+1]})
		}
		bestIdx := -1
		bestRank := float32(1e30)
		for i, p := range pairs {
			if rank, ok := t.ranks[p[0]+p[1]]; ok {
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
		out = append(out, t.lookup(p))
	}
	return out
}

func (t *TiktokenTokenizer) lookup(p string) uint32 {
	if t.ranks == nil {
		return 0
	}
	if rank, ok := t.ranks[p]; ok {
		return uint32(rank)
	}
	return 0
}

func (t *TiktokenTokenizer) Decode(tokens []uint32) (string, error) {
	var sb strings.Builder
	for _, id := range tokens {
		if id == t.special.BOS || id == t.special.EOS || id == t.special.Pad {
			continue
		}
		if piece, ok := t.reverse[id]; ok {
			sb.WriteString(piece)
		} else {
			sb.WriteRune(rune(id))
		}
	}
	return sb.String(), nil
}

func (t *TiktokenTokenizer) DecodeSkipSpecial(tokens []uint32) (string, error) {
	return t.Decode(tokens)
}
