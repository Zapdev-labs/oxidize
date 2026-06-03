package tokenizer

import (
	"strings"
)

// SentencePieceUnigramTokenizer implements SentencePiece's Unigram algorithm
// (greedy longest-match with score-based tiebreak).
type SentencePieceUnigramTokenizer struct {
	pieces  []string
	reverse map[uint32]string
	scores  []float32
	special SpecialTokens
}

// NewSentencePieceTokenizer constructs an SP Unigram tokenizer.
func NewSentencePieceTokenizer(pieces []string, scores []float32, special SpecialTokens) *SentencePieceUnigramTokenizer {
	rev := make(map[uint32]string, len(pieces))
	for i, p := range pieces {
		rev[uint32(i)] = p
	}
	if scores == nil {
		scores = make([]float32, len(pieces))
	}
	return &SentencePieceUnigramTokenizer{pieces: pieces, reverse: rev, scores: scores, special: special}
}

func (s *SentencePieceUnigramTokenizer) Name() string { return "sentencepiece" }

func (s *SentencePieceUnigramTokenizer) SpecialTokens() SpecialTokens { return s.special }
func (s *SentencePieceUnigramTokenizer) VocabSize() int               { return len(s.pieces) }

func (s *SentencePieceUnigramTokenizer) Encode(text string, opts EncodeOptions) ([]uint32, error) {
	text = strings.ReplaceAll(text, " ", "▁")
	if text == "" {
		return []uint32{}, nil
	}
	out := make([]uint32, 0, len(text))
	rest := text
	for len(rest) > 0 {
		best := s.special.Unknown
		bestScore := float32(-1e30)
		bestLen := 0
		// Greedy longest-prefix match; on tie pick the highest score.
		for i, p := range s.pieces {
			if len(p) > len(rest) {
				continue
			}
			if rest[:len(p)] == p {
				if len(p) > bestLen || (len(p) == bestLen && s.scores[i] > bestScore) {
					bestLen = len(p)
					bestScore = s.scores[i]
					best = uint32(i)
				}
			}
		}
		if best == s.special.Unknown && bestLen == 0 {
			// Fall back to a single UTF-8 rune encoded as the unk token.
			_, size := decodeRune([]byte(rest))
			if size <= 0 {
				size = 1
			}
			out = append(out, s.special.Unknown)
			rest = rest[size:]
			continue
		}
		out = append(out, best)
		rest = rest[bestLen:]
	}
	return out, nil
}

func (s *SentencePieceUnigramTokenizer) Decode(tokens []uint32) (string, error) {
	var sb strings.Builder
	for _, id := range tokens {
		if id == s.special.BOS || id == s.special.EOS || id == s.special.Pad {
			continue
		}
		sb.WriteString(s.reverse[id])
	}
	decoded := sb.String()
	// SentencePiece: ▁ (U+2581) represents a space.
	decoded = strings.ReplaceAll(decoded, "▁", " ")
	return decoded, nil
}

func (s *SentencePieceUnigramTokenizer) DecodeSkipSpecial(tokens []uint32) (string, error) {
	return s.Decode(tokens)
}
