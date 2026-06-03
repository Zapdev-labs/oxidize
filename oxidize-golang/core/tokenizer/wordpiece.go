package tokenizer

import (
	"strings"
)

// WordPieceTokenizer is a WordPiece tokenizer (BERT family).
type WordPieceTokenizer struct {
	vocab      map[string]uint32
	reverse    map[uint32]string
	special    SpecialTokens
	unkToken   string
	maxInputChars int
}

// NewWordPieceTokenizer constructs a WordPiece tokenizer.
func NewWordPieceTokenizer(pieces []string, special SpecialTokens) *WordPieceTokenizer {
	rev := make(map[uint32]string, len(pieces))
	vocab := make(map[string]uint32, len(pieces))
	for i, p := range pieces {
		vocab[p] = uint32(i)
		rev[uint32(i)] = p
	}
	unk := "[UNK]"
	if id, ok := vocab[unk]; ok {
		special.Unknown = id
	} else {
		unk = pieces[0]
	}
	return &WordPieceTokenizer{
		vocab: vocab, reverse: rev, special: special,
		unkToken: unk, maxInputChars: 100,
	}
}

func (w *WordPieceTokenizer) Name() string { return "wordpiece" }
func (w *WordPieceTokenizer) SpecialTokens() SpecialTokens { return w.special }
func (w *WordPieceTokenizer) VocabSize() int               { return len(w.vocab) }

func (w *WordPieceTokenizer) Encode(text string, opts EncodeOptions) ([]uint32, error) {
	if text == "" {
		return []uint32{}, nil
	}
	words := tokenizeBasic(text)
	out := make([]uint32, 0, len(words))
	for _, word := range words {
		subTokens := w.tokenizeWord(word)
		out = append(out, subTokens...)
	}
	return out, nil
}

func (w *WordPieceTokenizer) tokenizeWord(word string) []uint32 {
	if len(word) > w.maxInputChars {
		return []uint32{w.special.Unknown}
	}
	pieces := make([]string, 0)
	start := 0
	for start < len(word) {
		end := len(word)
		var cur string
		ok := false
		for end > start {
			sub := word[start:end]
			if start > 0 {
				sub = "##" + sub
			}
			if _, found := w.vocab[sub]; found {
				cur = sub
				ok = true
				break
			}
			end--
		}
		if !ok {
			return []uint32{w.special.Unknown}
		}
		pieces = append(pieces, cur)
		start = end
	}
	out := make([]uint32, 0, len(pieces))
	for _, p := range pieces {
		if id, ok := w.vocab[p]; ok {
			out = append(out, id)
		} else {
			out = append(out, w.special.Unknown)
		}
	}
	return out
}

func (w *WordPieceTokenizer) Decode(tokens []uint32) (string, error) {
	var sb strings.Builder
	for _, id := range tokens {
		if id == w.special.CLS || id == w.special.Separator || id == w.special.Pad {
			continue
		}
		piece := w.reverse[id]
		piece = strings.TrimPrefix(piece, "##")
		sb.WriteString(piece)
	}
	return sb.String(), nil
}

func (w *WordPieceTokenizer) DecodeSkipSpecial(tokens []uint32) (string, error) {
	return w.Decode(tokens)
}

func tokenizeBasic(text string) []string {
	var words []string
	current := strings.Builder{}
	for _, r := range text {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			current.WriteRune(r)
		} else {
			if current.Len() > 0 {
				words = append(words, current.String())
				current.Reset()
			}
		}
	}
	if current.Len() > 0 {
		words = append(words, current.String())
	}
	return words
}
