package video

import "fmt"

// PromptSegment is one block of a multimodal video prompt.
type PromptSegment struct {
	TextTokens []uint32
	Video      *VideoSegment
}

// VideoSegment holds per-frame embeddings flattened row-major.
type VideoSegment struct {
	Embeddings    []float32
	NumFrames     int
	LLMHiddenSize int
}

// VideoPrompt builds a flattened embedding sequence for video + text inputs.
type VideoPrompt struct {
	Segments              []PromptSegment
	VideoStartEmbedding   []float32
	VideoEndEmbedding     []float32
}

// NewVideoPrompt constructs an empty prompt.
func NewVideoPrompt() *VideoPrompt { return &VideoPrompt{} }

// AddText appends a text token block.
func (p *VideoPrompt) AddText(tokens []uint32) {
	p.Segments = append(p.Segments, PromptSegment{TextTokens: append([]uint32(nil), tokens...)})
}

// AddVideo appends a video embedding block.
func (p *VideoPrompt) AddVideo(embeddings []float32, numFrames, hidden int) {
	p.Segments = append(p.Segments, PromptSegment{
		Video: &VideoSegment{
			Embeddings:    append([]float32(nil), embeddings...),
			NumFrames:     numFrames,
			LLMHiddenSize: hidden,
		},
	})
}

// BuildSequence flattens segments using the token embedding table for text rows.
func (p *VideoPrompt) BuildSequence(table []float32, vocabSize, hiddenSize int) ([]float32, error) {
	llmHidden, err := p.inferHiddenSize(hiddenSize)
	if err != nil {
		return nil, err
	}
	totalRows, err := p.countRows(hiddenSize, llmHidden)
	if err != nil {
		return nil, err
	}
	out := make([]float32, totalRows*llmHidden)
	cursor := 0
	writeRow := func(row []float32) error {
		if len(row) != llmHidden {
			return &Error{Message: fmt.Sprintf("row width %d != %d", len(row), llmHidden)}
		}
		copy(out[cursor:cursor+llmHidden], row)
		cursor += llmHidden
		return nil
	}
	for _, seg := range p.Segments {
		if seg.Video != nil {
			if len(p.VideoStartEmbedding) == llmHidden {
				if err := writeRow(p.VideoStartEmbedding); err != nil {
					return nil, err
				}
			}
			v := seg.Video
			if v.NumFrames*v.LLMHiddenSize != len(v.Embeddings) {
				return nil, &Error{Message: "video embedding length mismatch"}
			}
			for f := 0; f < v.NumFrames; f++ {
				start := f * v.LLMHiddenSize
				if err := writeRow(v.Embeddings[start : start+v.LLMHiddenSize]); err != nil {
					return nil, err
				}
			}
			if len(p.VideoEndEmbedding) == llmHidden {
				if err := writeRow(p.VideoEndEmbedding); err != nil {
					return nil, err
				}
			}
			continue
		}
		for _, tok := range seg.TextTokens {
			if int(tok) >= vocabSize {
				return nil, &Error{Message: fmt.Sprintf("token %d >= vocab %d", tok, vocabSize)}
			}
			start := int(tok) * hiddenSize
			if start+hiddenSize > len(table) {
				return nil, &Error{Message: "embedding table too small"}
			}
			row := table[start : start+hiddenSize]
			if hiddenSize == llmHidden {
				if err := writeRow(row); err != nil {
					return nil, err
				}
				continue
			}
			padded := make([]float32, llmHidden)
			copy(padded, row)
			if err := writeRow(padded); err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}

func (p *VideoPrompt) inferHiddenSize(fallback int) (int, error) {
	for _, seg := range p.Segments {
		if seg.Video != nil && seg.Video.LLMHiddenSize > 0 {
			return seg.Video.LLMHiddenSize, nil
		}
	}
	if fallback <= 0 {
		return 0, &Error{Message: "cannot infer hidden size"}
	}
	return fallback, nil
}

func (p *VideoPrompt) countRows(hiddenSize, llmHidden int) (int, error) {
	rows := 0
	for _, seg := range p.Segments {
		if seg.Video != nil {
			extra := 0
			if len(p.VideoStartEmbedding) == llmHidden {
				extra++
			}
			if len(p.VideoEndEmbedding) == llmHidden {
				extra++
			}
			rows += extra + seg.Video.NumFrames
			continue
		}
		rows += len(seg.TextTokens)
	}
	if rows == 0 {
		return 0, &Error{Message: "empty prompt"}
	}
	_ = hiddenSize
	return rows, nil
}
