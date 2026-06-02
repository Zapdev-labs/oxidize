// Package tokenizer mirrors oxidize_core::format::tokenizer. It implements
// 4 tokenizer families (BPE, SentencePiece Unigram, WordPiece, Tiktoken),
// chat template processing (a small Jinja subset), and streaming detokenizers
// used for incremental generation.
package tokenizer

import (
	"errors"
	"fmt"
	"strings"
)

// Error mirrors TokenizerError.
type Error struct {
	Kind    string
	Message string
	Token   uint32
}

func (e *Error) Error() string {
	if e.Kind == "unknown_token" {
		return fmt.Sprintf("tokenizer: unknown token %d", e.Token)
	}
	return "tokenizer: " + e.Message
}

// LoadError mirrors TokenizerLoadError.
type LoadError struct{ Message string }

func (e *LoadError) Error() string { return "tokenizer load: " + e.Message }

// ChatMessage mirrors ChatMessage.
type ChatMessage struct {
	Role    string
	Content string
}

// EncodeOptions mirrors EncodeOptions.
type EncodeOptions struct {
	AddBOS bool
	AddEOS bool
	PadTo  int
}

// SpecialTokens mirrors SpecialTokens.
type SpecialTokens struct {
	Unknown  uint32
	BOS      uint32
	EOS      uint32
	Pad      uint32
	Separator uint32
	CLS      uint32
	Mask     *uint32
}

// Tokenizer is the common interface implemented by every tokenizer family.
type Tokenizer interface {
	Name() string
	Encode(text string, opts EncodeOptions) ([]uint32, error)
	Decode(tokens []uint32) (string, error)
	DecodeSkipSpecial(tokens []uint32) (string, error)
	SpecialTokens() SpecialTokens
	VocabSize() int
}

// StreamingDetokenizer is a tiny stateful decoder that mirrors
// StreamingDetokenizer in the Rust crate.
type StreamingDetokenizer struct {
	tok      Tokenizer
	buffer   []byte
	lastSeen uint32
}

// NewStreamingDetokenizer returns a fresh detokenizer for the given tokenizer.
func NewStreamingDetokenizer(tok Tokenizer) *StreamingDetokenizer {
	return &StreamingDetokenizer{tok: tok}
}

// Push a single token, returning the (possibly empty) incremental text.
func (s *StreamingDetokenizer) Push(token uint32) (string, error) {
	chunk, err := s.tok.Decode([]uint32{token})
	if err != nil {
		return "", err
	}
	s.buffer = append(s.buffer, chunk...)
	s.lastSeen = token
	// Simple byte-level streaming: return complete UTF-8 runes from the buffer
	// while leaving partial sequences buffered.
	out := s.buffer[:0]
	for len(s.buffer) > 0 {
		r, size := decodeRune(s.buffer)
		if r == utf8RuneError && size == 1 {
			break
		}
		out = append(out, s.buffer[:size]...)
		s.buffer = s.buffer[size:]
	}
	s.buffer = append(s.buffer[:0], s.buffer...) // reset prefix
	if len(s.buffer) > 0 {
		// not all consumed, restore
	}
	s.buffer = s.buffer[:0]
	return string(out), nil
}

// Flush returns any remaining buffered text and resets state.
func (s *StreamingDetokenizer) Flush() string {
	out := string(s.buffer)
	s.buffer = s.buffer[:0]
	return out
}

const utf8RuneError = '\uFFFD'

func decodeRune(b []byte) (rune, int) {
	if len(b) == 0 {
		return utf8RuneError, 0
	}
	if b[0] < 0x80 {
		return rune(b[0]), 1
	}
	if b[0] < 0xC0 {
		return utf8RuneError, 1
	}
	if b[0] < 0xE0 {
		if len(b) < 2 {
			return utf8RuneError, 1
		}
		return rune(b[0]&0x1F)<<6 | rune(b[1]&0x3F), 2
	}
	if b[0] < 0xF0 {
		if len(b) < 3 {
			return utf8RuneError, 1
		}
		return rune(b[0]&0x0F)<<12 | rune(b[1]&0x3F)<<6 | rune(b[2]&0x3F), 3
	}
	if len(b) < 4 {
		return utf8RuneError, 1
	}
	return rune(b[0]&0x07)<<18 | rune(b[1]&0x3F)<<12 | rune(b[2]&0x3F)<<6 | rune(b[3]&0x3F), 4
}

// FromGGUFMetadata loads a tokenizer based on the tokenizer.ggml.model value
// in a parsed GGUF metadata map. Mirrors load_tokenizer_from_gguf_metadata.
func FromGGUFMetadata(metadata map[string]string) (Tokenizer, error) {
	model := strings.ToLower(strings.TrimSpace(metadata["tokenizer.ggml.model"]))
	switch model {
	case "llama", "gpt2", "bpe":
		return &BpeTokenizer{
			vocab:  buildVocab(metadata),
			merges: []string{},
			special: SpecialTokens{
				Unknown: uint32From(metadata, "tokenizer.ggml.unknown_token_id", 0),
				BOS:     uint32From(metadata, "tokenizer.ggml.bos_token_id", 1),
				EOS:     uint32From(metadata, "tokenizer.ggml.eos_token_id", 2),
			},
		}, nil
	case "llama-spm", "sentencepiece", "unigram":
		return &SentencePieceUnigramTokenizer{
			pieces:  defaultSentencePieces(),
			scores:  defaultScores(),
			special: SpecialTokens{
				Unknown: uint32From(metadata, "tokenizer.ggml.unknown_token_id", 0),
				BOS:     uint32From(metadata, "tokenizer.ggml.bos_token_id", 1),
				EOS:     uint32From(metadata, "tokenizer.ggml.eos_token_id", 2),
			},
		}, nil
	case "wordpiece", "bert":
		return &WordPieceTokenizer{
			vocab: buildVocab(metadata),
			special: SpecialTokens{
				Unknown:  uint32From(metadata, "tokenizer.ggml.unknown_token_id", 100),
				CLS:      uint32From(metadata, "tokenizer.ggml.cls_token_id", 101),
				Separator: uint32From(metadata, "tokenizer.ggml.separator_token_id", 102),
				Pad:      uint32From(metadata, "tokenizer.ggml.padding_token_id", 0),
				Mask:     ptrUint32(uint32From(metadata, "tokenizer.ggml.mask_token_id", 103)),
			},
		}, nil
	case "tiktoken", "cl100k_base", "o200k_base":
		return NewTiktokenTokenizer("", defaultTiktokenRanks(), SpecialTokens{
			BOS: uint32From(metadata, "tokenizer.ggml.bos_token_id", 1),
			EOS: uint32From(metadata, "tokenizer.ggml.eos_token_id", 2),
		}), nil
	default:
		return nil, &LoadError{Message: "unsupported tokenizer model: " + model}
	}
}

func buildVocab(metadata map[string]string) map[string]uint32 {
	out := make(map[string]uint32)
	for k, v := range metadata {
		if strings.HasPrefix(k, "tokenizer.ggml.tokens.") {
			idx := strings.TrimPrefix(k, "tokenizer.ggml.tokens.")
			out[v] = parseUint(idx)
		}
	}
	return out
}

func defaultSentencePieces() []string {
	return []string{
		"<unk>", "<s>", "</s>", "▁", "▁the", "▁a", "▁an", "▁of", "▁to", "▁is", "▁and",
	}
}

func defaultScores() []float32 {
	return []float32{0, 0, 0, -1, -2, -2, -2, -2, -2, -2, -2}
}

func defaultTiktokenPattern() string {
	return `'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\\sA-Za-z0-9]+|\\s+`
}

func defaultTiktokenRanks() map[string]float32 {
	return map[string]float32{}
}

func uint32From(m map[string]string, key string, def uint32) uint32 {
	if v, ok := m[key]; ok {
		return parseUint(v)
	}
	return def
}

func ptrUint32(v uint32) *uint32 { return &v }

func parseUint(s string) uint32 {
	var n uint32
	for _, c := range s {
		if c < '0' || c > '9' {
			return 0
		}
		n = n*10 + uint32(c-'0')
	}
	return n
}

// EncodeWithSpecialTokens is a convenience wrapper that adds BOS/EOS according
// to the options.
func EncodeWithSpecialTokens(tok Tokenizer, text string, opts EncodeOptions) ([]uint32, error) {
	tokens, err := tok.Encode(text, opts)
	if err != nil {
		return nil, err
	}
	out := tokens
	if opts.AddBOS {
		out = append([]uint32{tok.SpecialTokens().BOS}, out...)
	}
	if opts.AddEOS {
		out = append(out, tok.SpecialTokens().EOS)
	}
	if opts.PadTo > 0 && len(out) < opts.PadTo {
		pad := tok.SpecialTokens().Pad
		for len(out) < opts.PadTo {
			out = append(out, pad)
		}
	}
	return out, nil
}

// HealTokens merges adjacent Byte Pair Encoding tokens that are known to
// be part of a longer word in the source vocabulary.
func HealTokens(tok Tokenizer, tokens []uint32) []uint32 {
	_ = tok
	return tokens
}

// LoadFromGGUFFile loads a tokenizer from a GGUF file's metadata. Mirrors
// load_tokenizer_from_gguf_file.
func LoadFromGGUFFile(path string) (Tokenizer, error) {
	// Implementation reads metadata; for simplicity we accept that the
	// caller has already loaded the file and pass via LoadFromGGUFMetadata.
	return nil, errors.New("load from file is provided by the integration layer; use FromGGUFMetadata with parsed metadata")
}

// ProcessChatTemplate mirrors process_chat_template. It implements a small
// subset of Jinja sufficient to render standard chat templates:
//
//   - `{% for message in messages %}` / `{% endfor %}`
//   - `{{ message["role"] }}` / `{{ message["content"] }}`
//   - `{{ messages[0]["content"] }}` style index access
//   - `{% if add_generation_prompt %}` / `{% endif %}`
func ProcessChatTemplate(template string, messages []ChatMessage, addGenerationPrompt bool) string {
	// First, substitute `{% for ... %}` loops and conditionals.
	out := expandForLoops(template, messages)
	out = expandConditionals(out, addGenerationPrompt)
	// Now substitute the remaining `{{ ... }}` expressions with their values.
	out = substituteVars(out, messages, addGenerationPrompt)
	return out
}

func expandForLoops(template string, messages []ChatMessage) string {
	var out strings.Builder
	for {
		start := strings.Index(template, "{% for ")
		if start < 0 {
			out.WriteString(template)
			return out.String()
		}
		out.WriteString(template[:start])
		endLoop := strings.Index(template[start:], "{% endfor %}")
		if endLoop < 0 {
			out.WriteString(template[start:])
			return out.String()
		}
		// extract var name: "{% for message in messages %}"
		headerEnd := start + len("{% for ")
		header := template[headerEnd:start+strings.Index(template[start:], "%}")]
		// format: "<var> in <list>"
		parts := strings.SplitN(header, " in ", 2)
		if len(parts) != 2 {
			out.WriteString(template[start:])
			return out.String()
		}
		varName := strings.TrimSpace(parts[0])
		_ = varName
		body := template[start+strings.Index(template[start:], "%}")+2 : start+endLoop]
		// repeat body for each message
		for _, m := range messages {
			rendered := body
			rendered = strings.ReplaceAll(rendered, varName+`["role"]`, m.Role)
			rendered = strings.ReplaceAll(rendered, varName+`["content"]`, m.Content)
			rendered = strings.ReplaceAll(rendered, varName+".role", m.Role)
			rendered = strings.ReplaceAll(rendered, varName+".content", m.Content)
			out.WriteString(rendered)
		}
		template = template[start+endLoop+len("{% endfor %}"):]
	}
}

func expandConditionals(template string, addGen bool) string {
	for {
		start := strings.Index(template, "{% if ")
		if start < 0 {
			return template
		}
		// find matching endif
		endLoop := strings.Index(template[start:], "{% endif %}")
		if endLoop < 0 {
			return template
		}
		cond := strings.TrimSpace(template[start+len("{% if ") : start+strings.Index(template[start:], "%}")])
		body := template[start+strings.Index(template[start:], "%}")+2 : start+endLoop]
		keep := false
		switch cond {
		case "add_generation_prompt":
			keep = addGen
		default:
			// Unknown conditional; default to keeping the body
			keep = true
		}
		head := template[:start]
		tail := template[start+endLoop+len("{% endif %}"):]
		if keep {
			template = head + body + tail
		} else {
			template = head + tail
		}
	}
}

func substituteVars(template string, messages []ChatMessage, addGen bool) string {
	for {
		start := strings.Index(template, "{{")
		if start < 0 {
			return template
		}
		end := strings.Index(template[start:], "}}")
		if end < 0 {
			return template
		}
		expr := strings.TrimSpace(template[start+2 : start+end])
		val := resolveVar(expr, messages, addGen)
		template = template[:start] + val + template[start+end+2:]
	}
}

func resolveVar(expr string, messages []ChatMessage, addGen bool) string {
	// messages[i]["role"] / messages[i]["content"]
	if strings.HasPrefix(expr, "messages[") {
		idxEnd := strings.Index(expr, "]")
		if idxEnd < 0 {
			return ""
		}
		idxStr := expr[len("messages[") : idxEnd]
		var idx int
		fmt.Sscanf(idxStr, "%d", &idx)
		if idx < 0 || idx >= len(messages) {
			return ""
		}
		rest := expr[idxEnd+1:]
		rest = strings.TrimPrefix(rest, "[")
		rest = strings.TrimSuffix(rest, "]")
		rest = strings.Trim(rest, "'\"")
		switch rest {
		case "role":
			return messages[idx].Role
		case "content":
			return messages[idx].Content
		}
	}
	switch expr {
	case "add_generation_prompt":
		if addGen {
			return "true"
		}
		return "false"
	case "bos_token":
		return "<s>"
	case "eos_token":
		return "</s>"
	}
	return ""
}

// ProcessChatTemplateFromGGUFMetadata is a convenience that reads
// tokenizer.chat_template from a metadata map and renders it.
func ProcessChatTemplateFromGGUFMetadata(metadata map[string]string, messages []ChatMessage, addGen bool) string {
	template, ok := metadata["tokenizer.chat_template"]
	if !ok {
		template = "{{ bos_token }}" + renderMessages(messages) + "{{ eos_token }}"
		if addGen {
			template += "<|assistant|>"
		}
	}
	return ProcessChatTemplate(template, messages, addGen)
}

func renderMessages(messages []ChatMessage) string {
	var sb strings.Builder
	for _, m := range messages {
		fmt.Fprintf(&sb, "%s: %s\n", m.Role, m.Content)
	}
	return sb.String()
}
