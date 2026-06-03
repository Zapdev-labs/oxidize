package tokenizer

import (
	"strings"
	"testing"
)

func TestBpeRoundtrip(t *testing.T) {
	tok := NewBpeTokenizer(
		[]string{"<unk>", "<s>", "</s>", "a", "b", "c", "ab", "abc", "▁"},
		[]string{"ab", "abc"},
		SpecialTokens{Unknown: 0, BOS: 1, EOS: 2, Pad: 0},
	)
	tokens, err := tok.Encode("abc", EncodeOptions{AddBOS: false})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if len(tokens) == 0 {
		t.Fatal("empty tokens")
	}
	decoded, err := tok.Decode(tokens)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !strings.Contains(decoded, "abc") {
		t.Fatalf("decoded = %q", decoded)
	}
}

func TestSentencePieceEncode(t *testing.T) {
	tok := NewSentencePieceTokenizer(
		[]string{"<unk>", "▁a", "▁ab", "a", "b"},
		[]float32{0, -1, -2, -2, -2},
		SpecialTokens{Unknown: 0, BOS: 0, EOS: 0, Pad: 0},
	)
	tokens, err := tok.Encode("ab", EncodeOptions{})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if len(tokens) == 0 {
		t.Fatal("empty tokens")
	}
	_, err = tok.Decode(tokens)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
}

func TestWordPieceEncode(t *testing.T) {
	tok := NewWordPieceTokenizer(
		[]string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world", "##s", "##ed"},
		SpecialTokens{Pad: 0, Unknown: 1, CLS: 2, Separator: 3},
	)
	tokens, err := tok.Encode("hello worlds", EncodeOptions{})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if len(tokens) < 2 {
		t.Fatalf("tokens = %v", tokens)
	}
}

func TestChatTemplateBasic(t *testing.T) {
	template := "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}"
	rendered := ProcessChatTemplate(template, []ChatMessage{
		{Role: "user", Content: "hi"},
	}, true)
	if !strings.Contains(rendered, "user: hi") {
		t.Fatalf("rendered = %q", rendered)
	}
	if !strings.Contains(rendered, "assistant:") {
		t.Fatalf("missing gen prompt: %q", rendered)
	}
}

func TestFromGGUFMetadata(t *testing.T) {
	tok, err := FromGGUFMetadata(map[string]string{
		"tokenizer.ggml.model":   "bpe",
		"tokenizer.ggml.bos_token_id": "1",
		"tokenizer.ggml.eos_token_id": "2",
	})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if tok.Name() != "bpe" {
		t.Fatalf("name = %q", tok.Name())
	}
}
