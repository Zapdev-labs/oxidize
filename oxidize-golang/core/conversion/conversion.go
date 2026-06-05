// Package conversion mirrors oxidize_core::format::conversion. It detects a
// model's architecture from GGUF metadata, renames HF-style tensor names to
// the canonical GGUF/Llama.cpp naming, and assembles conversion plans.
package conversion

import (
	"strconv"
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/core/ggufcore"
	"github.com/Zapdev-labs/oxidize/golang/core/quantization"
)

// Architecture mirrors ModelArchitecture.
type Architecture string

const (
	ArchLlama   Architecture = "llama"
	ArchMistral Architecture = "mistral"
	ArchQwen    Architecture = "qwen"
	ArchGemma   Architecture = "gemma"
	ArchPhi     Architecture = "phi"
	ArchUnknown Architecture = "unknown"
)

// DetectArchitecture inspects the metadata map and returns the architecture
// string used by Llama.cpp.
func DetectArchitecture(metadata map[string]ggufcore.MetadataValue) Architecture {
	if v, ok := metadata["general.architecture"]; ok {
		switch strings.ToLower(v.String) {
		case "llama":
			return ArchLlama
		case "mistral":
			return ArchMistral
		case "qwen", "qwen2":
			return ArchQwen
		case "gemma":
			return ArchGemma
		case "phi", "phi2":
			return ArchPhi
		}
	}
	return ArchUnknown
}

// MapHFTensorName converts a HuggingFace-style tensor name to the canonical
// llama.cpp / GGUF naming. The transformation covers:
//   - `model.layers.N.*` -> `blk.N.*`
//   - `model.embed_tokens.weight` -> `token_embd.weight`
//   - `model.norm.weight` -> `output_norm.weight`
//   - `lm_head.weight` -> `output.weight`
//   - `model.layers.N.self_attn.*` -> `blk.N.attn_*`
//   - `model.layers.N.mlp.*` -> `blk.N.ffn_*`
//   - MoE experts: `model.layers.N.mlp.experts.*` -> `blk.N.ffn_*_exps.*`
func MapHFTensorName(name string) string {
	if !strings.HasPrefix(name, "model.") && !strings.HasPrefix(name, "lm_head") {
		return name
	}
	parts := strings.Split(name, ".")
	if parts[0] == "lm_head" {
		if len(parts) >= 2 {
			return "output." + strings.Join(parts[1:], ".")
		}
		return "output.weight"
	}
	// model.* path
	if len(parts) < 2 {
		return name
	}
	switch parts[1] {
	case "embed_tokens":
		if len(parts) >= 3 {
			return "token_embd." + strings.Join(parts[2:], ".")
		}
		return "token_embd.weight"
	case "norm":
		return "output_norm.weight"
	case "layers":
		if len(parts) < 4 {
			return name
		}
		layer := parts[2]
		rest := parts[3:]
		switch rest[0] {
		case "self_attn":
			if len(rest) < 2 {
				return "blk." + layer + ".attn"
			}
			child := rest[1]
			// Preserve the trailing ".weight"/".bias" so attention biases
			// (present in Qwen2 etc.) map to attn_*.bias instead of colliding
			// with the weight tensor. Dropping/mis-naming them silently breaks
			// attention and yields fluent-but-incoherent output.
			suffix := rest[len(rest)-1]
			switch child {
			case "q_proj":
				return "blk." + layer + ".attn_q." + suffix
			case "k_proj":
				return "blk." + layer + ".attn_k." + suffix
			case "v_proj":
				return "blk." + layer + ".attn_v." + suffix
			case "o_proj":
				return "blk." + layer + ".attn_output." + suffix
			case "q_norm":
				return "blk." + layer + ".attn_q_norm.weight"
			case "k_norm":
				return "blk." + layer + ".attn_k_norm.weight"
			}
			return "blk." + layer + ".attn_" + strings.Join(rest[1:], "_")
		case "mlp":
			if len(rest) < 2 {
				return "blk." + layer + ".ffn"
			}
			child := rest[1]
			suffix := rest[len(rest)-1]
			switch child {
			case "gate_proj":
				return "blk." + layer + ".ffn_gate." + suffix
			case "up_proj":
				return "blk." + layer + ".ffn_up." + suffix
			case "down_proj":
				return "blk." + layer + ".ffn_down." + suffix
			case "experts":
				return "blk." + layer + ".ffn_gate_exps.weight"
			}
			return "blk." + layer + ".ffn_" + strings.Join(rest[1:], "_")
		case "input_layernorm":
			return "blk." + layer + ".attn_norm.weight"
		case "post_attention_layernorm":
			return "blk." + layer + ".ffn_norm.weight"
		}
	}
	return name
}

// Plan mirrors ConversionPlan.
type Plan struct {
	Architecture       Architecture
	TensorNameMap      map[string]string
	TargetQuantization *quantization.Type
	SpecialTokens      map[string]uint32
}

// BuildPlan assembles a conversion plan from a parsed GGUF file. The result
// maps every tensor name to its canonical form and the target quantization
// (if any).
func BuildPlan(file ggufcore.File, target *quantization.Type) Plan {
	arch := DetectArchitecture(file.Metadata)
	rename := make(map[string]string)
	for _, info := range file.TensorInfos {
		rename[info.Name] = MapHFTensorName(info.Name)
	}
	special := make(map[string]uint32)
	for _, key := range []string{
		"tokenizer.ggml.bos_token_id",
		"tokenizer.ggml.eos_token_id",
		"tokenizer.ggml.padding_token_id",
		"tokenizer.ggml.unknown_token_id",
	} {
		if id, ok := ParseSpecialTokenID(file.Metadata, key); ok {
			special[key] = id
		}
	}
	return Plan{
		Architecture:       arch,
		TensorNameMap:      rename,
		TargetQuantization: target,
		SpecialTokens:      special,
	}
}

// ParseSpecialTokenID extracts a numeric token id from metadata by key.
func ParseSpecialTokenID(metadata map[string]ggufcore.MetadataValue, key string) (uint32, bool) {
	v, ok := metadata[key]
	if !ok {
		return 0, false
	}
	switch v.Type {
	case ggufcore.MetadataType(0), ggufcore.MetadataType(2), ggufcore.MetadataType(4), ggufcore.MetadataType(10):
		return uint32(v.Uint64), true
	case ggufcore.MetadataType(1), ggufcore.MetadataType(3), ggufcore.MetadataType(5), ggufcore.MetadataType(11):
		if v.Int64 < 0 {
			return 0, false
		}
		return uint32(v.Int64), true
	}
	if s, err := strconv.ParseInt(v.String, 10, 32); err == nil {
		return uint32(s), true
	}
	return 0, false
}
