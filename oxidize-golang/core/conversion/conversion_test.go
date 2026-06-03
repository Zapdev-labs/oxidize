package conversion

import (
	"testing"

	"github.com/Zapdev-labs/oxidize/golang/core/ggufcore"
)

func TestDetectArchitecture(t *testing.T) {
	cases := map[string]Architecture{
		"llama":   ArchLlama,
		"mistral": ArchMistral,
		"qwen2":   ArchQwen,
		"gemma":   ArchGemma,
		"phi":     ArchPhi,
		"??":      ArchUnknown,
	}
	for arch, want := range cases {
		got := DetectArchitecture(map[string]ggufcore.MetadataValue{
			"general.architecture": {Type: ggufcore.MetadataType(8), String: arch},
		})
		if got != want {
			t.Fatalf("arch %q = %v, want %v", arch, got, want)
		}
	}
}

func TestMapHFTensorName(t *testing.T) {
	cases := map[string]string{
		"model.embed_tokens.weight":    "token_embd.weight",
		"model.norm.weight":            "output_norm.weight",
		"model.layers.0.self_attn.q_proj.weight": "blk.0.attn_q.weight",
		"model.layers.1.mlp.gate_proj.weight":   "blk.1.ffn_gate.weight",
		"model.layers.2.input_layernorm.weight": "blk.2.attn_norm.weight",
		"lm_head.weight":                "output.weight",
	}
	for in, want := range cases {
		if got := MapHFTensorName(in); got != want {
			t.Fatalf("%q -> %q, want %q", in, got, want)
		}
	}
}

func TestBuildPlan(t *testing.T) {
	plan := BuildPlan(ggufcore.File{
		TensorInfos: []ggufcore.TensorInfo{
			{Name: "model.embed_tokens.weight"},
			{Name: "model.layers.0.self_attn.q_proj.weight"},
		},
		Metadata: map[string]ggufcore.MetadataValue{
			"general.architecture":                {Type: ggufcore.MetadataType(8), String: "llama"},
			"tokenizer.ggml.bos_token_id":         {Type: ggufcore.MetadataType(4), Uint64: 1},
		},
	}, nil)
	if plan.Architecture != ArchLlama {
		t.Fatalf("arch = %v", plan.Architecture)
	}
	if plan.TensorNameMap["model.embed_tokens.weight"] != "token_embd.weight" {
		t.Fatalf("rename missing")
	}
	if plan.SpecialTokens["tokenizer.ggml.bos_token_id"] != 1 {
		t.Fatalf("special token = %v", plan.SpecialTokens)
	}
}
