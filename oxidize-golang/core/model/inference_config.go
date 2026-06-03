package model

import (
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/core/ggufcore"
)

// InferenceConfigFromGGUF builds config from GGUF metadata and tensor shapes.
func InferenceConfigFromGGUF(mapped *ggufcore.MappedFile) InferenceConfig {
	file := mapped.Parsed
	out := DefaultInferenceConfig()
	arch := ggufcore.Architecture(file)
	if arch == "" {
		arch = "llama"
	}
	out.Architecture = architectureFromGGUFString(arch)

	archKey := func(suffix string) string { return arch + "." + suffix }
	archU32 := func(suffix string) (uint32, bool) {
		if v, ok := file.Metadata[archKey(suffix)]; ok {
			if n, ok := v.AsUint64(); ok {
				return uint32(n), true
			}
		}
		if v, ok := file.Metadata["llama."+suffix]; ok {
			if n, ok := v.AsUint64(); ok {
				return uint32(n), true
			}
		}
		return 0, false
	}
	archF32 := func(suffix string) (float32, bool) {
		if v, ok := file.Metadata[archKey(suffix)]; ok {
			if f, ok := v.AsFloat32(); ok {
				return f, true
			}
		}
		if v, ok := file.Metadata["llama."+suffix]; ok {
			if f, ok := v.AsFloat32(); ok {
				return f, true
			}
		}
		return 0, false
	}

	if n, ok := archU32("context_length"); ok {
		out.ContextSize = int(n)
	}
	if n, ok := archU32("embedding_length"); ok {
		out.HiddenSize = int(n)
	}
	if n, ok := archU32("attention.head_count"); ok {
		out.NumAttentionHeads = int(n)
	}
	if n, ok := archU32("attention.head_count_kv"); ok {
		out.NumKeyValueHeads = int(n)
	} else {
		out.NumKeyValueHeads = out.NumAttentionHeads
	}
	if n, ok := archU32("attention.key_length"); ok {
		out.KeyValueHeadDim = int(n)
	}
	if n, ok := archU32("feed_forward_length"); ok {
		out.IntermediateSize = int(n)
	}
	if n, ok := archU32("block_count"); ok {
		out.LayerCount = int(n)
	}
	if f, ok := archF32("attention.layer_norm_rms_epsilon"); ok {
		out.RMSNormEps = f
	}
	if f, ok := archF32("rope.freq_base"); ok {
		out.RopeTheta = f
	}
	if n, ok := archU32("vocab_size"); ok {
		out.VocabSize = int(n)
	}
	if n, ok := archU32("attention.sliding_window"); ok {
		out.SlidingWindow = int(n)
	}
	if n, ok := archU32("expert_count"); ok {
		out.NumExperts = int(n)
	}
	if n, ok := archU32("expert_used_count"); ok {
		out.NumExpertsPerToken = int(n)
	}
	if out.NumExpertsPerToken == 0 {
		out.NumExpertsPerToken = 2
	}
	if out.NumExperts == 0 {
		for _, info := range ggufcore.MappedTensorInfos(file) {
			if strings.HasPrefix(info.Name, "blk.0.") && strings.Contains(info.Name, "ffn_gate_exps") {
				if n, ok := archU32("expert_count"); ok {
					out.NumExperts = int(n)
				}
				break
			}
		}
	}

	for _, info := range ggufcore.MappedTensorInfos(file) {
		switch info.Name {
		case "token_embd.weight", "tok_embeddings.weight", "model.embed_tokens.weight":
			if len(info.Dimensions) >= 2 {
				out.VocabSize = int(info.Dimensions[len(info.Dimensions)-2])
				out.HiddenSize = int(info.Dimensions[len(info.Dimensions)-1])
			}
		}
	}
	if out.VocabSize == 0 {
		out.VocabSize = 32000
	}
	return out
}

func architectureFromGGUFString(arch string) Architecture {
	switch arch {
	case "llama":
		return ArchLlamaModel
	case "mistral":
		return ArchMistralModel
	case "mixtral":
		return ArchMixtralModel
	case "deepseek", "deepseek_v2", "deepseek_v3", "deepseek_moe":
		return ArchDeepSeekModel
	case "qwen", "qwen2", "qwen2moe", "qwen3", "qwen35":
		return ArchQwenModel
	case "gemma", "gemma3", "gemma4":
		return ArchGemmaModel
	case "phi", "phi2", "phi3":
		return ArchPhiModel
	case "falcon":
		return ArchFalconModel
	case "gpt2":
		return ArchGpt2Model
	case "gptj":
		return ArchGptJModel
	case "gptneox":
		return ArchGptNeoXModel
	case "minimax", "minimax-m2", "minimax-text-01":
		return ArchMiniMaxModel
	default:
		return ArchLlamaModel
	}
}

// LoadInferenceFromGGUF loads weights and returns a ready InferenceModel.
func LoadInferenceFromGGUF(mapped *ggufcore.MappedFile) (*InferenceModel, error) {
	config := InferenceConfigFromGGUF(mapped)
	decoderCfg := LlamaDecoderConfigFromInference(config)
	stack, err := LoadLlamaDecoderStackFromGGUF(mapped, decoderCfg)
	if err != nil {
		return nil, err
	}
	if stack.Output.IsLoaded() && config.VocabSize == 0 {
		config.VocabSize = stack.Output.outputDim()
	}
	if stack.TokEmbeddings.IsLoaded() && config.VocabSize == 0 {
		config.VocabSize = stack.TokEmbeddings.outputDim()
	}
	return NewInferenceModel(config, WeightStorage{File: mapped}, stack), nil
}
