package model

import (
	"fmt"

	"github.com/Zapdev-labs/oxidize/golang/core/ggufcore"
	"github.com/Zapdev-labs/oxidize/golang/core/quantization"
)

type ggufWeightLoader struct {
	infos    []ggufcore.TensorInfo
	bytesAll []byte
	config   LlamaDecoderConfig
}

func newGGUFWeightLoader(mapped *ggufcore.MappedFile, config LlamaDecoderConfig) *ggufWeightLoader {
	return &ggufWeightLoader{
		infos:    ggufcore.MappedTensorInfos(mapped.Parsed),
		bytesAll: mapped.Bytes,
		config:   config,
	}
}

func (l *ggufWeightLoader) loadF32WithDims(name string) ([]float32, []uint64, bool, error) {
	var info *ggufcore.TensorInfo
	for i := range l.infos {
		if l.infos[i].Name == name {
			info = &l.infos[i]
			break
		}
	}
	if info == nil {
		return nil, nil, false, nil
	}
	qtype := quantization.FromGGMLType(info.GGMLType)
	count := 1
	for _, d := range info.Dimensions {
		count *= int(d)
	}
	qsize, err := quantization.QuantizedSize(qtype, count)
	if err != nil {
		return nil, nil, false, err
	}
	off := int(info.AbsoluteOffset)
	end := off + qsize
	if end > len(l.bytesAll) {
		return nil, nil, false, fmt.Errorf("tensor %s out of bounds", name)
	}
	out := make([]float32, count)
	if err := quantization.DequantizeScalar(qtype, l.bytesAll[off:end], out); err != nil {
		return nil, nil, false, fmt.Errorf("dequantize %s: %w", name, err)
	}
	return out, info.Dimensions, true, nil
}

func (l *ggufWeightLoader) loadProj(name string) (F32Weight, error) {
	var info *ggufcore.TensorInfo
	for i := range l.infos {
		if l.infos[i].Name == name {
			info = &l.infos[i]
			break
		}
	}
	if info == nil {
		return F32Weight{}, nil
	}
	if len(info.Dimensions) != 2 {
		data, dims, ok, err := l.loadF32WithDims(name)
		if err != nil || !ok {
			return F32Weight{}, err
		}
		return f32WeightFromDims(data, dims), nil
	}
	qtype := quantization.FromGGMLType(info.GGMLType)
	inDim := int(info.Dimensions[0])
	outDim := int(info.Dimensions[1])
	if quantizedGemvSupported(qtype, inDim) {
		count := outDim * inDim
		qsize, err := quantization.QuantizedSize(qtype, count)
		if err != nil {
			return F32Weight{}, err
		}
		off := int(info.AbsoluteOffset)
		end := off + qsize
		if end > len(l.bytesAll) {
			return F32Weight{}, fmt.Errorf("tensor %s out of bounds", name)
		}
		return NewF32WeightFromQuantized(append([]byte(nil), l.bytesAll[off:end]...), qtype, outDim, inDim), nil
	}
	data, _, ok, err := l.loadF32WithDims(name)
	if err != nil || !ok {
		return F32Weight{}, err
	}
	return NewF32WeightFromSlice(transposeF32(data, inDim, outDim), outDim, inDim), nil
}

func (l *ggufWeightLoader) loadProjAny(names ...string) (F32Weight, error) {
	for _, name := range names {
		w, err := l.loadProj(name)
		if err != nil {
			return F32Weight{}, err
		}
		if w.IsLoaded() {
			return w, nil
		}
	}
	return F32Weight{}, nil
}

func (l *ggufWeightLoader) loadF32Any(names ...string) ([]float32, error) {
	for _, name := range names {
		data, _, ok, err := l.loadF32WithDims(name)
		if err != nil {
			return nil, err
		}
		if ok {
			return data, nil
		}
	}
	return nil, nil
}

// LoadLlamaDecoderStackFromGGUF loads tok embeddings, output head, norms, and blk.* layers.
func LoadLlamaDecoderStackFromGGUF(mapped *ggufcore.MappedFile, config LlamaDecoderConfig) (*LlamaDecoderStack, error) {
	stack := NewLlamaDecoderStack(config)
	loader := newGGUFWeightLoader(mapped, config)

	if norm, err := loader.loadF32Any("output_norm.weight", "norm.weight", "model.norm.weight"); err != nil {
		return nil, err
	} else if len(norm) > 0 {
		stack.Norm = norm
	}
	output, err := loader.loadProjAny("lm_head.weight", "output.weight")
	if err != nil {
		return nil, err
	}
	if output.IsLoaded() {
		stack.Output = output
	}
	tokEmb, err := loader.loadProjAny("model.embed_tokens.weight", "tok_embeddings.weight", "token_embd.weight")
	if err != nil {
		return nil, err
	}
	if tokEmb.IsLoaded() {
		stack.TokEmbeddings = tokEmb
		if config.VocabSize == 0 {
			stack.Config.VocabSize = tokEmb.outputDim()
		}
	}
	if !stack.Output.IsLoaded() && stack.TokEmbeddings.IsLoaded() {
		stack.Output = stack.TokEmbeddings
	}

	for layerIdx := 0; layerIdx < config.LayerCount; layerIdx++ {
		prefix := fmt.Sprintf("blk.%d", layerIdx)
		inLN := ones(config.HiddenSize)
		if data, _, ok, _ := loader.loadF32WithDims(prefix + ".attn_norm.weight"); ok {
			inLN = data
		}
		postLN := ones(config.HiddenSize)
		if data, _, ok, _ := loader.loadF32WithDims(prefix + ".post_attention_norm.weight"); ok {
			postLN = data
		} else if data, _, ok, _ := loader.loadF32WithDims(prefix + ".ffn_norm.weight"); ok {
			postLN = data
		}
		// Gemma sandwich norm: post_attention_norm is the true post-attention
		// norm, ffn_norm is the pre-MLP norm, and post_ffw_norm is the post-FFN
		// norm. Load them separately so the forward pass can apply all four.
		var preFFN, postFFN []float32
		if config.SandwichNorm {
			if data, _, ok, _ := loader.loadF32WithDims(prefix + ".post_attention_norm.weight"); ok {
				postLN = data
			}
			if data, _, ok, _ := loader.loadF32WithDims(prefix + ".ffn_norm.weight"); ok {
				preFFN = data
			} else {
				preFFN = ones(config.HiddenSize)
			}
			if data, _ := loader.loadF32Any(prefix+".post_ffw_norm.weight", prefix+".post_ffn_norm.weight"); len(data) > 0 {
				postFFN = data
			} else {
				postFFN = ones(config.HiddenSize)
			}
		}
		qProj, err := loader.loadProj(prefix + ".attn_q.weight")
		if err != nil {
			return nil, err
		}
		kProj, err := loader.loadProj(prefix + ".attn_k.weight")
		if err != nil {
			return nil, err
		}
		vProj, err := loader.loadProj(prefix + ".attn_v.weight")
		if err != nil {
			return nil, err
		}
		oProj, err := loader.loadProj(prefix + ".attn_output.weight")
		if err != nil {
			return nil, err
		}
		qNorm := ones(config.KVHeadDim())
		if config.KVHeadDim() == 0 {
			qNorm = ones(config.HeadDim())
		}
		if data, _, ok, _ := loader.loadF32WithDims(prefix + ".attn_q_norm.weight"); ok {
			qNorm = data
		}
		kNorm := ones(config.KVHeadDim())
		if config.KVHeadDim() == 0 {
			kNorm = ones(config.HeadDim())
		}
		if data, _, ok, _ := loader.loadF32WithDims(prefix + ".attn_k_norm.weight"); ok {
			kNorm = data
		}
		gate, err := loader.loadProj(prefix + ".ffn_gate.weight")
		if err != nil {
			return nil, err
		}
		up, err := loader.loadProj(prefix + ".ffn_up.weight")
		if err != nil {
			return nil, err
		}
		down, err := loader.loadProj(prefix + ".ffn_down.weight")
		if err != nil {
			return nil, err
		}
		moe := MoELayer{NumExperts: config.NumExperts}
		if config.NumExperts > 0 {
			gateInp, err := loader.loadProj(prefix + ".ffn_gate_inp.weight")
			if err != nil {
				return nil, err
			}
			gateExps, err := loader.loadProj(prefix + ".ffn_gate_exps.weight")
			if err != nil {
				return nil, err
			}
			upExps, err := loader.loadProj(prefix + ".ffn_up_exps.weight")
			if err != nil {
				return nil, err
			}
			downExps, err := loader.loadProj(prefix + ".ffn_down_exps.weight")
			if err != nil {
				return nil, err
			}
			moe = MoELayer{
				GateInp:    gateInp,
				GateExps:   gateExps,
				UpExps:     upExps,
				DownExps:   downExps,
				NumExperts: config.NumExperts,
			}
		}
		stack.Layers = append(stack.Layers, DecoderLayer{
			InputLayernorm:         inLN,
			PostAttentionLayernorm: postLN,
			PreFFNLayernorm:        preFFN,
			PostFFNLayernorm:       postFFN,
			MLPGate:                gate,
			MLPUp:                  up,
			MLPDown:                down,
			MoE:                    moe,
			Attention: DecoderAttentionLayer{
				QProj:       qProj,
				KProj:       kProj,
				VProj:       vProj,
				OProj:       oProj,
				QNormWeight: qNorm,
				KNormWeight: kNorm,
			},
		})
	}
	return stack, nil
}
