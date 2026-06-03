package model

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/Zapdev-labs/oxidize/golang/core/ggufcore"
	"github.com/Zapdev-labs/oxidize/golang/core/quantization"
	"github.com/Zapdev-labs/oxidize/golang/core/safetensors"
	"github.com/Zapdev-labs/oxidize/golang/core/tensor"
)

// LoadDFlashFromGGUF loads a DFlash draft model from a mapped GGUF file.
func LoadDFlashFromGGUF(mapped *ggufcore.MappedFile, config DFlashConfig) (*DFlashDraftModel, error) {
	m := NewDFlashDraftModel(config)
	stack, err := LoadLlamaDecoderStackFromGGUF(mapped, LlamaDecoderConfigFromDFlash(config))
	if err != nil {
		return nil, err
	}
	m.Stack = stack

	file := mapped.Parsed
	infos := ggufcore.MappedTensorInfos(file)
	bytesAll := mapped.Bytes

	loadF32WithDims := func(name string) ([]float32, []uint64, bool, error) {
		var info *ggufcore.TensorInfo
		for i := range infos {
			if infos[i].Name == name {
				info = &infos[i]
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
		if end > len(bytesAll) {
			return nil, nil, false, fmt.Errorf("tensor %s out of bounds", name)
		}
		out := make([]float32, count)
		if err := quantization.DequantizeScalar(qtype, bytesAll[off:end], out); err != nil {
			return nil, nil, false, fmt.Errorf("dequantize %s: %w", name, err)
		}
		return out, info.Dimensions, true, nil
	}

	loadProj := func(name string) (F32Weight, error) {
		var info *ggufcore.TensorInfo
		for i := range infos {
			if infos[i].Name == name {
				info = &infos[i]
				break
			}
		}
		if info == nil {
			return F32Weight{}, nil
		}
		if len(info.Dimensions) != 2 {
			data, dims, ok, err := loadF32WithDims(name)
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
			if end > len(bytesAll) {
				return F32Weight{}, fmt.Errorf("tensor %s out of bounds", name)
			}
			return NewF32WeightFromQuantized(append([]byte(nil), bytesAll[off:end]...), qtype, outDim, inDim), nil
		}
		data, _, ok, err := loadF32WithDims(name)
		if err != nil || !ok {
			return F32Weight{}, err
		}
		return NewF32WeightFromSlice(transposeF32(data, inDim, outDim), outDim, inDim), nil
	}

	loadProjAny := func(names ...string) (F32Weight, error) {
		for _, name := range names {
			w, err := loadProj(name)
			if err != nil {
				return F32Weight{}, err
			}
			if w.IsLoaded() {
				return w, nil
			}
		}
		return F32Weight{}, nil
	}

	loadF32Any := func(names ...string) ([]float32, error) {
		for _, name := range names {
			data, _, ok, err := loadF32WithDims(name)
			if err != nil {
				return nil, err
			}
			if ok {
				return data, nil
			}
		}
		return nil, nil
	}

	fc, err := loadProjAny("fc.weight", "dflash_fc.weight", "model.fc.weight")
	if err != nil {
		return nil, err
	}
	if fc.IsLoaded() {
		m.FC = fc
	}
	if hiddenNorm, err := loadF32Any("hidden_norm.weight", "dflash_hidden_norm.weight", "model.hidden_norm.weight"); err != nil {
		return nil, err
	} else if len(hiddenNorm) > 0 {
		m.HiddenNorm = hiddenNorm
	}
	if m.Stack.Output.IsLoaded() {
		m.Config.VocabSize = m.Stack.Output.outputDim()
	} else if m.Stack.TokEmbeddings.IsLoaded() {
		m.Config.VocabSize = m.Stack.TokEmbeddings.outputDim()
	}
	return m, nil
}

// LoadDFlashExternalIOFromGGUF borrows lm_head / embeddings from another GGUF.
func (m *DFlashDraftModel) LoadExternalIOFromGGUF(mapped *ggufcore.MappedFile) error {
	file := mapped.Parsed
	infos := ggufcore.MappedTensorInfos(file)
	bytesAll := mapped.Bytes

	loadProj := func(name string) (F32Weight, error) {
		var info *ggufcore.TensorInfo
		for i := range infos {
			if infos[i].Name == name {
				info = &infos[i]
				break
			}
		}
		if info == nil || len(info.Dimensions) != 2 {
			return F32Weight{}, nil
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
			if end > len(bytesAll) {
				return F32Weight{}, fmt.Errorf("tensor %s out of bounds", name)
			}
			return NewF32WeightFromQuantized(append([]byte(nil), bytesAll[off:end]...), qtype, outDim, inDim), nil
		}
		count := inDim * outDim
		qsize, err := quantization.QuantizedSize(qtype, count)
		if err != nil {
			return F32Weight{}, err
		}
		off := int(info.AbsoluteOffset)
		data := make([]float32, count)
		if err := quantization.DequantizeScalar(qtype, bytesAll[off:off+qsize], data); err != nil {
			return F32Weight{}, err
		}
		return NewF32WeightFromSlice(transposeF32(data, inDim, outDim), outDim, inDim), nil
	}

	for _, name := range []string{"lm_head.weight", "output.weight"} {
		w, err := loadProj(name)
		if err != nil {
			return err
		}
		if w.IsLoaded() {
			m.Stack.Output = w
			m.Config.VocabSize = w.outputDim()
			break
		}
	}
	for _, name := range []string{"model.embed_tokens.weight", "tok_embeddings.weight", "token_embd.weight"} {
		w, err := loadProj(name)
		if err != nil {
			return err
		}
		if w.IsLoaded() {
			m.Stack.TokEmbeddings = w
			if !m.Stack.Output.IsLoaded() {
				m.Config.VocabSize = w.outputDim()
			}
			break
		}
	}
	return nil
}

// LoadDFlashFromSafeTensors loads weights from a HuggingFace-style SafeTensors file.
func LoadDFlashFromSafeTensors(mapped *safetensors.MappedFile, config DFlashConfig) (*DFlashDraftModel, error) {
	m := NewDFlashDraftModel(config)
	loadF32 := func(name string) ([]float32, error) {
		var info *safetensors.TensorInfo
		for i := range mapped.Tensors() {
			if mapped.Tensors()[i].Name == name {
				info = &mapped.Tensors()[i]
				break
			}
		}
		if info == nil {
			return nil, nil
		}
		raw, err := mapped.TensorData(name)
		if err != nil {
			return nil, err
		}
		count := 1
		for _, d := range info.Shape {
			count *= d
		}
		out := make([]float32, count)
		switch info.DType {
		case safetensors.DTypeF32:
			if len(raw) < count*4 {
				return nil, fmt.Errorf("tensor %s: insufficient f32 bytes", name)
			}
			for i := 0; i < count; i++ {
				out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
			}
		case safetensors.DTypeF16:
			if len(raw) < count*2 {
				return nil, fmt.Errorf("tensor %s: insufficient f16 bytes", name)
			}
			for i := 0; i < count; i++ {
				out[i] = tensor.F16LEToF32([2]byte{raw[i*2], raw[i*2+1]})
			}
		default:
			return nil, fmt.Errorf("tensor %s: unsupported dtype %s", name, info.DType)
		}
		return out, nil
	}

	if data, err := loadF32("model.fc.weight"); err != nil {
		return nil, err
	} else if len(data) > 0 {
		rows := config.HiddenSize
		cols := len(data) / rows
		m.FC = NewF32WeightFromSlice(data, rows, cols)
	}
	if data, err := loadF32("model.fc.bias"); err != nil {
		return nil, err
	} else if len(data) > 0 {
		m.FCBias = data
	}
	if data, err := loadF32("model.hidden_norm.weight"); err != nil {
		return nil, err
	} else if len(data) > 0 {
		m.HiddenNorm = data
	}
	if data, err := loadF32("model.norm.weight"); err != nil {
		return nil, err
	} else if len(data) > 0 {
		m.Stack.Norm = data
	}
	if data, err := loadF32("lm_head.weight"); err != nil {
		return nil, err
	} else if len(data) > 0 {
		rows := config.VocabSize
		cols := len(data) / rows
		m.Stack.Output = NewF32WeightFromSlice(data, rows, cols)
	}
	if data, err := loadF32("model.embed_tokens.weight"); err != nil {
		return nil, err
	} else if len(data) > 0 {
		rows := config.VocabSize
		cols := len(data) / rows
		m.Stack.TokEmbeddings = NewF32WeightFromSlice(data, rows, cols)
	}

	for layerIdx := 0; layerIdx < config.NumHiddenLayers; layerIdx++ {
		prefix := fmt.Sprintf("model.layers.%d", layerIdx)
		inLN, _ := loadF32(prefix + ".input_layernorm.weight")
		if len(inLN) == 0 {
			inLN = ones(config.HiddenSize)
		}
		postLN, _ := loadF32(prefix + ".post_attention_layernorm.weight")
		if len(postLN) == 0 {
			postLN = ones(config.HiddenSize)
		}
		qData, _ := loadF32(prefix + ".self_attn.q_proj.weight")
		kData, _ := loadF32(prefix + ".self_attn.k_proj.weight")
		vData, _ := loadF32(prefix + ".self_attn.v_proj.weight")
		oData, _ := loadF32(prefix + ".self_attn.o_proj.weight")
		qNorm, _ := loadF32(prefix + ".self_attn.q_norm.weight")
		if len(qNorm) == 0 {
			qNorm = ones(config.HeadDim())
		}
		kNorm, _ := loadF32(prefix + ".self_attn.k_norm.weight")
		if len(kNorm) == 0 {
			kNorm = ones(config.HeadDim())
		}
		gateData, _ := loadF32(prefix + ".mlp.gate_proj.weight")
		upData, _ := loadF32(prefix + ".mlp.up_proj.weight")
		downData, _ := loadF32(prefix + ".mlp.down_proj.weight")

		m.Stack.Layers = append(m.Stack.Layers, DecoderLayer{
			InputLayernorm:         inLN,
			PostAttentionLayernorm: postLN,
			MLPGate:                weightFrom2D(gateData, config.IntermediateSize, config.HiddenSize),
			MLPUp:                  weightFrom2D(upData, config.IntermediateSize, config.HiddenSize),
			MLPDown:                weightFrom2D(downData, config.HiddenSize, config.IntermediateSize),
			Attention: DecoderAttentionLayer{
				QProj:       weightFrom2D(qData, config.HiddenSize, config.HiddenSize),
				KProj:       weightFrom2D(kData, config.NumKeyValueHeads*config.KVHeadDim(), config.HiddenSize),
				VProj:       weightFrom2D(vData, config.NumKeyValueHeads*config.KVHeadDim(), config.HiddenSize),
				OProj:       weightFrom2D(oData, config.HiddenSize, config.HiddenSize),
				QNormWeight: qNorm,
				KNormWeight: kNorm,
			},
		})
	}
	return m, nil
}

func f32WeightFromDims(data []float32, dims []uint64) F32Weight {
	switch len(dims) {
	case 0:
		return F32Weight{}
	case 1:
		n := int(dims[0])
		return NewF32WeightFromSlice(data, n, 1)
	default:
		r := int(dims[0])
		c := int(dims[1])
		return NewF32WeightFromSlice(transposeF32(data, r, c), c, r)
	}
}

func weightFrom2D(data []float32, rows, cols int) F32Weight {
	if len(data) == 0 {
		return F32Weight{}
	}
	if cols == 0 {
		cols = len(data) / rows
	}
	return NewF32WeightFromSlice(data, rows, cols)
}

func ones(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = 1
	}
	return out
}
