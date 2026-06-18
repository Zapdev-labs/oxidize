package autotune

import (
	"fmt"
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/core/ggufcore"
	"github.com/Zapdev-labs/oxidize/golang/core/model"
	"github.com/Zapdev-labs/oxidize/golang/core/quantization"
)

// ModelFingerprint holds per-model facts for the tuning planner.
type ModelFingerprint struct {
	Architecture        string
	LayerCount          int
	HiddenSize          int
	NumAttentionHeads   int
	NumKVHeads          int
	HeadDim             int
	IntermediateSize    int
	VocabSize           int
	FileSizeBytes       uint64
	Quant               quantization.Type
	IsMoE               bool
	ExpertCount         int
	HasMTP              bool
}

// Fingerprint builds a fingerprint from a mmap'd GGUF file.
func Fingerprint(mapped *ggufcore.MappedFile) ModelFingerprint {
	cfg := model.InferenceConfigFromGGUF(mapped)
	fileSize := uint64(len(mapped.Bytes))
	quant, isMoE, expertCount, hasMTP := scanTensors(mapped.Parsed)
	arch := strings.ToLower(string(cfg.Architecture))
	if arch == "" {
		arch = strings.ToLower(ggufcore.Architecture(mapped.Parsed))
	}
	return ModelFingerprint{
		Architecture:      arch,
		LayerCount:          cfg.LayerCount,
		HiddenSize:          cfg.HiddenSize,
		NumAttentionHeads:   cfg.NumAttentionHeads,
		NumKVHeads:          cfg.NumKeyValueHeads,
		HeadDim:             cfg.KVHeadDim(),
		IntermediateSize:    cfg.IntermediateSize,
		VocabSize:           cfg.VocabSize,
		FileSizeBytes:       fileSize,
		Quant:               quant,
		IsMoE:               isMoE,
		ExpertCount:         expertCount,
		HasMTP:              hasMTP,
	}
}

// FingerprintFromParts builds a fingerprint for tests.
func FingerprintFromParts(
	architecture string,
	layerCount, hiddenSize, numAttentionHeads, numKVHeads, headDim, intermediateSize, vocabSize int,
	fileSizeBytes uint64,
	quant quantization.Type,
) ModelFingerprint {
	return ModelFingerprint{
		Architecture:      architecture,
		LayerCount:        layerCount,
		HiddenSize:        hiddenSize,
		NumAttentionHeads: numAttentionHeads,
		NumKVHeads:        numKVHeads,
		HeadDim:           headDim,
		IntermediateSize:  intermediateSize,
		VocabSize:         vocabSize,
		FileSizeBytes:     fileSizeBytes,
		Quant:             quant,
	}
}

func scanTensors(file ggufcore.File) (quantization.Type, bool, int, bool) {
	hist := map[uint32]uint64{}
	isMoE := false
	hasMTP := false
	maxExperts := 0
	for _, t := range file.TensorInfos {
		var elems uint64 = 1
		for _, d := range t.Dimensions {
			elems *= d
		}
		hist[t.GGMLType] += elems
		name := t.Name
		if strings.Contains(name, "_exps") || strings.Contains(name, "experts") {
			isMoE = true
		}
		if strings.Contains(name, "nextn") || strings.Contains(name, "mtp") {
			hasMTP = true
		}
		if strings.HasSuffix(name, ".ffn_gate_inp.weight") && len(t.Dimensions) >= 2 {
			n := int(t.Dimensions[len(t.Dimensions)-1])
			if n > maxExperts {
				maxExperts = n
			}
		}
	}
	bestType := uint32(0)
	var bestBytes uint64
	for k, v := range hist {
		if v > bestBytes {
			bestBytes = v
			bestType = k
		}
	}
	return quantization.FromGGMLType(bestType), isMoE, maxExperts, hasMTP
}

// KVBytesPerToken estimates KV cache bytes per token for a dtype width.
func KVBytesPerToken(m ModelFingerprint, kvDTypeBytes int) uint64 {
	if m.LayerCount == 0 || m.HeadDim == 0 {
		return 0
	}
	perLayer := uint64(m.NumKVHeads) * uint64(m.HeadDim) * 2 * uint64(kvDTypeBytes)
	return perLayer * uint64(m.LayerCount)
}

// PerLayerWeightBytes approximates per-layer weight bytes from file size.
func PerLayerWeightBytes(m ModelFingerprint) uint64 {
	if m.LayerCount == 0 {
		return 0
	}
	transformerShare := uint64(float64(m.FileSizeBytes) * 0.85)
	return transformerShare / uint64(m.LayerCount)
}

// ModelSummary returns a one-line model summary.
func ModelSummary(m ModelFingerprint) string {
	moe := ""
	if m.IsMoE {
		moe = fmt.Sprintf(" moe=%d", m.ExpertCount)
	}
	mtp := ""
	if m.HasMTP {
		mtp = " mtp=yes"
	}
	return fmt.Sprintf(
		"%s-like layers=%d hidden=%d heads=%d kv_heads=%d head_dim=%d vocab=%d size=%d MiB quant=%s%s%s",
		m.Architecture,
		m.LayerCount,
		m.HiddenSize,
		m.NumAttentionHeads,
		m.NumKVHeads,
		m.HeadDim,
		m.VocabSize,
		m.FileSizeBytes/(1024*1024),
		m.Quant.String(),
		moe,
		mtp,
	)
}
