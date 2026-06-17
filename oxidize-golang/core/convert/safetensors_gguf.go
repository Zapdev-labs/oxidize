// Package convert implements SafeTensors → GGUF conversion (metadata + tensor copy).
package convert

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/Zapdev-labs/oxidize/golang/core/conversion"
	"github.com/Zapdev-labs/oxidize/golang/core/quantization"
	"github.com/Zapdev-labs/oxidize/golang/core/safetensors"
	"github.com/Zapdev-labs/oxidize/golang/core/tensor"
	"github.com/Zapdev-labs/oxidize/golang/internal/gguf"
)

// Config controls safetensors → GGUF conversion.
type Config struct {
	InputPath       string
	OutputPath      string
	ArchOverride    string
	MapHFTensorName bool
	ConfigPath      string
}

// ConvertSafeTensorsToGGUF copies tensor payloads as F32 into a GGUF v3 file.
func ConvertSafeTensorsToGGUF(cfg Config) error {
	if strings.TrimSpace(cfg.InputPath) == "" {
		return fmt.Errorf("convert: empty input path")
	}
	if strings.TrimSpace(cfg.OutputPath) == "" {
		return fmt.Errorf("convert: empty output path")
	}
	st, err := safetensors.Load(cfg.InputPath)
	if err != nil {
		return fmt.Errorf("convert: load safetensors: %w", err)
	}
	tensors := st.Tensors()
	sort.Slice(tensors, func(i, j int) bool { return tensors[i].Name < tensors[j].Name })

	meta := map[string]gguf.MetadataValue{
		"general.quantization_version": {Type: gguf.MetadataUint32, Uint64: 2},
		"general.file_type":            {Type: gguf.MetadataUint32, Uint64: 1},
	}
	arch := strings.TrimSpace(cfg.ArchOverride)
	if arch == "" {
		arch = detectArch(cfg.ConfigPath, cfg.InputPath)
	}
	if arch != "" {
		meta["general.architecture"] = gguf.MetadataValue{Type: gguf.MetadataString, String: arch}
	}

	var infos []gguf.TensorInfo
	var body []byte
	align := uint64(32)
	for _, ti := range tensors {
		name := ti.Name
		if cfg.MapHFTensorName {
			name = conversion.MapHFTensorName(name)
		}
		raw, err := st.TensorData(ti.Name)
		if err != nil {
			return fmt.Errorf("convert: tensor %q: %w", ti.Name, err)
		}
		f32, dims, err := tensorToF32(ti, raw)
		if err != nil {
			return fmt.Errorf("convert: tensor %q: %w", ti.Name, err)
		}
		if len(dims) == 0 {
			continue
		}
		pad := int((align - uint64(len(body))%align) % align)
		if pad > 0 {
			body = append(body, make([]byte, pad)...)
		}
		offset := uint64(len(body))
		outBytes := make([]byte, len(f32)*4)
		for i, v := range f32 {
			binary.LittleEndian.PutUint32(outBytes[i*4:], math.Float32bits(v))
		}
		body = append(body, outBytes...)
		dimU64 := make([]uint64, len(dims))
		for i, d := range dims {
			dimU64[i] = uint64(d)
		}
		infos = append(infos, gguf.TensorInfo{
			Name:           name,
			Dimensions:     dimU64,
			GGMLType:       uint32(quantization.TypeF32),
			RelativeOffset: offset,
		})
	}
	header := gguf.WriterHeader{
		Version:          3,
		Metadata:         meta,
		Tensors:          infos,
		Alignment:        align,
		DataSectionStart: 0,
	}
	out, err := gguf.Encode(header, body)
	if err != nil {
		return fmt.Errorf("convert: encode gguf: %w", err)
	}
	if err := os.WriteFile(cfg.OutputPath, out, 0o644); err != nil {
		return fmt.Errorf("convert: write output: %w", err)
	}
	return nil
}

func detectArch(configPath, inputPath string) string {
	paths := []string{configPath}
	if configPath == "" {
		if fi, err := os.Stat(inputPath); err == nil && fi.IsDir() {
			paths = []string{filepath.Join(inputPath, "config.json")}
		} else {
			paths = []string{filepath.Join(filepath.Dir(inputPath), "config.json")}
		}
	}
	for _, p := range paths {
		if p == "" {
			continue
		}
		raw, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		var cfg map[string]json.RawMessage
		if json.Unmarshal(raw, &cfg) != nil {
			continue
		}
		if arch, ok := cfg["architectures"]; ok {
			var names []string
			if json.Unmarshal(arch, &names) == nil && len(names) > 0 {
				return strings.ToLower(names[0])
			}
		}
		if mt, ok := cfg["model_type"]; ok {
			var s string
			if json.Unmarshal(mt, &s) == nil {
				return strings.ToLower(s)
			}
		}
	}
	return "llama"
}

func tensorToF32(ti safetensors.TensorInfo, raw []byte) ([]float32, []int, error) {
	elems := 1
	for _, d := range ti.Shape {
		elems *= d
	}
	out := make([]float32, elems)
	switch ti.DType {
	case safetensors.DTypeF32:
		if len(raw) < elems*4 {
			return nil, nil, fmt.Errorf("f32 payload too small")
		}
		for i := 0; i < elems; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	case safetensors.DTypeF16:
		if len(raw) < elems*2 {
			return nil, nil, fmt.Errorf("f16 payload too small")
		}
		for i := 0; i < elems; i++ {
			out[i] = tensor.F16BitsToF32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	default:
		return nil, nil, fmt.Errorf("unsupported dtype %s", ti.DType)
	}
	return out, ti.Shape, nil
}
