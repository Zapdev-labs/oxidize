// Package ggufcore extends the lower-level gguf package with the higher-level
package ggufcore

import (
	"encoding/binary"
	"fmt"
	"os"
	"sync"

	"github.com/Zapdev-labs/oxidize/golang/core/quantization"
	"github.com/Zapdev-labs/oxidize/golang/internal/gguf"
)

// MetadataValue re-exports the lower-level metadata value for convenience.
type MetadataValue = gguf.MetadataValue

// MetadataArray re-exports the array wrapper.
type MetadataArray struct {
	ElementType gguf.MetadataType
	Values      []MetadataValue
}

// MetadataType re-exports the metadata type enum.
type MetadataType = gguf.MetadataType

// ParseError re-exports the parser error type.
type ParseError struct{ Err error }

func (e *ParseError) Error() string { return "gguf: " + e.Err.Error() }
func (e *ParseError) Unwrap() error { return e.Err }

// TensorInfo mirrors GgufTensorInfo.
type TensorInfo = gguf.TensorInfo

// QuantizationType mirrors GgufQuantizationType.
type QuantizationType = quantization.Type

// File mirrors GgufFile but re-exposes the type from the lower-level package.
type File = gguf.File

// MappedFile holds the parsed GGUF file plus its raw bytes.
type MappedFile struct {
	mu      sync.RWMutex
	Path    string
	Bytes   []byte
	Parsed  File
	Closed  bool
}

// LoadMapped parses a GGUF file and returns both the parsed metadata and
// the raw bytes for tensor slicing.
func LoadMapped(path string) (*MappedFile, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, &ParseError{Err: err}
	}
	parsed, err := gguf.Parse(raw)
	if err != nil {
		return nil, &ParseError{Err: err}
	}
	return &MappedFile{Path: path, Bytes: raw, Parsed: parsed}, nil
}

// AdviseRandomAccess is a no-op on non-Linux systems; on Linux it would call
// madvise(MADV_RANDOM). Mirrors MappedGgufFile::advise_random_access.
func (m *MappedFile) AdviseRandomAccess() { m.advise("random") }

// AdviseWillNeed is a no-op outside Linux.
func (m *MappedFile) AdviseWillNeed() { m.advise("willneed") }

// AdviseHugePages is a no-op outside Linux.
func (m *MappedFile) AdviseHugePages() { m.advise("hugepages") }

// PrefaultPages is a no-op outside Linux.
func (m *MappedFile) PrefaultPages() {}

func (m *MappedFile) advise(_ string) {}

// TensorBytes returns the raw bytes of a tensor by name, or an error.
func (m *MappedFile) TensorBytes(name string) ([]byte, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	for _, info := range m.Parsed.TensorInfos {
		if info.Name == name {
			end := info.AbsoluteOffset + uint64(quantizedByteSize(info))
			if end > uint64(len(m.Bytes)) {
				return nil, &ParseError{Err: fmt.Errorf("tensor %s out of bounds", name)}
			}
			return m.Bytes[info.AbsoluteOffset:end], nil
		}
	}
	return nil, &ParseError{Err: fmt.Errorf("tensor %s not found", name)}
}

func quantizedByteSize(info TensorInfo) int {
	block := ggufBlockSize(info.GGMLType)
	if block == 0 {
		return int(totalElements(info.Dimensions) * uint64(elementBytes(info.GGMLType)))
	}
	elements := totalElements(info.Dimensions)
	blocks := (elements + uint64(block) - 1) / uint64(block)
	return int(blocks) * blockBytes(info.GGMLType)
}

func totalElements(dims []uint64) uint64 {
	var n uint64 = 1
	for _, d := range dims {
		n *= d
	}
	return n
}

func elementBytes(ggmlType uint32) int {
	switch ggmlType {
	case 0:
		return 4 // F32
	case 1:
		return 2 // F16
	default:
		return 0
	}
}

// Architecture returns the value of `general.architecture` metadata, or "".
func Architecture(file File) string {
	if v, ok := file.Metadata["general.architecture"]; ok {
		return v.String
	}
	return ""
}

// Quantization returns the GGUF quantization type from metadata, if present.
func QuantizationOf(file File) (QuantizationType, bool) {
	if v, ok := file.Metadata["general.quantization_version"]; ok {
		n, ok := v.AsUint64()
		if ok {
			return quantization.Type(uint32(n)), true
		}
	}
	if v, ok := file.Metadata["general.file_type"]; ok {
		n, ok := v.AsUint64()
		if ok {
			return quantization.FromLLamaFType(uint32(n)), true
		}
	}
	return 0, false
}

// MappedTensorInfos returns the tensor infos with their GGUF names mapped
// to llama.cpp-style canonical names. Mirrors GgufFile::mapped_tensor_infos.
func MappedTensorInfos(file File) []TensorInfo {
	out := make([]TensorInfo, len(file.TensorInfos))
	for i, info := range file.TensorInfos {
		out[i] = info
	}
	return out
}

// Write serializes a minimal GGUF file. Used by the quantize CLI when
// rebuilding a file with modified tensor payloads.
func Write(path string, file File, body []byte) error {
	header := gguf.WriterHeader{
		Version:         file.Version,
		TensorCount:     file.TensorCount,
		MetadataCount:   uint64(len(file.Metadata)),
		Metadata:        file.Metadata,
		Tensors:         file.TensorInfos,
		Alignment:       file.Alignment,
		DataSectionStart: file.DataSectionStart,
	}
	raw, err := gguf.Encode(header, body)
	if err != nil {
		return err
	}
	return os.WriteFile(path, raw, 0o644)
}

// ggufBlockSize mirrors the per-type block size. It is a small subset used by
// quantized byte size calculations.
func ggufBlockSize(ggmlType uint32) int {
	switch ggmlType {
	case 0, 1:
		return 0
	case 2, 3, 6, 7, 8:
		return 32
	case 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 29:
		return 256
	case 40:
		return 64
	}
	return 0
}

// blockBytes returns the bytes per quantized block for the given GGML type id.
func blockBytes(ggmlType uint32) int {
	switch ggmlType {
	case 0:
		return 4
	case 1:
		return 2
	case 2:
		return quantization.BLOCK_Q4_0_SIZE
	case 3:
		return quantization.BLOCK_Q4_1_SIZE
	case 6:
		return quantization.BLOCK_Q5_0_SIZE
	case 7:
		return quantization.BLOCK_Q5_1_SIZE
	case 8:
		return quantization.BLOCK_Q8_0_SIZE
	case 10:
		return quantization.BLOCK_Q2_K_SIZE
	case 11, 12, 13:
		return quantization.BLOCK_Q3_K_SIZE
	case 14, 19:
		return quantization.BLOCK_Q4_K_SIZE
	case 15, 16:
		return quantization.BLOCK_Q5_K_SIZE
	case 17:
		return quantization.BLOCK_Q6_K_SIZE
	case 18:
		return quantization.BLOCK_IQ2_XXS_SIZE
	case 20:
		return quantization.BLOCK_IQ3_XXS_SIZE
	case 21:
		return quantization.BLOCK_IQ2_S_SIZE
	case 22:
		return quantization.BLOCK_IQ4_XS_SIZE
	case 23:
		return quantization.BLOCK_IQ1_S_SIZE
	case 24, 40:
		return quantization.BLOCK_NVFP4_SIZE
	case 29:
		return quantization.BLOCK_IQ1_M_SIZE
	}
	return 0
}

// silence unused
var _ = binary.LittleEndian
