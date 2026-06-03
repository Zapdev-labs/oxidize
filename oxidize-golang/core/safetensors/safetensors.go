// Package safetensors mirrors oxidize_core::format::safetensors. It reads
// SafeTensors files (header + tensor info + lazy access to tensor bytes) and
// is used by the model loader for HF-style model files.
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// DType mirrors safetensors::Dtype values used in tensor metadata.
type DType uint8

const (
	DTypeBool DType = iota
	DTypeI8
	DTypeI16
	DTypeI32
	DTypeI64
	DTypeU8
	DTypeF16
	DTypeBF16
	DTypeF32
	DTypeF64
	DTypeUnknown
)

func (d DType) String() string {
	switch d {
	case DTypeBool:
		return "bool"
	case DTypeI8:
		return "i8"
	case DTypeI16:
		return "i16"
	case DTypeI32:
		return "i32"
	case DTypeI64:
		return "i64"
	case DTypeU8:
		return "u8"
	case DTypeF16:
		return "f16"
	case DTypeBF16:
		return "bf16"
	case DTypeF32:
		return "f32"
	case DTypeF64:
		return "f64"
	default:
		return fmt.Sprintf("dtype(%d)", uint8(d))
	}
}

// SizeInBytes returns the byte size of a single element.
func (d DType) SizeInBytes() int {
	switch d {
	case DTypeBool, DTypeI8, DTypeU8:
		return 1
	case DTypeI16, DTypeF16, DTypeBF16:
		return 2
	case DTypeI32, DTypeF32:
		return 4
	case DTypeI64, DTypeF64:
		return 8
	default:
		return 0
	}
}

// Error mirrors SafeTensorsError.
type Error struct{ Message string }

func (e *Error) Error() string { return "safetensors: " + e.Message }

// TensorInfo describes a single tensor entry in a SafeTensors file.
type TensorInfo struct {
	Name           string
	Shape          []int
	DType          DType
	AbsoluteOffset int
	SizeBytes      int
}

// MappedFile is the result of opening a SafeTensors file.
type MappedFile struct {
	Path    string
	bytes   []byte
	tensors []TensorInfo
}

// Load reads a SafeTensors file from disk.
func Load(path string) (*MappedFile, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, &Error{Message: err.Error()}
	}
	return Parse(raw)
}

// LoadReader reads a SafeTensors file from a reader.
func LoadReader(r io.Reader) (*MappedFile, error) {
	raw, err := io.ReadAll(r)
	if err != nil {
		return nil, &Error{Message: err.Error()}
	}
	return Parse(raw)
}

// Parse parses an in-memory SafeTensors file.
func Parse(raw []byte) (*MappedFile, error) {
	if len(raw) < 8 {
		return nil, &Error{Message: "file too small"}
	}
	headerLen := binary.LittleEndian.Uint64(raw[0:8])
	if uint64(len(raw)) < 8+headerLen {
		return nil, &Error{Message: "header truncated"}
	}
	headerBytes := raw[8 : 8+headerLen]
	var header map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return nil, &Error{Message: "header json: " + err.Error()}
	}
	var infos []TensorInfo
	dataStart := int(8 + headerLen)
	cursor := dataStart
	for name, metaJSON := range header {
		if name == "__metadata__" {
			continue
		}
		var meta struct {
			DType   string `json:"dtype"`
			Shape   []int  `json:"shape"`
			Offsets [2]int `json:"data_offsets"`
		}
		if err := json.Unmarshal(metaJSON, &meta); err != nil {
			return nil, &Error{Message: "tensor " + name + ": " + err.Error()}
		}
		dt, err := parseDType(meta.DType)
		if err != nil {
			return nil, &Error{Message: "tensor " + name + ": " + err.Error()}
		}
		elements := 1
		for _, d := range meta.Shape {
			elements *= d
		}
		sizeBytes := elements * dt.SizeInBytes()
		absOffset := dataStart + meta.Offsets[0]
		if absOffset+sizeBytes > len(raw) {
			return nil, &Error{Message: "tensor " + name + " out of bounds"}
		}
		_ = cursor
		infos = append(infos, TensorInfo{
			Name:           name,
			Shape:          meta.Shape,
			DType:          dt,
			AbsoluteOffset: absOffset,
			SizeBytes:      sizeBytes,
		})
	}
	return &MappedFile{bytes: raw, tensors: infos}, nil
}

// Tensors returns the per-tensor information.
func (m *MappedFile) Tensors() []TensorInfo { return m.tensors }

// Bytes returns the full raw file bytes.
func (m *MappedFile) Bytes() []byte { return m.bytes }

// TensorData returns a copy of the bytes for a single tensor.
func (m *MappedFile) TensorData(name string) ([]byte, error) {
	for _, info := range m.tensors {
		if info.Name == name {
			end := info.AbsoluteOffset + info.SizeBytes
			if end > len(m.bytes) {
				return nil, &Error{Message: name + " out of bounds"}
			}
			out := make([]byte, info.SizeBytes)
			copy(out, m.bytes[info.AbsoluteOffset:end])
			return out, nil
		}
	}
	return nil, &Error{Message: "tensor not found: " + name}
}

func parseDType(s string) (DType, error) {
	switch s {
	case "bool":
		return DTypeBool, nil
	case "i8":
		return DTypeI8, nil
	case "i16":
		return DTypeI16, nil
	case "i32":
		return DTypeI32, nil
	case "i64":
		return DTypeI64, nil
	case "u8":
		return DTypeU8, nil
	case "f16":
		return DTypeF16, nil
	case "bf16":
		return DTypeBF16, nil
	case "f32":
		return DTypeF32, nil
	case "f64":
		return DTypeF64, nil
	default:
		return DTypeUnknown, fmt.Errorf("unsupported dtype %q", s)
	}
}
