package gguf

import (
	"bytes"
	"encoding/binary"
	"fmt"
)

// WriterHeader describes a complete GGUF file to be encoded.
type WriterHeader struct {
	Version          uint32
	TensorCount      uint64
	MetadataCount    uint64
	Metadata         map[string]MetadataValue
	Tensors          []TensorInfo
	Alignment        uint64
	DataSectionStart uint64
}

// Encode serializes the given GGUF header and returns the final file bytes
// consisting of the header followed by `body` (which is assumed to be already
// aligned to DataSectionStart).
func Encode(h WriterHeader, body []byte) ([]byte, error) {
	if h.Version == 0 {
		h.Version = 3
	}
	if h.Alignment == 0 {
		h.Alignment = 32
	}
	if h.MetadataCount == 0 {
		h.MetadataCount = uint64(len(h.Metadata))
	}
	if h.TensorCount == 0 {
		h.TensorCount = uint64(len(h.Tensors))
	}
	var buf bytes.Buffer
	// Magic
	buf.WriteString("GGUF")
	// Version
	if err := binary.Write(&buf, binary.LittleEndian, h.Version); err != nil {
		return nil, err
	}
	// Tensor count
	if err := binary.Write(&buf, binary.LittleEndian, h.TensorCount); err != nil {
		return nil, err
	}
	// Metadata count
	if err := binary.Write(&buf, binary.LittleEndian, h.MetadataCount); err != nil {
		return nil, err
	}
	// Metadata
	for k, v := range h.Metadata {
		if err := writeString(&buf, k); err != nil {
			return nil, err
		}
		if err := writeValue(&buf, v); err != nil {
			return nil, err
		}
	}
	// Tensor infos
	for _, t := range h.Tensors {
		if err := writeString(&buf, t.Name); err != nil {
			return nil, err
		}
		if err := binary.Write(&buf, binary.LittleEndian, uint32(len(t.Dimensions))); err != nil {
			return nil, err
		}
		for _, d := range t.Dimensions {
			if err := binary.Write(&buf, binary.LittleEndian, d); err != nil {
				return nil, err
			}
		}
		if err := binary.Write(&buf, binary.LittleEndian, t.GGMLType); err != nil {
			return nil, err
		}
		if err := binary.Write(&buf, binary.LittleEndian, t.RelativeOffset); err != nil {
			return nil, err
		}
	}
	// Pad to data section start
	headerEnd := uint64(buf.Len())
	dataStart, err := alignUp(headerEnd, h.Alignment)
	if err != nil {
		return nil, err
	}
	for uint64(buf.Len()) < dataStart {
		buf.WriteByte(0)
	}
	if uint64(buf.Len()) != dataStart {
		return nil, fmt.Errorf("alignment mismatch")
	}
	// Tensor relative offsets must be relative to data section start
	for i := range h.Tensors {
		h.Tensors[i].RelativeOffset = 0
	}
	buf.Write(body)
	return buf.Bytes(), nil
}

func writeString(buf *bytes.Buffer, s string) error {
	if err := binary.Write(buf, binary.LittleEndian, uint64(len(s))); err != nil {
		return err
	}
	_, err := buf.WriteString(s)
	return err
}

func writeValue(buf *bytes.Buffer, v MetadataValue) error {
	if err := binary.Write(buf, binary.LittleEndian, uint32(v.Type)); err != nil {
		return err
	}
	return writeValueBody(buf, v)
}

func writeValueBody(buf *bytes.Buffer, v MetadataValue) error {
	switch v.Type {
	case MetadataUint8:
		return binary.Write(buf, binary.LittleEndian, uint8(v.Uint64))
	case MetadataInt8:
		return binary.Write(buf, binary.LittleEndian, int8(v.Int64))
	case MetadataUint16:
		return binary.Write(buf, binary.LittleEndian, uint16(v.Uint64))
	case MetadataInt16:
		return binary.Write(buf, binary.LittleEndian, int16(v.Int64))
	case MetadataUint32:
		return binary.Write(buf, binary.LittleEndian, uint32(v.Uint64))
	case MetadataInt32:
		return binary.Write(buf, binary.LittleEndian, int32(v.Int64))
	case MetadataFloat32:
		return binary.Write(buf, binary.LittleEndian, float32(v.Float64))
	case MetadataBool:
		var b uint8
		if v.Bool {
			b = 1
		}
		return binary.Write(buf, binary.LittleEndian, b)
	case MetadataString:
		return writeString(buf, v.String)
	case MetadataArray:
		if len(v.Array) == 0 {
			// write dummy element type
			if err := binary.Write(buf, binary.LittleEndian, uint32(MetadataUint8)); err != nil {
				return err
			}
			return binary.Write(buf, binary.LittleEndian, uint64(0))
		}
		et := uint32(v.Array[0].Type)
		if err := binary.Write(buf, binary.LittleEndian, et); err != nil {
			return err
		}
		if err := binary.Write(buf, binary.LittleEndian, uint64(len(v.Array))); err != nil {
			return err
		}
		for _, e := range v.Array {
			if err := writeValueBody(buf, e); err != nil {
				return err
			}
		}
		return nil
	case MetadataUint64:
		return binary.Write(buf, binary.LittleEndian, v.Uint64)
	case MetadataInt64:
		return binary.Write(buf, binary.LittleEndian, v.Int64)
	case MetadataFloat64:
		return binary.Write(buf, binary.LittleEndian, v.Float64)
	default:
		return fmt.Errorf("unsupported metadata type %v", v.Type)
	}
}

// alignUp is provided by parse.go
