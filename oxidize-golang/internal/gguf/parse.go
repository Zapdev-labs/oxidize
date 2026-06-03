package gguf

import (
	"bytes"
	"os"
)

const defaultAlignment uint64 = 32

type Header struct {
	Version  uint32
	Metadata map[string]MetadataValue
}

func LoadFile(path string) (File, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return File{}, err
	}
	return Parse(raw)
}

func LoadMetadata(path string) (Header, error) {
	file, err := os.Open(path)
	if err != nil {
		return Header{}, err
	}
	defer func() { _ = file.Close() }()

	r := newReader(file)
	version, _, metadata, err := parseHeader(r)
	if err != nil {
		return Header{}, err
	}
	return Header{Version: version, Metadata: metadata}, nil
}

func Parse(raw []byte) (File, error) {
	r := newReader(bytes.NewReader(raw))
	version, tensorCount, metadata, err := parseHeader(r)
	if err != nil {
		return File{}, err
	}
	tensors := make([]TensorInfo, 0, tensorCount)
	for range tensorCount {
		tensor, readErr := r.readTensor()
		if readErr != nil {
			return File{}, readErr
		}
		tensors = append(tensors, tensor)
	}
	alignment := defaultAlignment
	if value, ok := metadata["general.alignment"]; ok {
		number, ok := value.AsUint64()
		if !ok || number == 0 || number&(number-1) != 0 {
			return File{}, errInvalidAlignment(number)
		}
		alignment = number
	}
	dataStart, err := alignUp(uint64(r.position()), alignment)
	if err != nil {
		return File{}, err
	}
	if dataStart > uint64(len(raw)) {
		return File{}, errUnexpectedEOF()
	}
	limit := uint64(len(raw))
	for index := range tensors {
		absOff := dataStart + tensors[index].RelativeOffset
		if absOff < dataStart {
			return File{}, errIntegerOverflow()
		}
		elementCount, err := tensorElementCount(tensors[index].Dimensions)
		if err != nil {
			return File{}, err
		}
		byteSize, err := tensorByteSize(tensors[index].GGMLType, elementCount)
		if err != nil {
			return File{}, err
		}
		if absOff > limit {
			return File{}, errUnexpectedEOF()
		}
		if byteSize > limit-absOff {
			return File{}, errUnexpectedEOF()
		}
		tensors[index].AbsoluteOffset = absOff
	}
	return File{Version: version, TensorCount: tensorCount, Metadata: metadata, TensorInfos: tensors, Alignment: alignment, DataSectionStart: dataStart}, nil
}

func parseHeader(r *reader) (uint32, uint64, map[string]MetadataValue, error) {
	magic, err := r.readExact(4)
	if err != nil {
		return 0, 0, nil, err
	}
	if string(magic) != "GGUF" {
		return 0, 0, nil, errInvalidMagic()
	}
	version, err := r.readU32()
	if err != nil {
		return 0, 0, nil, err
	}
	if version != 2 && version != 3 {
		return 0, 0, nil, errUnsupportedVersion(version)
	}
	tensorCount, err := r.readU64()
	if err != nil {
		return 0, 0, nil, err
	}
	metadataCount, err := r.readU64()
	if err != nil {
		return 0, 0, nil, err
	}
	metadata := make(map[string]MetadataValue, metadataCount)
	for range metadataCount {
		key, readErr := r.readString()
		if readErr != nil {
			return 0, 0, nil, readErr
		}
		valueType, readErr := r.readU32()
		if readErr != nil {
			return 0, 0, nil, readErr
		}
		value, readErr := r.readValue(MetadataType(valueType))
		if readErr != nil {
			return 0, 0, nil, readErr
		}
		metadata[key] = value
	}
	return version, tensorCount, metadata, nil
}

func alignUp(value uint64, alignment uint64) (uint64, error) {
	mask := alignment - 1
	sum := value + mask
	if sum < value {
		return 0, errIntegerOverflow()
	}
	return sum &^ mask, nil
}
