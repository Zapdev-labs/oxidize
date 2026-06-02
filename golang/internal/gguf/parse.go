package gguf

import "os"

const defaultAlignment uint64 = 32

func LoadFile(path string) (File, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return File{}, err
	}
	return Parse(raw)
}

func Parse(raw []byte) (File, error) {
	r := newReader(raw)
	magic, err := r.readExact(4)
	if err != nil {
		return File{}, err
	}
	if string(magic) != "GGUF" {
		return File{}, errInvalidMagic()
	}
	version, err := r.readU32()
	if err != nil {
		return File{}, err
	}
	if version != 2 && version != 3 {
		return File{}, errUnsupportedVersion(version)
	}
	tensorCount, err := r.readU64()
	if err != nil {
		return File{}, err
	}
	metadataCount, err := r.readU64()
	if err != nil {
		return File{}, err
	}
	metadata := make(map[string]MetadataValue, metadataCount)
	for range make([]struct{}, metadataCount) {
		key, readErr := r.readString()
		if readErr != nil {
			return File{}, readErr
		}
		valueType, readErr := r.readU32()
		if readErr != nil {
			return File{}, readErr
		}
		value, readErr := r.readValue(MetadataType(valueType))
		if readErr != nil {
			return File{}, readErr
		}
		metadata[key] = value
	}
	tensors := make([]TensorInfo, 0, tensorCount)
	for range make([]struct{}, tensorCount) {
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
	for index := range tensors {
		tensors[index].AbsoluteOffset = dataStart + tensors[index].RelativeOffset
		if tensors[index].AbsoluteOffset > uint64(len(raw)) {
			return File{}, errUnexpectedEOF()
		}
	}
	return File{Version: version, TensorCount: tensorCount, Metadata: metadata, TensorInfos: tensors, Alignment: alignment, DataSectionStart: dataStart}, nil
}

func alignUp(value uint64, alignment uint64) (uint64, error) {
	mask := alignment - 1
	sum := value + mask
	if sum < value {
		return 0, errIntegerOverflow()
	}
	return sum &^ mask, nil
}
