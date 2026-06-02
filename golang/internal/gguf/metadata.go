package gguf

func (r *reader) readTensor() (TensorInfo, error) {
	name, err := r.readString()
	if err != nil {
		return TensorInfo{}, err
	}
	dimensionCount, err := r.readU32()
	if err != nil {
		return TensorInfo{}, err
	}
	dimensions := make([]uint64, 0, dimensionCount)
	for range make([]struct{}, dimensionCount) {
		value, readErr := r.readU64()
		if readErr != nil {
			return TensorInfo{}, readErr
		}
		dimensions = append(dimensions, value)
	}
	ggmlType, err := r.readU32()
	if err != nil {
		return TensorInfo{}, err
	}
	relativeOffset, err := r.readU64()
	if err != nil {
		return TensorInfo{}, err
	}
	return TensorInfo{Name: name, Dimensions: dimensions, GGMLType: ggmlType, RelativeOffset: relativeOffset}, nil
}

func (r *reader) readValue(kind MetadataType) (MetadataValue, error) {
	switch kind {
	case MetadataUint8:
		value, err := r.readU8()
		return MetadataValue{Type: kind, Uint64: uint64(value)}, err
	case MetadataInt8:
		value, err := r.readI8()
		return MetadataValue{Type: kind, Int64: int64(value)}, err
	case MetadataUint16:
		value, err := r.readU16()
		return MetadataValue{Type: kind, Uint64: uint64(value)}, err
	case MetadataInt16:
		value, err := r.readI16()
		return MetadataValue{Type: kind, Int64: int64(value)}, err
	case MetadataUint32:
		value, err := r.readU32()
		return MetadataValue{Type: kind, Uint64: uint64(value)}, err
	case MetadataInt32:
		value, err := r.readI32()
		return MetadataValue{Type: kind, Int64: int64(value)}, err
	case MetadataFloat32:
		value, err := r.readF32()
		return MetadataValue{Type: kind, Float64: float64(value)}, err
	case MetadataBool:
		value, err := r.readU8()
		return MetadataValue{Type: kind, Bool: value != 0}, err
	case MetadataString:
		value, err := r.readString()
		return MetadataValue{Type: kind, String: value}, err
	case MetadataArray:
		elementType, err := r.readU32()
		if err != nil {
			return MetadataValue{}, err
		}
		length, err := r.readU64()
		if err != nil {
			return MetadataValue{}, err
		}
		values := make([]MetadataValue, 0, length)
		for range make([]struct{}, length) {
			value, readErr := r.readValue(MetadataType(elementType))
			if readErr != nil {
				return MetadataValue{}, readErr
			}
			values = append(values, value)
		}
		return MetadataValue{Type: kind, Array: values}, nil
	case MetadataUint64:
		value, err := r.readU64()
		return MetadataValue{Type: kind, Uint64: value}, err
	case MetadataInt64:
		value, err := r.readI64()
		return MetadataValue{Type: kind, Int64: value}, err
	case MetadataFloat64:
		value, err := r.readF64()
		return MetadataValue{Type: kind, Float64: value}, err
	default:
		return MetadataValue{}, errUnknownMetadataType(uint32(kind))
	}
}
