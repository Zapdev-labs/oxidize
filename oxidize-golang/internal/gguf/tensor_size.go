package gguf

// TensorStorageBytes returns the on-disk payload size for a tensor header.
func TensorStorageBytes(info TensorInfo) (uint64, error) {
	count, err := tensorElementCount(info.Dimensions)
	if err != nil {
		return 0, err
	}
	return tensorByteSize(info.GGMLType, count)
}

func tensorElementCount(dimensions []uint64) (uint64, error) {
	count := uint64(1)
	for _, dimension := range dimensions {
		if dimension == 0 {
			return 0, nil
		}
		if count > ^uint64(0)/dimension {
			return 0, errIntegerOverflow()
		}
		count *= dimension
	}
	return count, nil
}

func tensorByteSize(ggmlType uint32, elementCount uint64) (uint64, error) {
	if elementCount == 0 {
		return 0, nil
	}
	switch ggmlType {
	case 0:
		return multiplyUint64(elementCount, 4)
	case 1:
		return multiplyUint64(elementCount, 2)
	case 2:
		return quantizedTensorSize(elementCount, 32, 18)
	case 3:
		return quantizedTensorSize(elementCount, 32, 20)
	case 6:
		return quantizedTensorSize(elementCount, 32, 22)
	case 7:
		return quantizedTensorSize(elementCount, 32, 24)
	case 8:
		return quantizedTensorSize(elementCount, 32, 34)
	case 10:
		return quantizedTensorSize(elementCount, 256, 84)
	case 11:
		return quantizedTensorSize(elementCount, 256, 110)
	case 12:
		return quantizedTensorSize(elementCount, 256, 144)
	case 13:
		return quantizedTensorSize(elementCount, 256, 176)
	case 14:
		return quantizedTensorSize(elementCount, 256, 210)
	case 15:
		return quantizedTensorSize(elementCount, 256, 66)
	case 16:
		return quantizedTensorSize(elementCount, 256, 74)
	case 17:
		return quantizedTensorSize(elementCount, 256, 98)
	case 18:
		return quantizedTensorSize(elementCount, 256, 50)
	case 19:
		return quantizedTensorSize(elementCount, 256, 18)
	case 20:
		return quantizedTensorSize(elementCount, 256, 110)
	case 21:
		return quantizedTensorSize(elementCount, 256, 82)
	case 22:
		return quantizedTensorSize(elementCount, 256, 34)
	case 23:
		return quantizedTensorSize(elementCount, 256, 56)
	case 24:
		return quantizedTensorSize(elementCount, 64, 34)
	case 29:
		return quantizedTensorSize(elementCount, 256, 56)
	case 40:
		return quantizedTensorSize(elementCount, 64, 34)
	default:
		return 0, errUnknownGGMLType(ggmlType)
	}
}

func quantizedTensorSize(elementCount, blockSize, bytesPerBlock uint64) (uint64, error) {
	blocks := elementCount / blockSize
	if elementCount%blockSize != 0 {
		blocks++
	}
	return multiplyUint64(blocks, bytesPerBlock)
}

func multiplyUint64(left, right uint64) (uint64, error) {
	if left == 0 || right == 0 {
		return 0, nil
	}
	if left > ^uint64(0)/right {
		return 0, errIntegerOverflow()
	}
	return left * right, nil
}
