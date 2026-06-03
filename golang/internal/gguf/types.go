package gguf

type File struct {
	Version          uint32
	TensorCount      uint64
	Metadata         map[string]MetadataValue
	TensorInfos      []TensorInfo
	Alignment        uint64
	DataSectionStart uint64
}

type TensorInfo struct {
	Name           string
	Dimensions     []uint64
	GGMLType       uint32
	RelativeOffset uint64
	AbsoluteOffset uint64
}

type MetadataType uint32

const (
	MetadataUint8 MetadataType = iota
	MetadataInt8
	MetadataUint16
	MetadataInt16
	MetadataUint32
	MetadataInt32
	MetadataFloat32
	MetadataBool
	MetadataString
	MetadataArray
	MetadataUint64
	MetadataInt64
	MetadataFloat64
)

type MetadataValue struct {
	Type    MetadataType
	Uint64  uint64
	Int64   int64
	Float64 float64
	Bool    bool
	String  string
	Array   []MetadataValue
}

func (v MetadataValue) AsUint64() (uint64, bool) {
	switch v.Type {
	case MetadataUint8, MetadataUint16, MetadataUint32, MetadataUint64:
		return v.Uint64, true
	case MetadataInt8, MetadataInt16, MetadataInt32, MetadataInt64:
		if v.Int64 >= 0 {
			return uint64(v.Int64), true
		}
	}
	return 0, false
}
