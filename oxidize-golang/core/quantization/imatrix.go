package quantization

// IMatrix is a per-tensor importance matrix used to guide mixed-precision
// quantization. It mirrors `IMatrix` in oxidize-core/src/compute/quantization.rs.
type IMatrix struct {
	values []float32
}

// NewIMatrix constructs an IMatrix from a slice of importance values.
func NewIMatrix(values []float32) *IMatrix {
	return &IMatrix{values: append([]float32(nil), values...)}
}

// FromValues is an alias for NewIMatrix.
func (m *IMatrix) FromValues(values []float32) *IMatrix { return NewIMatrix(values) }

// Values returns a copy of the importance values.
func (m *IMatrix) Values() []float32 {
	return append([]float32(nil), m.values...)
}

// ImportanceAt returns the importance for a given linear index, or 1.0 if
// the matrix is shorter than the index.
func (m *IMatrix) ImportanceAt(index int) float32 {
	if m == nil || index >= len(m.values) {
		return 1.0
	}
	if m.values[index] <= 0 {
		return 1.0
	}
	return m.values[index]
}

// MixedLayerPlan describes a single layer's target quantization for a mixed
// quantize pass. Mirrors MixedLayerPlan.
type MixedLayerPlan struct {
	Name        string
	ValueCount  int
	Target      Type
}

// QuantizedLayer is the output of quantize_mixed_scalar.
type QuantizedLayer struct {
	Name   string
	Target Type
	Bytes  []byte
}

// QuantizeMixedScalar quantizes a batch of (name, values) pairs using the
// plans in `plans` and returns the resulting QuantizedLayer slice.
func QuantizeMixedScalar(plans []MixedLayerPlan, tensors map[string][]float32) ([]QuantizedLayer, error) {
	out := make([]QuantizedLayer, 0, len(plans))
	for _, p := range plans {
		src, ok := tensors[p.Name]
		if !ok {
			return nil, &Error{Message: "missing tensor " + p.Name}
		}
		if len(src) != p.ValueCount {
			return nil, &Error{Message: "value count mismatch for " + p.Name}
		}
		size, err := QuantizedSize(p.Target, p.ValueCount)
		if err != nil {
			return nil, err
		}
		buf := make([]byte, size)
		if err := QuantizeScalar(p.Target, src, buf, nil); err != nil {
			return nil, err
		}
		out = append(out, QuantizedLayer{Name: p.Name, Target: p.Target, Bytes: buf})
	}
	return out, nil
}
