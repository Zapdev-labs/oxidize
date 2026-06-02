package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"testing"
)

func TestParseSimple(t *testing.T) {
	// Build a minimal SafeTensors file: 1 f32 tensor of shape [2,2].
	payload := []float32{1, 2, 3, 4}
	tensorBytes := make([]byte, len(payload)*4)
	for i, v := range payload {
		binary.LittleEndian.PutUint32(tensorBytes[i*4:], math.Float32bits(v))
	}
	header := map[string]any{
		"a": map[string]any{
			"dtype":   "f32",
			"shape":   []int{2, 2},
			"data_offsets": [2]int{0, len(tensorBytes)},
		},
	}
	hb, _ := json.Marshal(header)
	raw := make([]byte, 8+len(hb)+len(tensorBytes))
	binary.LittleEndian.PutUint64(raw[0:8], uint64(len(hb)))
	copy(raw[8:], hb)
	copy(raw[8+len(hb):], tensorBytes)
	m, err := Parse(raw)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(m.Tensors()) != 1 {
		t.Fatalf("tensors = %d", len(m.Tensors()))
	}
	data, err := m.TensorData("a")
	if err != nil {
		t.Fatalf("tensor data: %v", err)
	}
	if len(data) != 16 {
		t.Fatalf("data len = %d", len(data))
	}
}
