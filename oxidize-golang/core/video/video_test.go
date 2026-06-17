package video

import "testing"

func TestRawFrameDecoder(t *testing.T) {
	frame, err := NewDecodedFrame(2, 2, make([]byte, 12))
	if err != nil {
		t.Fatal(err)
	}
	dec := RawFrameDecoder{}
	out, err := dec.Decode(VideoSource{SingleImage: frame})
	if err != nil || len(out) != 1 {
		t.Fatalf("decode: %v len=%d", err, len(out))
	}
}

func TestSampleIndicesUniform(t *testing.T) {
	idx, err := SampleIndices(100, 8, SampleUniform)
	if err != nil {
		t.Fatal(err)
	}
	if len(idx) != 8 {
		t.Fatalf("expected 8 indices, got %d", len(idx))
	}
}

func TestVideoPromptBuildSequence(t *testing.T) {
	table := make([]float32, 4*2)
	for i := range table {
		table[i] = float32(i)
	}
	p := NewVideoPrompt()
	p.AddText([]uint32{0, 1})
	out, err := p.BuildSequence(table, 4, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 4 {
		t.Fatalf("expected 4 floats, got %d", len(out))
	}
}
