package oxk

import (
	"math"
	"os"
	"strings"
	"sync"
	"testing"
)

// Deterministic pseudo-random byte stream (xorshift), no rand dep.
func fillPseudo(bytes []byte, state uint64) uint64 {
	for i := range bytes {
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17
		bytes[i] = byte(state)
	}
	return state
}

func randomFixture(rows, blocksPerRow int, seed uint64) ([]byte, []byte) {
	weights := make([]byte, rows*blocksPerRow*BLOCK_Q4_K_SIZE)
	fillPseudo(weights, seed)
	// Keep f16 d/dmin fields finite and small: rewrite each block header
	// with exponents well inside the f16 normal range.
	for blockIdx := 0; blockIdx < len(weights)/BLOCK_Q4_K_SIZE; blockIdx++ {
		block := weights[blockIdx*BLOCK_Q4_K_SIZE:]
		for half := 0; half < 2; half++ {
			raw := uint16(block[half*2]) | uint16(block[half*2+1])<<8
			tamed := (raw & 0x83ff) | (0x3000 + ((raw >> 10) & 0x7) * 0x400)
			block[half*2] = byte(tamed)
			block[half*2+1] = byte(tamed >> 8)
		}
	}
	vectorBytes := make([]byte, blocksPerRow*QK_K)
	fillPseudo(vectorBytes, seed*0x9e37_79b9_7f4a_7c15)
	vector := make([]float32, blocksPerRow*QK_K)
	for i, b := range vectorBytes {
		vector[i] = (float32(b) - 127.5) / 32.0
	}
	q8k := make([]byte, blocksPerRow*BLOCK_Q8_K_BYTES)
	if err := QuantizeQ8KInto(vector, blocksPerRow, q8k); err != nil {
		panic(err)
	}
	return weights, q8k
}

func TestTileVariantsMatchScalarExactly(t *testing.T) {
	for _, tc := range []struct{ rows, bpr int }{
		{8, 16},
		{12, 4},
		{32, 8},
	} {
		weights, q8k := randomFixture(tc.rows, tc.bpr, 1)
		rowBytes := tc.bpr * BLOCK_Q4_K_SIZE
		scalar := make([]float32, tc.rows)
		for r := 0; r < tc.rows; r++ {
			row := weights[r*rowBytes : (r+1)*rowBytes]
			v, err := Q4kQ8kRowDotScalar(row, tc.bpr, q8k)
			if err != nil {
				t.Fatal(err)
			}
			scalar[r] = v
		}

		// x1
		for r := 0; r < tc.rows; r++ {
			row := weights[r*rowBytes : (r+1)*rowBytes]
			got, err := Q4kQ8kRowDotX1Scalar(row, tc.bpr, q8k)
			if err != nil {
				t.Fatal(err)
			}
			if math.Float32bits(got) != math.Float32bits(scalar[r]) {
				t.Errorf("x1 row %d mismatch: got %v want %v", r, got, scalar[r])
			}
		}

		// x4
		if tc.rows >= 4 {
			var quad [4]float32
			if err := Q4kQ8kRowDotX4Scalar(weights, rowBytes, tc.bpr, q8k, quad[:]); err != nil {
				t.Fatal(err)
			}
			for r := 0; r < 4; r++ {
				if math.Float32bits(quad[r]) != math.Float32bits(scalar[r]) {
					t.Errorf("x4 row %d mismatch: got %v want %v", r, quad[r], scalar[r])
				}
			}
		}

		// x8
		if tc.rows >= 8 {
			var octet [8]float32
			if err := Q4kQ8kRowDotX8Scalar(weights, rowBytes, tc.bpr, q8k, octet[:]); err != nil {
				t.Fatal(err)
			}
			for r := 0; r < 8; r++ {
				if math.Float32bits(octet[r]) != math.Float32bits(scalar[r]) {
					t.Errorf("x8 row %d mismatch: got %v want %v", r, octet[r], scalar[r])
				}
			}
		}

		// x16
		if tc.rows >= 16 {
			var hex [16]float32
			if err := Q4kQ8kRowDotX16Scalar(weights, rowBytes, tc.bpr, q8k, hex[:]); err != nil {
				t.Fatal(err)
			}
			for r := 0; r < 16; r++ {
				if math.Float32bits(hex[r]) != math.Float32bits(scalar[r]) {
					t.Errorf("x16 row %d mismatch: got %v want %v", r, hex[r], scalar[r])
				}
			}
		}
	}
}

func TestGemvRangeMatchesScalar(t *testing.T) {
	// 13 rows exercises the x8 + x4 + x1 tail split.
	weights, q8k := randomFixture(13, 8, 7)
	rowBytes := 8 * BLOCK_Q4_K_SIZE
	out := make([]float32, 13)
	if err := GemvQ4kRange(weights, 8, q8k, out); err != nil {
		t.Fatal(err)
	}
	for r := 0; r < 13; r++ {
		row := weights[r*rowBytes : (r+1)*rowBytes]
		want, err := Q4kQ8kRowDotScalar(row, 8, q8k)
		if err != nil {
			t.Fatal(err)
		}
		if math.Float32bits(out[r]) != math.Float32bits(want) {
			t.Errorf("row %d mismatch: got %v want %v", r, out[r], want)
		}
	}
}

func TestMaxTileDefaultIs16(t *testing.T) {
	os.Unsetenv("OXIDIZE_OXK_TILE")
	// Reset the once so we re-read
	maxTileOnce = sync.Once{}
	maxTileVal = 0
	if MaxTile() != 16 {
		t.Errorf("expected default tile 16, got %d", MaxTile())
	}
}

func TestMaxTileEnvOverride(t *testing.T) {
	os.Setenv("OXIDIZE_OXK_TILE", "8")
	maxTileOnce = sync.Once{}
	maxTileVal = 0
	if MaxTile() != 8 {
		t.Errorf("expected tile 8, got %d", MaxTile())
	}
	os.Unsetenv("OXIDIZE_OXK_TILE")
	maxTileOnce = sync.Once{}
	maxTileVal = 0
}

func TestTuneIsBlockAlignedAndStable(t *testing.T) {
	t1 := Tune()
	if t1.PfBytes%BLOCK_Q4_K_SIZE != 0 {
		t.Errorf("pf_bytes not block aligned: %d", t1.PfBytes)
	}
	t2 := Tune()
	if t1.PfBytes != t2.PfBytes || t1.PfNta != t2.PfNta {
		t.Error("tune not stable")
	}
}

func TestSummaryMentionsVendor(t *testing.T) {
	s := OxkCpuSummary()
	if !strings.Contains(s, "vendor=") {
		t.Errorf("summary missing vendor: %s", s)
	}
}

func TestCpuInfoIsStable(t *testing.T) {
	a := GetCpuInfo()
	b := GetCpuInfo()
	if a.Family != b.Family || a.Model != b.Model {
		t.Error("cpuinfo not stable")
	}
}
