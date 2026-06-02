package simd

import "testing"

func TestLaneWidthF32(t *testing.T) {
	cases := map[Backend]int{
		BackendScalar: 1, BackendSse2: 4, BackendAvx: 8,
		BackendAvx2: 8, BackendAvx512f: 16, BackendNeon: 4,
	}
	for b, want := range cases {
		if got := b.LaneWidthF32(); got != want {
			t.Fatalf("%v lane = %d, want %d", b, got, want)
		}
	}
}

func TestPreferredNonEmpty(t *testing.T) {
	if Preferred() == Backend(255) {
		t.Fatal("preferred should be set")
	}
	avail := Available()
	if len(avail) == 0 {
		t.Fatal("available should not be empty")
	}
}
