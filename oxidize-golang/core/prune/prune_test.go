package prune

import "testing"

func TestMagnitudePrune(t *testing.T) {
	weights := []float32{0, 1, 2, 3, 4, 5, 6, 7}
	rep, err := MagnitudePrune(weights, 2, 4, Options{Sparsity: 0.5})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Kept != 4 || rep.Pruned != 4 {
		t.Fatalf("unexpected report: %+v", rep)
	}
	if weights[0] != 0 || weights[3] != 3 {
		t.Fatalf("expected top magnitudes kept in row0, got %v", weights[:4])
	}
}
