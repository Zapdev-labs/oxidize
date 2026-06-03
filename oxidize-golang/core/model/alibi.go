package model

import "math"

// AlibiSlopes returns per-head slopes for ALiBi (Press et al.), matching oxidize-core mlx.rs.
func AlibiSlopes(numHeads int) []float32 {
	if numHeads <= 0 {
		return nil
	}
	out := make([]float32, numHeads)
	denom := float32(numHeads)
	for h := 0; h < numHeads; h++ {
		exp := -8.0 / denom * float32(h+1)
		out[h] = -float32(math.Pow(2, float64(exp)))
	}
	return out
}
