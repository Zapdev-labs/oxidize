// Package prune implements magnitude pruning for dense weight matrices.
package prune

import (
	"fmt"
	"math"
	"sort"
)

// Options controls magnitude pruning.
type Options struct {
	Sparsity float32
}

// Report summarizes a prune run.
type Report struct {
	PrunedRows int
	Kept       int
	Pruned     int
}

// MagnitudeMask returns a keep-mask for row-major weights [rows, cols].
func MagnitudeMask(weights []float32, rows, cols int, sparsity float32) ([]bool, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("prune: invalid dims rows=%d cols=%d", rows, cols)
	}
	if len(weights) < rows*cols {
		return nil, fmt.Errorf("prune: weights too small")
	}
	if sparsity < 0 || sparsity >= 1 {
		return nil, fmt.Errorf("prune: sparsity out of range")
	}
	keepPerRow := int(math.Round(float64(cols) * float64(1-sparsity)))
	if keepPerRow <= 0 {
		keepPerRow = 1
	}
	if keepPerRow > cols {
		keepPerRow = cols
	}
	mask := make([]bool, rows*cols)
	for r := 0; r < rows; r++ {
		start := r * cols
		row := weights[start : start+cols]
		type idxScore struct {
			i int
			v float32
		}
		scores := make([]idxScore, cols)
		for i, v := range row {
			av := v
			if av < 0 {
				av = -av
			}
			scores[i] = idxScore{i: i, v: av}
		}
		sort.Slice(scores, func(i, j int) bool { return scores[i].v > scores[j].v })
		for k := 0; k < keepPerRow; k++ {
			mask[start+scores[k].i] = true
		}
	}
	return mask, nil
}

// ApplyMaskInPlace zeroes pruned entries in weights.
func ApplyMaskInPlace(weights []float32, mask []bool) {
	for i := range weights {
		if i < len(mask) && !mask[i] {
			weights[i] = 0
		}
	}
}

// MagnitudePrune applies per-row magnitude pruning in place.
func MagnitudePrune(weights []float32, rows, cols int, opts Options) (Report, error) {
	mask, err := MagnitudeMask(weights, rows, cols, opts.Sparsity)
	if err != nil {
		return Report{}, err
	}
	kept, pruned := 0, 0
	for i := range mask {
		if mask[i] {
			kept++
		} else {
			pruned++
		}
	}
	ApplyMaskInPlace(weights, mask)
	return Report{PrunedRows: rows, Kept: kept, Pruned: pruned}, nil
}
