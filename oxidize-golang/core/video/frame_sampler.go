package video

import "sort"

// SampleIndices picks frame indices from [0, totalFrames) using strategy.
func SampleIndices(totalFrames, targetFrames int, strategy FrameSamplingStrategy) ([]int, error) {
	if totalFrames <= 0 || targetFrames <= 0 {
		return nil, ErrFrameCountOutRange
	}
	var indices []int
	switch strategy {
	case SampleDense:
		indices = dense(totalFrames, targetFrames, 1)
	default:
		indices = uniform(totalFrames, targetFrames)
	}
	if len(indices) == 0 {
		return nil, ErrEmptySample
	}
	return indices, nil
}

// LumaHistogramRGB builds a 16-bin normalized luma histogram for an RGB frame.
func LumaHistogramRGB(data []byte) []float32 {
	hist := make([]float32, 16)
	if len(data) == 0 {
		return hist
	}
	var total float32
	for i := 0; i+2 < len(data); i += 3 {
		luma := 0.299*float32(data[i]) + 0.587*float32(data[i+1]) + 0.114*float32(data[i+2])
		bin := int(luma / 16)
		if bin > 15 {
			bin = 15
		}
		hist[bin]++
		total++
	}
	if total > 0 {
		for i := range hist {
			hist[i] /= total
		}
	}
	return hist
}

// SampleIndicesAdaptive keeps first/last frames and fills remaining slots by
// histogram distance. Falls back to uniform when lumaHists is too short.
func SampleIndicesAdaptive(totalFrames, targetFrames int, lumaHists []float32) ([]int, error) {
	if totalFrames <= 0 || targetFrames <= 0 {
		return nil, ErrFrameCountOutRange
	}
	if len(lumaHists) < totalFrames*16 {
		return SampleIndices(totalFrames, targetFrames, SampleAdaptive)
	}
	if totalFrames <= targetFrames {
		out := make([]int, totalFrames)
		for i := range out {
			out[i] = i
		}
		return out, nil
	}
	chosen := map[int]struct{}{0: {}, totalFrames - 1: {}}
	out := []int{0, totalFrames - 1}
	for len(out) < targetFrames {
		bestIdx := -1
		var bestScore float32
		for cand := 0; cand < totalFrames; cand++ {
			if _, ok := chosen[cand]; ok {
				continue
			}
			score := minHistDistance(cand, out, lumaHists)
			if bestIdx < 0 || score > bestScore {
				bestIdx = cand
				bestScore = score
			}
		}
		if bestIdx < 0 {
			break
		}
		chosen[bestIdx] = struct{}{}
		out = append(out, bestIdx)
	}
	sort.Ints(out)
	if len(out) == 0 {
		return nil, ErrEmptySample
	}
	return out, nil
}

func uniform(total, target int) []int {
	if total <= target {
		out := make([]int, total)
		for i := range out {
			out[i] = i
		}
		return out
	}
	step := float64(total-1) / float64(target-1)
	out := make([]int, 0, target)
	seen := map[int]struct{}{}
	for i := 0; i < target; i++ {
		idx := int(float64(i)*step + 0.5)
		if idx >= total {
			idx = total - 1
		}
		if _, ok := seen[idx]; !ok {
			seen[idx] = struct{}{}
			out = append(out, idx)
		}
	}
	sort.Ints(out)
	return out
}

func dense(total, target, stride int) []int {
	if stride <= 0 {
		stride = 1
	}
	out := make([]int, 0, target)
	for i := 0; i < total && len(out) < target; i += stride {
		out = append(out, i)
	}
	return out
}

func minHistDistance(cand int, chosen []int, hists []float32) float32 {
	candHist := hists[cand*16 : (cand+1)*16]
	var best float32
	for _, idx := range chosen {
		other := hists[idx*16 : (idx+1)*16]
		d := l1(candHist, other)
		if best == 0 || d < best {
			best = d
		}
	}
	return best
}

func l1(a, b []float32) float32 {
	var s float32
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		s += d
	}
	return s
}
