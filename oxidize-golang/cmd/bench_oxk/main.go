package main

import (
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/Zapdev-labs/oxidize/golang/core/oxk"
)

// Full decode-token GEMV plan for Qwen3-30B-A3B, so the reported tok/s reflects
// one real token's worth of work (every attention + MoE projection across all
// layers) instead of a single isolated GEMV.
//
// Canonical config (Qwen3-30B-A3B):
//
//	hidden_size           = 2048
//	num_hidden_layers     = 48
//	q_proj out            = 32 heads * 128 head_dim = 4096
//	kv_proj out           = 4  kv_heads * 128       = 512
//	num_experts           = 128, num_experts_per_tok = 8
//	moe_intermediate_size = 768
//	vocab_size            = 151936
//
// Only the active experts (8) are streamed per token, matching real MoE decode.
// All `cols` are multiples of QK_K (256) as required by the OXK kernel.
//
// Keep this plan in sync with bench_oxk.py and oxk_token_bench.rs.

const (
	hiddenSize = 2048
	numLayers  = 48
	qProjOut   = 4096
	kvProjOut  = 512
	numExperts = 8 // active per token
	moeInter   = 768
	routerOut  = 128
	vocabSize  = 151936
)

type gemvOp struct {
	label string
	rows  int
	cols  int
	count int // instances per token
}

// oneLayerPlan returns the GEMVs executed for a single transformer layer.
func oneLayerPlan() []gemvOp {
	return []gemvOp{
		{"attn.q", qProjOut, hiddenSize, 1},
		{"attn.k", kvProjOut, hiddenSize, 1},
		{"attn.v", kvProjOut, hiddenSize, 1},
		{"attn.o", hiddenSize, qProjOut, 1},
		{"moe.router", routerOut, hiddenSize, 1},
		{"moe.gate", moeInter, hiddenSize, numExperts},
		{"moe.up", moeInter, hiddenSize, numExperts},
		{"moe.down", hiddenSize, moeInter, numExperts},
	}
}

// tokenPlan returns every GEMV for one decode token over nLayers + lm_head.
func tokenPlan(nLayers int) []gemvOp {
	var ops []gemvOp
	for l := 0; l < nLayers; l++ {
		ops = append(ops, oneLayerPlan()...)
	}
	ops = append(ops, gemvOp{"lm_head", vocabSize, hiddenSize, 1})
	return ops
}

func planBytes(ops []gemvOp) int {
	total := 0
	for _, op := range ops {
		rowBytes := (op.cols / oxk.QK_K) * oxk.BLOCK_Q4_K_SIZE
		total += op.rows * rowBytes * op.count
	}
	return total
}

func planFlops(ops []gemvOp) float64 {
	var total float64
	for _, op := range ops {
		total += float64(op.rows) * float64(op.cols) * 2.0 * float64(op.count)
	}
	return total
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func fillPseudo(b []byte) {
	var state uint64 = 0x5eed
	for i := range b {
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17
		b[i] = byte(state)
	}
	// Tame the per-block f16 headers (d, dmin) so accumulators stay finite.
	for off := 0; off+4 <= len(b); off += oxk.BLOCK_Q4_K_SIZE {
		for h := 0; h < 2; h++ {
			raw := uint16(b[off+h*2]) | uint16(b[off+h*2+1])<<8
			tamed := (raw & 0x83ff) | (0x3000 + ((raw>>10)&0x7)*0x400)
			b[off+h*2] = byte(tamed)
			b[off+h*2+1] = byte(tamed >> 8)
		}
	}
}

func main() {
	fmt.Println("=== Go OXK full decode-token Benchmark (Qwen3-30B-A3B) ===")
	fmt.Println(oxk.OxkCpuSummary())

	// OXK_BENCH_LAYERS lets you time a subset of layers (cheap on slow ports);
	// tok/s is always projected to the full 48-layer token from measured GB/s.
	nLayers := envInt("OXK_BENCH_LAYERS", numLayers)
	if nLayers < 1 {
		nLayers = 1
	}
	if nLayers > numLayers {
		nLayers = numLayers
	}
	tokens := envInt("OXK_BENCH_TOKENS", 3)
	if tokens < 1 {
		tokens = 1
	}

	timedOps := tokenPlan(nLayers)
	fullOps := tokenPlan(numLayers)
	timedBytes := planBytes(timedOps)
	fullBytes := planBytes(fullOps)
	fullFlops := planFlops(fullOps)

	// One contiguous weight buffer for the timed pass; each op streams a
	// distinct region so the kernel hits DRAM like a real decode (no cache reuse).
	weights := make([]byte, timedBytes)
	fillPseudo(weights)

	// Pre-quantized Q8_K inputs, one per distinct cols width; reused per op.
	q8kByCols := map[int][]byte{}
	maxRows := 0
	for _, op := range timedOps {
		if _, ok := q8kByCols[op.cols]; !ok {
			blocks := op.cols / oxk.QK_K
			input := make([]float32, op.cols)
			for i := range input {
				input[i] = float32(i%255)/64.0 - 2.0
			}
			q8k := make([]byte, blocks*oxk.BLOCK_Q8_K_BYTES)
			if err := oxk.QuantizeQ8KInto(input, blocks, q8k); err != nil {
				panic(err)
			}
			q8kByCols[op.cols] = q8k
		}
		if op.rows > maxRows {
			maxRows = op.rows
		}
	}
	out := make([]float32, maxRows)

	runToken := func() float64 {
		var sink float64
		cursor := 0
		for _, op := range timedOps {
			blocks := op.cols / oxk.QK_K
			rowBytes := blocks * oxk.BLOCK_Q4_K_SIZE
			q8k := q8kByCols[op.cols]
			for c := 0; c < op.count; c++ {
				region := weights[cursor : cursor+op.rows*rowBytes]
				if err := oxk.GemvQ4kRange(region, blocks, q8k, out[:op.rows]); err != nil {
					panic(err)
				}
				cursor += op.rows * rowBytes
				sink += float64(out[0])
			}
		}
		return sink
	}

	// Warmup.
	_ = runToken()

	start := time.Now()
	var sink float64
	for i := 0; i < tokens; i++ {
		sink += runToken()
	}
	elapsed := time.Since(start).Seconds()

	gbps := float64(timedBytes) * float64(tokens) / 1e9 / elapsed
	gflops := planFlops(timedOps) * float64(tokens) / 1e9 / elapsed
	fullTokenSec := float64(fullBytes) / 1e9 / gbps
	projTokS := 1.0 / fullTokenSec

	fmt.Printf("\nTimed: %d layer(s) + lm_head, %d token-pass(es)  (sink=%.3e)\n", nLayers, tokens, sink)
	fmt.Printf("Streamed per timed pass: %.2f MB\n", float64(timedBytes)/1e6)
	fmt.Printf("Throughput:              %.2f GFLOP/s\n", gflops)
	fmt.Printf("Memory bandwidth:        %.2f GB/s\n", gbps)
	fmt.Printf("Full token (48L) weight bytes: %.2f GB, %.1f GFLOP\n", float64(fullBytes)/1e9, fullFlops/1e9)
	fmt.Printf("Projected full-token decode:   %.3f s/token  =>  %.3f tok/s\n", fullTokenSec, projTokS)
}
