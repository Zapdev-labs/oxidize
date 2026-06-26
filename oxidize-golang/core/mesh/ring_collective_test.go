package mesh

import (
	"context"
	"reflect"
	"testing"
	"time"
)

func runRing(t *testing.T, backends []*RingBackend, fn func(ctx context.Context, b *RingBackend) error) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	errs := make(chan error, len(backends))
	for _, b := range backends {
		go func(b *RingBackend) { errs <- fn(ctx, b) }(b)
	}
	for range backends {
		if err := <-errs; err != nil {
			t.Fatalf("ring op failed: %v", err)
		}
	}
}

func TestRingAllSumTwoRanks(t *testing.T) {
	backends := CreateMockRing(2)
	data := [][]float32{
		{1, 2, 3, 4},
		{10, 20, 30, 40},
	}
	runRing(t, backends, func(ctx context.Context, b *RingBackend) error {
		return b.AllSum(ctx, data[b.Rank])
	})
	expected := []float32{11, 22, 33, 44}
	for i, d := range data {
		if !reflect.DeepEqual(d, expected) {
			t.Fatalf("rank %d got %v, expected %v", i, d, expected)
		}
	}
}

func TestRingAllSumFourRanks(t *testing.T) {
	const numRanks = 4
	backends := CreateMockRing(numRanks)
	data := make([][]float32, numRanks)
	for r := 0; r < numRanks; r++ {
		d := make([]float32, 8)
		for i := range d {
			d[i] = float32(r*10 + i)
		}
		data[r] = d
	}
	runRing(t, backends, func(ctx context.Context, b *RingBackend) error {
		return b.AllSum(ctx, data[b.Rank])
	})
	expected := make([]float32, 8)
	for i := 0; i < 8; i++ {
		var sum float32
		for r := 0; r < numRanks; r++ {
			sum += float32(r*10 + i)
		}
		expected[i] = sum
	}
	for r := 0; r < numRanks; r++ {
		if !reflect.DeepEqual(data[r], expected) {
			t.Fatalf("rank %d got %v, expected %v", r, data[r], expected)
		}
	}
}

func TestRingAllSumSingleRankNoop(t *testing.T) {
	b := CreateMockRing(1)[0]
	data := []float32{1, 2, 3}
	if err := b.AllSum(context.Background(), data); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(data, []float32{1, 2, 3}) {
		t.Fatalf("single-rank should be noop, got %v", data)
	}
}

func TestRingAllGatherFourRanks(t *testing.T) {
	const numRanks = 4
	const chunk = 4
	backends := CreateMockRing(numRanks)
	outs := make([][]float32, numRanks)
	runRing(t, backends, func(ctx context.Context, b *RingBackend) error {
		data := make([]float32, chunk)
		for i := range data {
			data[i] = float32(b.Rank*10 + i)
		}
		out := make([]float32, chunk*numRanks)
		outs[b.Rank] = out
		return b.AllGather(ctx, data, out)
	})
	expected := make([]float32, 0, chunk*numRanks)
	for r := 0; r < numRanks; r++ {
		for i := 0; i < chunk; i++ {
			expected = append(expected, float32(r*10+i))
		}
	}
	for r := 0; r < numRanks; r++ {
		if !reflect.DeepEqual(outs[r], expected) {
			t.Fatalf("rank %d got %v, expected %v", r, outs[r], expected)
		}
	}
}

func TestRingTcpAllSum(t *testing.T) {
	const numRanks = 4
	const n = 4096
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	backends, closeRing, err := CreateTcpRing(ctx, numRanks)
	if err != nil {
		t.Fatalf("create tcp ring: %v", err)
	}
	defer closeRing()

	data := make([][]float32, numRanks)
	for r := 0; r < numRanks; r++ {
		d := make([]float32, n)
		for i := range d {
			d[i] = float32(r) * float32(i)
		}
		data[r] = d
	}
	runRing(t, backends, func(ctx context.Context, b *RingBackend) error {
		return b.AllSum(ctx, data[b.Rank])
	})
	expected := make([]float32, n)
	for i := 0; i < n; i++ {
		var sum float32
		for r := 0; r < numRanks; r++ {
			sum += float32(r) * float32(i)
		}
		expected[i] = sum
	}
	for r := 0; r < numRanks; r++ {
		if !reflect.DeepEqual(data[r], expected) {
			t.Fatalf("tcp rank %d mismatch", r)
		}
	}
}

func TestPipelineSendRecv(t *testing.T) {
	backends := CreateMockRing(2)
	ctx := context.Background()
	activations := []float32{1.5, 2.5, 3.5}
	var got []float32
	errs := make(chan error, 2)
	go func() { errs <- PipelineSend(ctx, backends[0], activations) }()
	go func() {
		var err error
		got, err = PipelineRecv(ctx, backends[1], len(activations))
		errs <- err
	}()
	for i := 0; i < 2; i++ {
		if err := <-errs; err != nil {
			t.Fatal(err)
		}
	}
	if !reflect.DeepEqual(got, activations) {
		t.Fatalf("got %v, expected %v", got, activations)
	}
}

func TestF32ByteRoundtrip(t *testing.T) {
	in := []float32{0, 1, -1, 3.14159, 1e9}
	bytes := f32SliceToBytes(in)
	out, err := bytesToF32Slice(bytes)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(in, out) {
		t.Fatalf("roundtrip mismatch: %v vs %v", in, out)
	}
}
