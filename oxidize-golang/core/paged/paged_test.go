package paged

import "testing"

func TestBlockPoolAllocateRelease(t *testing.T) {
	p := NewBlockPool(2, 4)
	id, err := p.Allocate()
	if err != nil {
		t.Fatal(err)
	}
	if id < 0 || id >= 2 {
		t.Fatalf("unexpected id %d", id)
	}
	p.Release(id)
	if p.FreeCount() != 2 {
		t.Fatalf("expected 2 free, got %d", p.FreeCount())
	}
}

func TestSchedulerAddStep(t *testing.T) {
	s := NewScheduler(SchedulerConfig{BlockSize: 4, TotalBlocks: 8, MaxRequests: 2, MaxTokensPerReq: 16, Preemption: "recompute"})
	if _, err := s.AddRequest([]int{1, 2, 3, 4, 5}, 16); err != nil {
		t.Fatal(err)
	}
	scheduled, err := s.Step()
	if err != nil {
		t.Fatal(err)
	}
	if len(scheduled) != 1 {
		t.Fatalf("expected 1 scheduled, got %d", len(scheduled))
	}
	q, a, f := s.Stats()
	if q != 0 || a != 1 {
		t.Fatalf("expected q=0,a=1 got %d,%d (free=%d)", q, a, f)
	}
}

func TestSchedulerPreempt(t *testing.T) {
	s := NewScheduler(SchedulerConfig{BlockSize: 4, TotalBlocks: 4, MaxRequests: 1, MaxTokensPerReq: 16, Preemption: "recompute"})
	if _, err := s.AddRequest([]int{1, 2, 3, 4}, 16); err != nil {
		t.Fatal(err)
	}
	scheduled, _ := s.Step()
	if len(scheduled) != 1 {
		t.Fatalf("expected scheduled, got %d", len(scheduled))
	}
	// Add a second request that will preempt the first when blocks are exhausted
	if _, err := s.AddRequest([]int{5, 6, 7, 8}, 16); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Step(); err != nil {
		t.Fatal(err)
	}
}

func TestComputeBlockHash(t *testing.T) {
	h1 := ComputeBlockHash([]int{1, 2, 3, 4})
	h2 := ComputeBlockHash([]int{1, 2, 3, 4})
	if h1 != h2 {
		t.Fatal("expected identical hashes for identical tokens")
	}
	h3 := ComputeBlockHash([]int{1, 2, 3, 5})
	if h1 == h3 {
		t.Fatal("expected different hashes for different tokens")
	}
}
