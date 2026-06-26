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

func TestPrefixCacheLRU(t *testing.T) {
	p := NewBlockPool(2, 4)
	h := ComputeBlockHash([]int{1, 2, 3, 4})
	id, err := p.AllocateBlocks(1)
	if err != nil {
		t.Fatal(err)
	}
	p.InsertPrefixCache(h, id[0])
	got, ok := p.LookupPrefixCache(h)
	if !ok || got != id[0] {
		t.Fatalf("expected cache hit on %d, got %d ok=%v", id[0], got, ok)
	}
	if p.PrefixCacheLen() != 1 {
		t.Fatalf("expected 1 cache entry, got %d", p.PrefixCacheLen())
	}
	// Free the block; entry becomes evictable.
	if err := p.DecRef(id[0]); err != nil {
		t.Fatal(err)
	}
	if !p.EvictLRUPrefixCacheEntry() {
		t.Fatal("expected eviction of ref==0 entry")
	}
	if p.PrefixCacheLen() != 0 {
		t.Fatalf("expected empty cache, got %d", p.PrefixCacheLen())
	}
}

func TestCopyOnWrite(t *testing.T) {
	p := NewBlockPool(4, 4)
	id, err := p.AllocateBlocks(1)
	if err != nil {
		t.Fatal(err)
	}
	bid := id[0]
	// Not shared yet -> no copy.
	if _, copied, err := p.CopyOnWrite(bid); err != nil || copied {
		t.Fatalf("expected no copy for unshared block, copied=%v err=%v", copied, err)
	}
	// Share it.
	if err := p.IncRef(bid); err != nil {
		t.Fatal(err)
	}
	newID, copied, err := p.CopyOnWrite(bid)
	if err != nil || !copied {
		t.Fatalf("expected copy for shared block, copied=%v err=%v", copied, err)
	}
	if newID == bid {
		t.Fatal("expected a distinct new block id")
	}
	if p.RefCount(bid) != 1 {
		t.Fatalf("expected original ref==1 after COW, got %d", p.RefCount(bid))
	}
}

func TestDecRefReturnsToFreeList(t *testing.T) {
	p := NewBlockPool(2, 4)
	id, err := p.AllocateBlocks(1)
	if err != nil {
		t.Fatal(err)
	}
	if p.FreeCount() != 1 {
		t.Fatalf("expected 1 free after alloc, got %d", p.FreeCount())
	}
	if err := p.DecRef(id[0]); err != nil {
		t.Fatal(err)
	}
	if p.FreeCount() != 2 {
		t.Fatalf("expected 2 free after dec_ref, got %d", p.FreeCount())
	}
	// dec_ref on freed block must error.
	if err := p.DecRef(id[0]); err == nil {
		t.Fatal("expected error decrementing freed block")
	}
}

func TestAllocateBlocksAllOrNothing(t *testing.T) {
	p := NewBlockPool(2, 4)
	if _, err := p.AllocateBlocks(3); err == nil {
		t.Fatal("expected error for over-allocation")
	}
	if p.FreeCount() != 2 {
		t.Fatalf("expected all blocks still free, got %d", p.FreeCount())
	}
}

func TestSchedulerV2ThreePhaseAndBudget(t *testing.T) {
	s := NewSchedulerV2(SchedulerV2Config{MaxBatchedTokens: 8, PrefillChunkSize: 4, MaxRunningSeqs: 4}, 64, 4)
	a := s.AddRequest([]int{1, 2, 3, 4, 5}, 4, 2, true)
	b := s.AddRequest([]int{6, 7, 8}, 4, 2, true)

	// Step 1: both should prefill (chunked). Budget 8 covers chunk(4)+chunk(3).
	res, err := s.Step()
	if err != nil {
		t.Fatal(err)
	}
	if res.PrefillTokens == 0 {
		t.Fatalf("expected prefill tokens, got %d", res.PrefillTokens)
	}
	if len(res.ScheduledSeqIDs) != 2 {
		t.Fatalf("expected 2 scheduled, got %d", len(res.ScheduledSeqIDs))
	}
	batch := s.BuildInputBatch(res)
	if batch.BatchSize != 2 {
		t.Fatalf("expected batch size 2, got %d", batch.BatchSize)
	}
	if batch.TotalTokens > 8 {
		t.Fatalf("budget exceeded: %d", batch.TotalTokens)
	}

	// Finish b's prefill, then it should decode. Simulate sampled tokens.
	// First drive a to completion of prefill across a couple steps.
	for i := 0; i < 4; i++ {
		res, err = s.Step()
		if err != nil {
			t.Fatal(err)
		}
		sampled := map[int]int{}
		for _, id := range res.ScheduledSeqIDs {
			seq, _ := s.GetSequence(id)
			if seq.RemainingPrefillTokens() == 0 {
				sampled[id] = 9
			}
		}
		if err := s.PostprocessStep(sampled); err != nil {
			t.Fatal(err)
		}
	}
	_ = a
	_ = b
}

func TestSchedulerV2PrefixCacheReuse(t *testing.T) {
	s := NewSchedulerV2(SchedulerV2Config{MaxBatchedTokens: 64, PrefillChunkSize: 64, MaxRunningSeqs: 4}, 64, 4)
	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}
	// First request prefills and populates the prefix cache.
	id1 := s.AddRequest(prompt, 2, 2, true)
	if _, err := s.Step(); err != nil {
		t.Fatal(err)
	}
	seq1, _ := s.GetSequence(id1)
	if seq1.RemainingPrefillTokens() != 0 {
		t.Fatalf("expected fully prefilled, remaining=%d", seq1.RemainingPrefillTokens())
	}
	if s.Pool().PrefixCacheLen() == 0 {
		t.Fatal("expected prefix cache to be populated")
	}
	// Second request with identical prompt should find prefix hits.
	id2 := s.AddRequest(prompt, 2, 2, true)
	hits, err := s.FindPrefixCacheHits(id2)
	if err != nil {
		t.Fatal(err)
	}
	if hits == 0 {
		t.Fatal("expected prefix cache hits for identical prompt")
	}
}

func TestSchedulerV2Preempt(t *testing.T) {
	s := NewSchedulerV2(SchedulerV2Config{MaxBatchedTokens: 64, PrefillChunkSize: 64, MaxRunningSeqs: 4}, 64, 4)
	id := s.AddRequest([]int{1, 2, 3, 4, 5}, 4, 2, true)
	if _, err := s.Step(); err != nil {
		t.Fatal(err)
	}
	freeBefore := s.Pool().FreeCount()
	if err := s.PreemptSequence(id); err != nil {
		t.Fatal(err)
	}
	if s.Pool().FreeCount() <= freeBefore {
		t.Fatalf("expected blocks freed on preempt: before=%d after=%d", freeBefore, s.Pool().FreeCount())
	}
	if s.WaitingCount() != 1 {
		t.Fatalf("expected sequence back in waiting, got %d", s.WaitingCount())
	}
	seq, _ := s.GetSequence(id)
	if seq.NumPrefilled() != 0 {
		t.Fatalf("expected prefill reset, got %d", seq.NumPrefilled())
	}
}

func TestSchedulerV2DrainAndReinit(t *testing.T) {
	s := NewSchedulerV2(DefaultSchedulerV2Config(), 64, 4)
	s.AddRequest([]int{1, 2, 3, 4}, 4, 2, true)
	if _, err := s.Step(); err != nil {
		t.Fatal(err)
	}
	if err := s.DrainAndReinitialize(); err != nil {
		t.Fatal(err)
	}
	if s.WaitingCount() != 0 || s.RunningCount() != 0 {
		t.Fatal("expected empty queues after drain")
	}
	if s.Pool().FreeCount() != s.Pool().TotalCount() {
		t.Fatalf("expected all blocks free after drain: %d/%d", s.Pool().FreeCount(), s.Pool().TotalCount())
	}
}
