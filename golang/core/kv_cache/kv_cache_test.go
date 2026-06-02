package kv_cache

import "testing"

func TestNewCache(t *testing.T) {
	c := NewCache(DefaultConfig())
	if c == nil {
		t.Fatal("nil cache")
	}
	if c.OccupiedTokens() != 0 {
		t.Fatalf("occupied = %d", c.OccupiedTokens())
	}
}

func TestAppendGet(t *testing.T) {
	cfg := DefaultConfig()
	cfg.LayerCount = 2
	cfg.HeadCount = 1
	cfg.HeadDim = 4
	c := NewCache(cfg)
	if err := c.Append(0, []float32{1, 2, 3, 4}, []float32{5, 6, 7, 8}); err != nil {
		t.Fatalf("err: %v", err)
	}
	if c.OccupiedTokens() != 1 {
		t.Fatalf("occupied = %d", c.OccupiedTokens())
	}
	keys, values, length, err := c.Get(0)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if length != 1 {
		t.Fatalf("length = %d", length)
	}
	if keys[0] != 1 || values[3] != 8 {
		t.Fatalf("data mismatch: k=%v v=%v", keys[:4], values[:4])
	}
}

func TestAppendLayerOutOfBounds(t *testing.T) {
	c := NewCache(DefaultConfig())
	if err := c.Append(99, []float32{1}, []float32{1}); err == nil {
		t.Fatal("expected layer error")
	}
}

func TestContinuousBatch(t *testing.T) {
	cfg := DefaultConfig()
	cfg.LayerCount = 1
	cfg.HeadCount = 1
	cfg.HeadDim = 2
	c := NewContinuousBatchCache(cfg)
	if err := c.CreateSequence(1); err != nil {
		t.Fatalf("create: %v", err)
	}
	if err := c.Append(1, 0, []float32{1, 2}, []float32{3, 4}); err != nil {
		t.Fatalf("append: %v", err)
	}
	if c.SequenceLength(1, 0) != 1 {
		t.Fatalf("length = %d", c.SequenceLength(1, 0))
	}
	if err := c.RemoveSequence(1); err != nil {
		t.Fatalf("remove: %v", err)
	}
	if c.SequenceLength(1, 0) != 0 {
		t.Fatal("length should be 0 after remove")
	}
}
