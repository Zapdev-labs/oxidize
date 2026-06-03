package kv_cache

import (
	"math"
	"path/filepath"
	"testing"
)

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

// TestFloatBitsRoundTrip guards against a regression where float32bits /
// float32frombits were implemented via mutual recursion through asBytesF32,
// producing a stack overflow and/or always returning zero on load.
func TestFloatBitsRoundTrip(t *testing.T) {
	cases := []float32{0, 1, -1, 1.5, -1.5, 1.0e-7, 1.0e7, math.MaxFloat32, math.SmallestNonzeroFloat32}
	for _, v := range cases {
		got := float32frombits(float32bits(v))
		if got != v {
			t.Fatalf("roundtrip lost value: %v -> bits -> %v", v, got)
		}
	}
}

// TestSaveLoadRoundTrip ensures the on-disk cache encoding roundtrips
// float32 keys and values byte-for-byte. This exercises the float32
// (de)serialization helpers used by SaveToDisk / LoadFromDisk.
func TestSaveLoadRoundTrip(t *testing.T) {
	cfg := DefaultConfig()
	cfg.LayerCount = 2
	cfg.HeadCount = 1
	cfg.HeadDim = 4
	c := NewCache(cfg)
	if err := c.Append(0, []float32{1, 2, 3, 4}, []float32{5, 6, 7, 8}); err != nil {
		t.Fatalf("append 0: %v", err)
	}
	if err := c.Append(1, []float32{-1, -2, -3, -4}, []float32{-5, -6, -7, -8}); err != nil {
		t.Fatalf("append 1: %v", err)
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "cache.bin")
	if err := c.SaveToDisk(path); err != nil {
		t.Fatalf("save: %v", err)
	}
	loaded, err := LoadFromDisk(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	k0, v0, l0, err := loaded.Get(0)
	if err != nil {
		t.Fatalf("get 0: %v", err)
	}
	if l0 != 1 || k0[0] != 1 || k0[3] != 4 || v0[0] != 5 || v0[3] != 8 {
		t.Fatalf("layer 0 mismatch: k=%v v=%v len=%d", k0[:4], v0[:4], l0)
	}
	k1, v1, l1, err := loaded.Get(1)
	if err != nil {
		t.Fatalf("get 1: %v", err)
	}
	if l1 != 1 || k1[0] != -1 || k1[3] != -4 || v1[0] != -5 || v1[3] != -8 {
		t.Fatalf("layer 1 mismatch: k=%v v=%v len=%d", k1[:4], v1[:4], l1)
	}
}
