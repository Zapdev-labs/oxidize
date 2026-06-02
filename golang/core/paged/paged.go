// Package paged implements a vLLM-style paged attention scheduler.
package paged

import (
	"errors"
	"sync"
)

// SchedulerConfig mirrors PagedSchedulerConfig.
type SchedulerConfig struct {
	BlockSize       int
	TotalBlocks     int
	MaxRequests     int
	MaxTokensPerReq int
	Preemption      string // "recompute" or "swap"
}

// DefaultSchedulerConfig returns sensible defaults for paged attention.
func DefaultSchedulerConfig() SchedulerConfig {
	return SchedulerConfig{BlockSize: 16, TotalBlocks: 1024, MaxRequests: 32, MaxTokensPerReq: 2048, Preemption: "recompute"}
}

// BlockHash mirrors BlockHash.
type BlockHash [16]byte

// String returns a hex string of the hash.
func (h BlockHash) String() string {
	out := make([]byte, 32)
	const hex = "0123456789abcdef"
	for i, b := range h {
		out[2*i] = hex[b>>4]
		out[2*i+1] = hex[b&0xf]
	}
	return string(out)
}

// PhysicalBlock mirrors PhysicalBlock.
type PhysicalBlock struct {
	ID    int
	Ref   int
	Dirty bool
}

// BlockPool mirrors BlockPool.
type BlockPool struct {
	mu      sync.Mutex
	blocks  []PhysicalBlock
	free    []int
	BlockSize int
}

// NewBlockPool constructs a pool with `n` blocks of given size.
func NewBlockPool(n, blockSize int) *BlockPool {
	if n <= 0 {
		n = 1
	}
	if blockSize <= 0 {
		blockSize = 16
	}
	bp := &BlockPool{blocks: make([]PhysicalBlock, n), free: make([]int, 0, n), BlockSize: blockSize}
	for i := 0; i < n; i++ {
		bp.free = append(bp.free, i)
		bp.blocks[i] = PhysicalBlock{ID: i}
	}
	return bp
}

// Allocate returns a free block id.
func (p *BlockPool) Allocate() (int, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.free) == 0 {
		return 0, errors.New("paged: no free blocks")
	}
	id := p.free[len(p.free)-1]
	p.free = p.free[:len(p.free)-1]
	p.blocks[id].Ref = 1
	return id, nil
}

// Release decrements the reference count of a block.
func (p *BlockPool) Release(id int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if id < 0 || id >= len(p.blocks) {
		return
	}
	p.blocks[id].Ref--
	if p.blocks[id].Ref <= 0 {
		p.blocks[id].Ref = 0
		p.blocks[id].Dirty = false
		p.free = append(p.free, id)
	}
}

// FreeCount returns the number of free blocks.
func (p *BlockPool) FreeCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.free)
}

// TotalCount returns the number of blocks managed.
func (p *BlockPool) TotalCount() int { return len(p.blocks) }

// BlockTable mirrors BlockTable.
type BlockTable struct {
	RequestID int
	Blocks    []int
}

// Request mirrors PagedRequest.
type Request struct {
	ID           int
	Tokens       []int
	BlockTable   []int
	MaxTokens    int
	Priority     int
	Preempted    bool
}

// SchedulerError mirrors PagedSchedulerError.
type SchedulerError struct{ Message string }

func (e *SchedulerError) Error() string { return "paged: " + e.Message }

// Scheduler mirrors PagedScheduler.
type Scheduler struct {
	mu        sync.Mutex
	cfg       SchedulerConfig
	pool      *BlockPool
	queue     []*Request
	active    map[int]*Request
	sequence  int
	hashTable map[BlockHash]int
}

// NewScheduler constructs a new scheduler.
func NewScheduler(cfg SchedulerConfig) *Scheduler {
	return &Scheduler{
		cfg:       cfg,
		pool:      NewBlockPool(cfg.TotalBlocks, cfg.BlockSize),
		active:    map[int]*Request{},
		hashTable: map[BlockHash]int{},
	}
}

// AddRequest enqueues a new request.
func (s *Scheduler) AddRequest(tokens []int, maxTokens int) (*Request, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.active)+len(s.queue) >= s.cfg.MaxRequests {
		return nil, &SchedulerError{Message: "max requests reached"}
	}
	s.sequence++
	r := &Request{ID: s.sequence, Tokens: append([]int(nil), tokens...), MaxTokens: maxTokens, Priority: 0}
	blocks := (len(tokens) + s.cfg.BlockSize - 1) / s.cfg.BlockSize
	if blocks == 0 {
		blocks = 1
	}
	for i := 0; i < blocks; i++ {
		id, err := s.pool.Allocate()
		if err != nil {
			return nil, err
		}
		r.BlockTable = append(r.BlockTable, id)
	}
	s.queue = append(s.queue, r)
	return r, nil
}

// Step runs one scheduling iteration.
func (s *Scheduler) Step() ([]*Request, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.queue) == 0 {
		return nil, nil
	}
	preempted := 0
	if s.cfg.Preemption == "recompute" && s.pool.FreeCount() == 0 {
		// Preempt the lowest priority request
		var victim *Request
		for _, r := range s.active {
			if victim == nil || r.Priority < victim.Priority {
				victim = r
			}
		}
		if victim != nil {
			for _, b := range victim.BlockTable {
				s.pool.Release(b)
			}
			delete(s.active, victim.ID)
			victim.Preempted = true
			preempted++
		}
	}
	if preempted > 0 {
		return nil, nil
	}
	scheduled := []*Request{}
	for _, r := range s.queue {
		if len(s.active) >= s.cfg.MaxRequests {
			break
		}
		s.active[r.ID] = r
		scheduled = append(scheduled, r)
	}
	s.queue = s.queue[len(scheduled):]
	return scheduled, nil
}

// Finish removes a request and releases its blocks.
func (s *Scheduler) Finish(id int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r, ok := s.active[id]
	if !ok {
		return
	}
	for _, b := range r.BlockTable {
		s.pool.Release(b)
	}
	delete(s.active, id)
}

// Stats returns a snapshot.
func (s *Scheduler) Stats() (queue, active, free int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.queue), len(s.active), s.pool.FreeCount()
}

// RegisterBlockHash registers a hash for prefix sharing.
func (s *Scheduler) RegisterBlockHash(h BlockHash, blockID int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.hashTable[h] = blockID
}

// LookupBlockHash returns a block id for the given hash.
func (s *Scheduler) LookupBlockHash(h BlockHash) (int, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	id, ok := s.hashTable[h]
	return id, ok
}

// ComputeBlockHash computes a simple block hash from tokens.
func ComputeBlockHash(tokens []int) BlockHash {
	var h BlockHash
	for i, t := range tokens {
		b := byte(t)
		h[i%len(h)] ^= b
	}
	return h
}
