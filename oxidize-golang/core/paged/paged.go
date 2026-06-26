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
	// BlockHash is the prefix-cache hash for this block, or hasHash=false when
	// the block is not currently part of the prefix cache.
	Hash    BlockHash
	HasHash bool
	// LastAccessed is the monotonic access counter used for LRU eviction.
	LastAccessed uint64
}

// BlockPool mirrors BlockPool.
type BlockPool struct {
	mu        sync.Mutex
	blocks    []PhysicalBlock
	free      []int
	BlockSize int
	// prefixCache maps a block hash to the physical block id holding that prefix.
	prefixCache map[BlockHash]int
	// accessCounter is a monotonically increasing LRU clock.
	accessCounter uint64
}

// NewBlockPool constructs a pool with `n` blocks of given size.
func NewBlockPool(n, blockSize int) *BlockPool {
	if n <= 0 {
		n = 1
	}
	if blockSize <= 0 {
		blockSize = 16
	}
	bp := &BlockPool{
		blocks:      make([]PhysicalBlock, n),
		free:        make([]int, 0, n),
		BlockSize:   blockSize,
		prefixCache: make(map[BlockHash]int),
	}
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

// AllocatedCount returns the number of currently allocated (non-free) blocks.
func (p *BlockPool) AllocatedCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.blocks) - len(p.free)
}

// allocateLocked pops a free block and marks it allocated (ref=1). Caller holds mu.
func (p *BlockPool) allocateLocked() (int, error) {
	if len(p.free) == 0 {
		return 0, errors.New("paged: no free blocks")
	}
	id := p.free[len(p.free)-1]
	p.free = p.free[:len(p.free)-1]
	p.blocks[id].Ref = 1
	p.blocks[id].HasHash = false
	p.blocks[id].Dirty = false
	return id, nil
}

// AllocateBlocks allocates `n` physical blocks with all-or-nothing semantics:
// if fewer than `n` blocks are free, no blocks are allocated and an error is
// returned. Mirrors BlockPool::allocate_blocks.
func (p *BlockPool) AllocateBlocks(n int) ([]int, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if n <= 0 {
		return nil, nil
	}
	if len(p.free) < n {
		return nil, errors.New("paged: no free blocks")
	}
	ids := make([]int, 0, n)
	for i := 0; i < n; i++ {
		id, err := p.allocateLocked()
		if err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	return ids, nil
}

// IncRef increments the reference count of an allocated block. Returns an error
// if the block was not allocated (ref==0). Mirrors BlockPool::inc_ref.
func (p *BlockPool) IncRef(id int) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if id < 0 || id >= len(p.blocks) {
		return errors.New("paged: invalid block id")
	}
	if p.blocks[id].Ref == 0 {
		return errors.New("paged: block not allocated")
	}
	p.blocks[id].Ref++
	return nil
}

// DecRef decrements the reference count of a block, returning it to the free
// list when the count reaches zero. Returns an error if the block was not
// allocated. Mirrors BlockPool::dec_ref.
func (p *BlockPool) DecRef(id int) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if id < 0 || id >= len(p.blocks) {
		return errors.New("paged: invalid block id")
	}
	if p.blocks[id].Ref == 0 {
		return errors.New("paged: block not allocated")
	}
	p.blocks[id].Ref--
	if p.blocks[id].Ref == 0 {
		p.blocks[id].Dirty = false
		if !p.isFreeLocked(id) {
			p.free = append(p.free, id)
		}
	}
	return nil
}

func (p *BlockPool) isFreeLocked(id int) bool {
	for _, f := range p.free {
		if f == id {
			return true
		}
	}
	return false
}

// RefCount returns the reference count of a block (0 if invalid).
func (p *BlockPool) RefCount(id int) int {
	p.mu.Lock()
	defer p.mu.Unlock()
	if id < 0 || id >= len(p.blocks) {
		return 0
	}
	return p.blocks[id].Ref
}

// LookupPrefixCache returns the physical block id for a cached prefix hash,
// updating its LRU access time. Stale entries (block freed) are pruned and
// reported as not found. Mirrors BlockPool::lookup_prefix_cache.
func (p *BlockPool) LookupPrefixCache(h BlockHash) (int, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	id, ok := p.prefixCache[h]
	if !ok {
		return 0, false
	}
	if id < 0 || id >= len(p.blocks) || p.blocks[id].Ref == 0 {
		delete(p.prefixCache, h)
		return 0, false
	}
	p.accessCounter++
	p.blocks[id].LastAccessed = p.accessCounter
	return id, true
}

// InsertPrefixCache records a hash → block mapping for prefix reuse. The block
// must be allocated. If the hash already exists the existing mapping wins
// (first-seen). Mirrors BlockPool::insert_prefix_cache.
func (p *BlockPool) InsertPrefixCache(h BlockHash, id int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if id < 0 || id >= len(p.blocks) || p.blocks[id].Ref == 0 {
		return
	}
	p.blocks[id].Hash = h
	p.blocks[id].HasHash = true
	p.accessCounter++
	p.blocks[id].LastAccessed = p.accessCounter
	if _, exists := p.prefixCache[h]; !exists {
		p.prefixCache[h] = id
	}
}

// EvictLRUPrefixCacheEntry removes the least-recently-used cache entry whose
// block ref count is zero. Returns true if an entry was evicted. Mirrors
// BlockPool::evict_lru_prefix_cache_entry.
func (p *BlockPool) EvictLRUPrefixCacheEntry() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	var (
		victimHash  BlockHash
		victimID    = -1
		victimClock uint64
	)
	for h, id := range p.prefixCache {
		if id < 0 || id >= len(p.blocks) {
			continue
		}
		if p.blocks[id].Ref != 0 {
			continue
		}
		if victimID == -1 || p.blocks[id].LastAccessed < victimClock {
			victimHash = h
			victimID = id
			victimClock = p.blocks[id].LastAccessed
		}
	}
	if victimID == -1 {
		return false
	}
	delete(p.prefixCache, victimHash)
	p.blocks[victimID].HasHash = false
	return true
}

// ClearPrefixCache removes all prefix-cache entries (e.g. on model switch).
// Mirrors BlockPool::clear_prefix_cache.
func (p *BlockPool) ClearPrefixCache() {
	p.mu.Lock()
	defer p.mu.Unlock()
	for h, id := range p.prefixCache {
		if id >= 0 && id < len(p.blocks) && p.blocks[id].HasHash && p.blocks[id].Hash == h {
			p.blocks[id].HasHash = false
		}
	}
	p.prefixCache = make(map[BlockHash]int)
}

// PrefixCacheLen returns the number of prefix-cache entries.
func (p *BlockPool) PrefixCacheLen() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.prefixCache)
}

// CopyOnWrite implements copy-on-write for a shared block. If the block's ref
// count is > 1, a new block is allocated, the original's ref is decremented,
// and the new block id is returned (found=true). If the block is not shared,
// (0, false, nil) is returned. Mirrors BlockPool::copy_on_write.
func (p *BlockPool) CopyOnWrite(id int) (int, bool, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if id < 0 || id >= len(p.blocks) {
		return 0, false, errors.New("paged: invalid block id")
	}
	if p.blocks[id].Ref <= 1 {
		return 0, false, nil
	}
	newID, err := p.allocateLocked()
	if err != nil {
		return 0, false, err
	}
	// Decrement original (still > 0 since it was > 1).
	p.blocks[id].Ref--
	return newID, true, nil
}

// BlockTable mirrors BlockTable.
type BlockTable struct {
	RequestID int
	Blocks    []int
}

// Request mirrors PagedRequest.
type Request struct {
	ID         int
	Tokens     []int
	BlockTable []int
	MaxTokens  int
	Priority   int
	Preempted  bool
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

// AddRequest enqueues a new request. The MaxRequests limit applies to
// the active set, not the queue, so callers can buffer more requests
// than the concurrency limit and let the scheduler pick/preempt from
// the queue as needed.
func (s *Scheduler) AddRequest(tokens []int, maxTokens int) (*Request, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.active) >= s.cfg.MaxRequests && len(s.queue) > 0 {
		// The queue is non-empty and the active set is already full;
		// refuse rather than growing unboundedly.
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

// ComputeBlockHash computes a deterministic FNV-1a block hash from tokens,
// mirroring the Rust `compute_block_hash` (64-bit FNV-1a). The 64-bit digest is
// stored in the leading 8 bytes of the [16]byte hash so existing String()/map
// behaviour is preserved while collisions are far less likely than the old
// XOR-fold scheme.
func ComputeBlockHash(tokens []int) BlockHash {
	const (
		fnvOffset uint64 = 0xcbf29ce484222325
		fnvPrime  uint64 = 0x100000001b3
	)
	h := fnvOffset
	for _, t := range tokens {
		h *= fnvPrime
		h ^= uint64(uint32(t))
	}
	var out BlockHash
	for i := 0; i < 8; i++ {
		out[i] = byte(h >> (8 * uint(7-i)))
	}
	return out
}
