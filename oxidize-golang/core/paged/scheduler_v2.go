package paged

// This file ports the vLLM-style three-phase scheduler from

import (
	"fmt"
	"sync"
)

// SeqID uniquely identifies a sequence (request) in the v2 scheduler.
type SeqID = int

// SequenceStatus is the lifecycle state of a sequence.
type SequenceStatus int

const (
	// StatusWaiting means the sequence has not yet started prefill.
	StatusWaiting SequenceStatus = iota
	// StatusRunning means the sequence is actively batched.
	StatusRunning
	// StatusFinished means the sequence completed and blocks were reclaimed.
	StatusFinished
)

// SchedulerV2Config mirrors the Rust SchedulerConfig (scheduler/config.rs).
type SchedulerV2Config struct {
	// MaxBatchedTokens is the token budget per scheduler step.
	MaxBatchedTokens int
	// PrefillChunkSize is the default tokens per prefill chunk.
	PrefillChunkSize int
	// MaxRunningSeqs is the maximum number of simultaneously running sequences.
	MaxRunningSeqs int
}

// DefaultSchedulerV2Config returns the Rust defaults (512 / 16 / 8).
func DefaultSchedulerV2Config() SchedulerV2Config {
	return SchedulerV2Config{MaxBatchedTokens: 512, PrefillChunkSize: 16, MaxRunningSeqs: 8}
}

// Sequence is a single generation request managed by the v2 scheduler.
// Mirrors scheduler/sequence.rs::Sequence.
type Sequence struct {
	id              SeqID
	status          SequenceStatus
	promptTokens    []int
	generatedTokens []int
	blockTable      *SeqBlockTable
	arrivalOrder    int
	maxNewTokens    int
	stopToken       int
	hasStopToken    bool
	numPrefilled    int
}

// NewSequence constructs a waiting sequence. arrivalOrder is assigned by the
// scheduler in AddSequence.
func NewSequence(id SeqID, promptTokens []int, blockSize, maxNewTokens int, stopToken int, hasStopToken bool) *Sequence {
	return &Sequence{
		id:           id,
		status:       StatusWaiting,
		promptTokens: append([]int(nil), promptTokens...),
		blockTable:   NewSeqBlockTable(blockSize),
		maxNewTokens: maxNewTokens,
		stopToken:    stopToken,
		hasStopToken: hasStopToken,
	}
}

// ID returns the sequence id.
func (s *Sequence) ID() SeqID { return s.id }

// Status returns the current lifecycle status.
func (s *Sequence) Status() SequenceStatus { return s.status }

// PromptTokens returns the prompt token ids.
func (s *Sequence) PromptTokens() []int { return s.promptTokens }

// GeneratedTokens returns the tokens generated so far.
func (s *Sequence) GeneratedTokens() []int { return s.generatedTokens }

// NumTokens returns prompt + generated token count.
func (s *Sequence) NumTokens() int { return len(s.promptTokens) + len(s.generatedTokens) }

// NumPrefilled returns the number of prompt tokens already in the KV cache.
func (s *Sequence) NumPrefilled() int { return s.numPrefilled }

// RemainingPrefillTokens returns prompt tokens not yet prefilled.
func (s *Sequence) RemainingPrefillTokens() int {
	r := len(s.promptTokens) - s.numPrefilled
	if r < 0 {
		return 0
	}
	return r
}

func (s *Sequence) recordPrefilled(count int) {
	s.numPrefilled += count
	if s.numPrefilled > len(s.promptTokens) {
		s.numPrefilled = len(s.promptTokens)
	}
}

func (s *Sequence) appendToken(token int) { s.generatedTokens = append(s.generatedTokens, token) }

// IsFinished reports whether the sequence has reached a stop condition.
func (s *Sequence) IsFinished() bool {
	if len(s.generatedTokens) >= s.maxNewTokens {
		return true
	}
	if s.hasStopToken && len(s.generatedTokens) > 0 &&
		s.generatedTokens[len(s.generatedTokens)-1] == s.stopToken {
		return true
	}
	return false
}

// SeqBlockTable is the per-sequence logical→physical block mapping.
// Mirrors block_pool.rs::BlockTable.
type SeqBlockTable struct {
	logicalToPhysical []int
	numTokens         int
	blockSize         int
}

// NewSeqBlockTable creates an empty block table with the given block size.
func NewSeqBlockTable(blockSize int) *SeqBlockTable {
	if blockSize <= 0 {
		blockSize = 16
	}
	return &SeqBlockTable{blockSize: blockSize}
}

// NumTokens returns the number of tokens tracked by the table.
func (t *SeqBlockTable) NumTokens() int { return t.numTokens }

// NumBlocks returns the number of logical blocks assigned.
func (t *SeqBlockTable) NumBlocks() int { return len(t.logicalToPhysical) }

// BlockSize returns the tokens-per-block size.
func (t *SeqBlockTable) BlockSize() int { return t.blockSize }

// PhysicalBlocks returns the physical block ids in logical order.
func (t *SeqBlockTable) PhysicalBlocks() []int { return t.logicalToPhysical }

// GetPhysicalBlock returns the physical block id for a logical index.
func (t *SeqBlockTable) GetPhysicalBlock(logical int) (int, bool) {
	if logical < 0 || logical >= len(t.logicalToPhysical) {
		return 0, false
	}
	return t.logicalToPhysical[logical], true
}

// AppendBlock appends a new physical block to the mapping.
func (t *SeqBlockTable) AppendBlock(id int) { t.logicalToPhysical = append(t.logicalToPhysical, id) }

// SetPhysicalBlock updates the physical mapping for a logical block (COW).
func (t *SeqBlockTable) SetPhysicalBlock(logical, id int) {
	if logical >= 0 && logical < len(t.logicalToPhysical) {
		t.logicalToPhysical[logical] = id
	}
}

// AppendToken advances the token count, returning true if a new block is
// required. Mirrors BlockTable::append_token.
func (t *SeqBlockTable) AppendToken() bool {
	idx := t.numTokens % t.blockSize
	needs := idx == 0 && t.numTokens > 0
	t.numTokens++
	return needs
}

// BlocksNeededForTokens returns how many additional blocks are needed to store
// `n` more tokens. Mirrors BlockTable::blocks_needed_for_tokens.
func (t *SeqBlockTable) BlocksNeededForTokens(n int) int {
	future := t.numTokens + n
	futureBlocks := (future + t.blockSize - 1) / t.blockSize
	need := futureBlocks - len(t.logicalToPhysical)
	if need < 0 {
		return 0
	}
	return need
}

// InputBatch is the flattened batch for a single forward pass.
// Mirrors scheduler/config.rs::InputBatch.
type InputBatch struct {
	BatchSize   int
	SeqIDs      []SeqID
	TokenIDs    [][]int
	Positions   [][]int
	BlockTables [][]int
	NumTokens   []int
	TotalTokens int
	IsPrefill   []bool
	ContextLens []int
}

// SchedulerStepResult is returned by SchedulerV2.Step.
// Mirrors scheduler/config.rs::SchedulerStepResult.
type SchedulerStepResult struct {
	ScheduledSeqIDs  []SeqID
	PrefillTokens    int
	DecodeTokens     int
	SeqPrefillTokens map[SeqID]int
	SeqDecodeTokens  map[SeqID]int
}

// SchedulerV2 is the budgeted three-phase paged-attention scheduler.
// Mirrors scheduler/core.rs::Scheduler.
type SchedulerV2 struct {
	mu             sync.Mutex
	config         SchedulerV2Config
	pool           *BlockPool
	sequences      map[SeqID]*Sequence
	waiting        []SeqID
	running        []SeqID
	nextArrival    int
	nextSeqID      int
	usePrefixCache bool
	// lastCowTriggered records whether the most recent cowDecodeBlockLocked
	// actually copied a block. Only meaningful while mu is held.
	lastCowTriggered bool
}

// NewSchedulerV2 constructs a scheduler with its own block pool.
func NewSchedulerV2(config SchedulerV2Config, totalBlocks, blockSize int) *SchedulerV2 {
	if config.MaxBatchedTokens <= 0 {
		config.MaxBatchedTokens = 512
	}
	if config.PrefillChunkSize <= 0 {
		config.PrefillChunkSize = 16
	}
	if config.MaxRunningSeqs <= 0 {
		config.MaxRunningSeqs = 8
	}
	return &SchedulerV2{
		config:         config,
		pool:           NewBlockPool(totalBlocks, blockSize),
		sequences:      make(map[SeqID]*Sequence),
		usePrefixCache: true,
	}
}

// NewSchedulerV2WithPool constructs a scheduler over an existing pool.
func NewSchedulerV2WithPool(config SchedulerV2Config, pool *BlockPool) *SchedulerV2 {
	s := NewSchedulerV2(config, pool.TotalCount(), pool.BlockSize)
	s.pool = pool
	return s
}

// Pool returns the underlying block pool.
func (s *SchedulerV2) Pool() *BlockPool { return s.pool }

// Config returns the scheduler configuration.
func (s *SchedulerV2) Config() SchedulerV2Config { return s.config }

// SetPrefixCacheEnabled toggles prefix-cache-aware prefill.
func (s *SchedulerV2) SetPrefixCacheEnabled(enabled bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.usePrefixCache = enabled
}

// AddSequence enqueues a fully-constructed sequence into the waiting queue.
func (s *SchedulerV2) AddSequence(seq *Sequence) {
	s.mu.Lock()
	defer s.mu.Unlock()
	seq.arrivalOrder = s.nextArrival
	s.nextArrival++
	if seq.id <= 0 {
		s.nextSeqID++
		seq.id = s.nextSeqID
	} else if seq.id > s.nextSeqID {
		s.nextSeqID = seq.id
	}
	s.sequences[seq.id] = seq
	s.waiting = append(s.waiting, seq.id)
}

// AddRequest is a convenience that builds a sequence and enqueues it, returning
// the assigned id.
func (s *SchedulerV2) AddRequest(promptTokens []int, maxNewTokens, stopToken int, hasStopToken bool) SeqID {
	s.mu.Lock()
	blockSize := s.pool.BlockSize
	s.nextSeqID++
	id := s.nextSeqID
	s.mu.Unlock()
	seq := NewSequence(id, promptTokens, blockSize, maxNewTokens, stopToken, hasStopToken)
	s.AddSequence(seq)
	return id
}

// GetSequence returns the sequence for an id.
func (s *SchedulerV2) GetSequence(id SeqID) (*Sequence, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	seq, ok := s.sequences[id]
	return seq, ok
}

// WaitingCount returns the number of waiting sequences.
func (s *SchedulerV2) WaitingCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.waiting)
}

// RunningCount returns the number of running sequences.
func (s *SchedulerV2) RunningCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.running)
}

// applyPrefillChunk allocates blocks and advances token counters for a chunk.
// Caller holds mu. Mirrors core.rs::apply_prefill_chunk.
func (s *SchedulerV2) applyPrefillChunk(seq *Sequence, chunkSize int) error {
	blocksNeeded := seq.blockTable.BlocksNeededForTokens(chunkSize)
	if blocksNeeded > 0 {
		ids, err := s.pool.AllocateBlocks(blocksNeeded)
		if err != nil {
			return err
		}
		for _, id := range ids {
			seq.blockTable.AppendBlock(id)
		}
	}
	for i := 0; i < chunkSize; i++ {
		seq.blockTable.AppendToken()
	}
	seq.recordPrefilled(chunkSize)
	return nil
}

// Step performs one scheduler step using the three-phase policy:
func (s *SchedulerV2) Step() (*SchedulerStepResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	budget := s.config.MaxBatchedTokens
	scheduled := make([]SeqID, 0, len(s.running)+len(s.waiting))
	prefillTokens := 0
	decodeTokens := 0
	seqPrefill := make(map[SeqID]int)
	seqDecode := make(map[SeqID]int)

	runningIDs := append([]SeqID(nil), s.running...)

	// --- Phase 1: decode for fully-prefilled running sequences ---
	for _, id := range runningIDs {
		seq, ok := s.sequences[id]
		if !ok {
			return nil, fmt.Errorf("paged: sequence %d not found", id)
		}
		if seq.IsFinished() || seq.RemainingPrefillTokens() > 0 {
			continue
		}
		if budget == 0 {
			break
		}
		needsBlock := seq.blockTable.AppendToken()
		if needsBlock {
			id2, err := s.pool.AllocateBlocks(1)
			if err != nil {
				return nil, err
			}
			seq.blockTable.AppendBlock(id2[0])
		} else {
			if err := s.cowDecodeBlockLocked(seq); err != nil {
				return nil, err
			}
		}
		scheduled = append(scheduled, id)
		budget--
		decodeTokens++
		seqDecode[id] = 1
	}

	// --- Phase 2: continue prefill for partially-prefilled running sequences ---
	for _, id := range runningIDs {
		seq := s.sequences[id]
		if seq == nil || seq.IsFinished() || seq.RemainingPrefillTokens() == 0 {
			continue
		}
		chunkSize := minInt(minInt(seq.RemainingPrefillTokens(), s.config.PrefillChunkSize), budget)
		if chunkSize == 0 {
			continue
		}
		if err := s.prefillChunkLocked(seq, chunkSize); err != nil {
			return nil, err
		}
		scheduled = append(scheduled, id)
		budget -= chunkSize
		prefillTokens += chunkSize
		seqPrefill[id] += chunkSize
	}

	// --- Phase 3: prefill chunks from the waiting queue (FCFS) ---
	stillWaiting := make([]SeqID, 0, len(s.waiting))
	runningCount := len(scheduled)
	for _, id := range s.waiting {
		seq, ok := s.sequences[id]
		if !ok {
			return nil, fmt.Errorf("paged: sequence %d not found", id)
		}
		if seq.status != StatusWaiting {
			stillWaiting = append(stillWaiting, id)
			continue
		}
		if runningCount >= s.config.MaxRunningSeqs {
			stillWaiting = append(stillWaiting, id)
			continue
		}
		remaining := seq.RemainingPrefillTokens()
		if remaining == 0 {
			seq.status = StatusRunning
			scheduled = append(scheduled, id)
			runningCount++
			continue
		}
		chunkSize := minInt(minInt(remaining, s.config.PrefillChunkSize), budget)
		if chunkSize == 0 {
			stillWaiting = append(stillWaiting, id)
			continue
		}
		if err := s.prefillChunkLocked(seq, chunkSize); err != nil {
			return nil, err
		}
		seq.status = StatusRunning
		scheduled = append(scheduled, id)
		runningCount++
		budget -= chunkSize
		prefillTokens += chunkSize
		seqPrefill[id] += chunkSize
	}

	s.waiting = stillWaiting
	s.running = append([]SeqID(nil), scheduled...)

	return &SchedulerStepResult{
		ScheduledSeqIDs:  scheduled,
		PrefillTokens:    prefillTokens,
		DecodeTokens:     decodeTokens,
		SeqPrefillTokens: seqPrefill,
		SeqDecodeTokens:  seqDecode,
	}, nil
}

// prefillChunkLocked dispatches to the prefix-cache-aware path when enabled.
func (s *SchedulerV2) prefillChunkLocked(seq *Sequence, chunkSize int) error {
	if s.usePrefixCache {
		_, err := s.applyPrefillChunkWithPrefixCacheLocked(seq, chunkSize)
		return err
	}
	return s.applyPrefillChunk(seq, chunkSize)
}

// BuildInputBatch flattens a step result into a single batch.
// Mirrors core.rs::build_input_batch.
func (s *SchedulerV2) BuildInputBatch(res *SchedulerStepResult) InputBatch {
	s.mu.Lock()
	defer s.mu.Unlock()

	batch := InputBatch{}
	for _, id := range res.ScheduledSeqIDs {
		seq, ok := s.sequences[id]
		if !ok {
			continue
		}
		prefillCount := res.SeqPrefillTokens[id]
		decodeCount := res.SeqDecodeTokens[id]

		if prefillCount > 0 {
			start := seq.numPrefilled - prefillCount
			if start < 0 {
				start = 0
			}
			end := start + prefillCount
			if end > len(seq.promptTokens) {
				end = len(seq.promptTokens)
			}
			chunk := append([]int(nil), seq.promptTokens[start:end]...)
			pos := make([]int, 0, end-start)
			for p := start; p < end; p++ {
				pos = append(pos, p)
			}
			batch.SeqIDs = append(batch.SeqIDs, id)
			batch.TokenIDs = append(batch.TokenIDs, chunk)
			batch.Positions = append(batch.Positions, pos)
			batch.BlockTables = append(batch.BlockTables, append([]int(nil), seq.blockTable.PhysicalBlocks()...))
			batch.NumTokens = append(batch.NumTokens, prefillCount)
			batch.IsPrefill = append(batch.IsPrefill, true)
			batch.ContextLens = append(batch.ContextLens, seq.numPrefilled)
			batch.TotalTokens += prefillCount
		} else if decodeCount > 0 && !seq.IsFinished() {
			decodePos := seq.NumTokens() - 1
			if decodePos < 0 {
				decodePos = 0
			}
			tok := []int{}
			if len(seq.generatedTokens) > 0 {
				tok = []int{seq.generatedTokens[len(seq.generatedTokens)-1]}
			}
			batch.SeqIDs = append(batch.SeqIDs, id)
			batch.TokenIDs = append(batch.TokenIDs, tok)
			batch.Positions = append(batch.Positions, []int{decodePos})
			batch.BlockTables = append(batch.BlockTables, append([]int(nil), seq.blockTable.PhysicalBlocks()...))
			batch.NumTokens = append(batch.NumTokens, 1)
			batch.IsPrefill = append(batch.IsPrefill, false)
			batch.ContextLens = append(batch.ContextLens, seq.NumTokens())
			batch.TotalTokens++
		}
	}
	batch.BatchSize = len(batch.SeqIDs)
	return batch
}

// FindPrefixCacheHits returns how many prompt tokens of a sequence can be served
// from the prefix cache. Mirrors prefix_cache.rs::find_prefix_cache_hits.
func (s *SchedulerV2) FindPrefixCacheHits(id SeqID) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	seq, ok := s.sequences[id]
	if !ok {
		return 0, fmt.Errorf("paged: sequence %d not found", id)
	}
	return s.findPrefixCacheHitsLocked(seq), nil
}

func (s *SchedulerV2) findPrefixCacheHitsLocked(seq *Sequence) int {
	prompt := seq.promptTokens
	if len(prompt) == 0 {
		return 0
	}
	blockSize := seq.blockTable.BlockSize()
	cached := 0
	numBlocks := (len(prompt) + blockSize - 1) / blockSize
	for blockIdx := 0; blockIdx < numBlocks; blockIdx++ {
		blockEnd := minInt((blockIdx+1)*blockSize, len(prompt))
		hash := ComputeBlockHash(prompt[:blockEnd])
		if _, found := s.pool.LookupPrefixCache(hash); found {
			cached = blockEnd
		} else {
			break
		}
	}
	return cached
}

// applyPrefillChunkWithPrefixCacheLocked allocates/reuses blocks for a chunk,
// sharing cached prefix blocks and returning the count of newly-computed
// tokens. Mirrors prefix_cache.rs::apply_prefill_chunk_with_prefix_cache.
func (s *SchedulerV2) applyPrefillChunkWithPrefixCacheLocked(seq *Sequence, chunkSize int) (int, error) {
	prompt := seq.promptTokens
	blockSize := seq.blockTable.BlockSize()
	alreadyPrefilled := seq.numPrefilled
	thisChunk := minInt(seq.RemainingPrefillTokens(), chunkSize)
	if thisChunk == 0 {
		return 0, nil
	}

	// Compute total cached prefix length for this prompt.
	cachedTotal := 0
	if len(prompt) > 0 {
		numBlocks := (len(prompt) + blockSize - 1) / blockSize
		for blockIdx := 0; blockIdx < numBlocks; blockIdx++ {
			blockEnd := minInt((blockIdx+1)*blockSize, len(prompt))
			hash := ComputeBlockHash(prompt[:blockEnd])
			if _, found := s.pool.LookupPrefixCache(hash); found {
				cachedTotal = blockEnd
			} else {
				break
			}
		}
	}

	chunkEnd := alreadyPrefilled + thisChunk
	cachedInChunk := 0
	if cachedTotal > alreadyPrefilled {
		cachedInChunk = minInt(cachedTotal, chunkEnd) - alreadyPrefilled
		if cachedInChunk < 0 {
			cachedInChunk = 0
		}
	}
	newTokens := thisChunk - cachedInChunk

	// Ensure block table has physical blocks for all tokens up to chunkEnd.
	targetBlocks := (chunkEnd + blockSize - 1) / blockSize
	currentBlocks := seq.blockTable.NumBlocks()
	for blockIdx := currentBlocks; blockIdx < targetBlocks; blockIdx++ {
		blockEnd := minInt((blockIdx+1)*blockSize, len(prompt))
		hash := ComputeBlockHash(prompt[:blockEnd])
		var blockID int
		if blockEnd <= cachedTotal {
			if cid, found := s.pool.LookupPrefixCache(hash); found {
				if err := s.pool.IncRef(cid); err != nil {
					return 0, err
				}
				blockID = cid
			} else {
				ids, err := s.pool.AllocateBlocks(1)
				if err != nil {
					return 0, err
				}
				blockID = ids[0]
			}
		} else {
			ids, err := s.pool.AllocateBlocks(1)
			if err != nil {
				return 0, err
			}
			blockID = ids[0]
		}
		seq.blockTable.AppendBlock(blockID)
	}

	// Advance token counters.
	for i := 0; i < thisChunk; i++ {
		seq.blockTable.AppendToken()
	}
	seq.recordPrefilled(thisChunk)

	// Insert newly-computed blocks into the prefix cache.
	for blockIdx := 0; blockIdx < targetBlocks; blockIdx++ {
		blockEnd := minInt((blockIdx+1)*blockSize, len(prompt))
		if blockEnd > cachedTotal {
			hash := ComputeBlockHash(prompt[:blockEnd])
			if pid, found := seq.blockTable.GetPhysicalBlock(blockIdx); found {
				s.pool.InsertPrefixCache(hash, pid)
			}
		}
	}

	return newTokens, nil
}

// CowDecodeBlock triggers copy-on-write on a sequence's last block if shared.
// Mirrors prefix_cache.rs::cow_decode_block.
func (s *SchedulerV2) CowDecodeBlock(id SeqID) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	seq, ok := s.sequences[id]
	if !ok {
		return false, fmt.Errorf("paged: sequence %d not found", id)
	}
	err := s.cowDecodeBlockLocked(seq)
	if err != nil {
		return false, err
	}
	return s.lastCowTriggered, nil
}

func (s *SchedulerV2) cowDecodeBlockLocked(seq *Sequence) error {
	s.lastCowTriggered = false
	lastLogical := seq.blockTable.NumBlocks() - 1
	if lastLogical < 0 {
		return nil
	}
	originalID, ok := seq.blockTable.GetPhysicalBlock(lastLogical)
	if !ok {
		return nil
	}
	newID, copied, err := s.pool.CopyOnWrite(originalID)
	if err != nil {
		return err
	}
	if copied {
		seq.blockTable.SetPhysicalBlock(lastLogical, newID)
		s.lastCowTriggered = true
	}
	return nil
}

// PostprocessStep appends sampled tokens, detects finished sequences, and
// reclaims their blocks. Mirrors lifecycle.rs::postprocess_step.
func (s *SchedulerV2) PostprocessStep(sampled map[SeqID]int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	for id, token := range sampled {
		seq, ok := s.sequences[id]
		if !ok || seq.status != StatusRunning {
			continue
		}
		seq.appendToken(token)
	}

	var finished []SeqID
	for _, id := range s.running {
		seq, ok := s.sequences[id]
		if !ok {
			continue
		}
		if seq.IsFinished() {
			finished = append(finished, id)
		}
	}
	for _, id := range finished {
		if err := s.finishSequenceLocked(id); err != nil {
			return err
		}
	}
	return nil
}

func (s *SchedulerV2) finishSequenceLocked(id SeqID) error {
	seq, ok := s.sequences[id]
	if !ok {
		return fmt.Errorf("paged: sequence %d not found", id)
	}
	seq.status = StatusFinished
	for _, b := range seq.blockTable.PhysicalBlocks() {
		if err := s.pool.DecRef(b); err != nil {
			return err
		}
	}
	s.running = removeID(s.running, id)
	return nil
}

// FinishSequence marks a sequence finished and reclaims its blocks.
func (s *SchedulerV2) FinishSequence(id SeqID) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.finishSequenceLocked(id)
}

// PreemptSequence frees a sequence's blocks, resets its prefill state, and
// returns it to the front of the waiting queue. Mirrors
// prefix_cache.rs::preempt_sequence.
func (s *SchedulerV2) PreemptSequence(id SeqID) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	seq, ok := s.sequences[id]
	if !ok {
		return fmt.Errorf("paged: sequence %d not found", id)
	}
	for _, b := range seq.blockTable.PhysicalBlocks() {
		if err := s.pool.DecRef(b); err != nil {
			return err
		}
	}
	seq.blockTable = NewSeqBlockTable(seq.blockTable.BlockSize())
	seq.numPrefilled = 0
	seq.status = StatusWaiting
	s.running = removeID(s.running, id)
	if !containsID(s.waiting, id) {
		s.waiting = append([]SeqID{id}, s.waiting...)
	}
	return nil
}

// RemoveSequence removes a sequence entirely, freeing blocks if still running.
// Mirrors lifecycle.rs::remove_sequence.
func (s *SchedulerV2) RemoveSequence(id SeqID) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	seq, ok := s.sequences[id]
	if !ok {
		return fmt.Errorf("paged: sequence %d not found", id)
	}
	delete(s.sequences, id)
	if seq.status == StatusRunning {
		for _, b := range seq.blockTable.PhysicalBlocks() {
			if err := s.pool.DecRef(b); err != nil {
				return err
			}
		}
		s.running = removeID(s.running, id)
	}
	s.waiting = removeID(s.waiting, id)
	return nil
}

// InvalidatePrefixCache clears the prefix cache (e.g. on model switch).
func (s *SchedulerV2) InvalidatePrefixCache() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.pool.ClearPrefixCache()
}

// DrainAndReinitialize frees all blocks, clears the prefix cache, and resets the
// scheduler so it can accept a new backend/model. Mirrors
// lifecycle.rs::drain_and_reinitialize.
func (s *SchedulerV2) DrainAndReinitialize() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, seq := range s.sequences {
		for _, b := range seq.blockTable.PhysicalBlocks() {
			// Ignore errors: blocks may already be free.
			_ = s.pool.DecRef(b)
		}
	}
	s.pool.ClearPrefixCache()
	s.sequences = make(map[SeqID]*Sequence)
	s.waiting = nil
	s.running = nil
	s.nextArrival = 0
	return nil
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func removeID(ids []SeqID, id SeqID) []SeqID {
	out := ids[:0]
	for _, v := range ids {
		if v != id {
			out = append(out, v)
		}
	}
	return out
}

func containsID(ids []SeqID, id SeqID) bool {
	for _, v := range ids {
		if v == id {
			return true
		}
	}
	return false
}
