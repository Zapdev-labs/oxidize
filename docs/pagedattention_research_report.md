# PagedAttention Research Report — For Rust Implementation

## 1. How PagedAttention Works Conceptually

### 1.1 The Problem
In LLM serving, the KV cache (key-value tensors from attention layers) is the dominant memory consumer after model weights. Existing systems store each request's KV cache in a **contiguous** memory block pre-allocated to the maximum sequence length. This causes severe waste:
- **Internal fragmentation**: allocated slots never used because actual output length < max length
- **Reservation**: the entire block is reserved for the request lifetime even if mostly empty
- **External fragmentation**: variable-size blocks leave unusable gaps between them

Only ~20–40% of KV cache memory actually stores token states in naive systems.

### 1.2 The OS Analogy
PagedAttention is inspired by **virtual memory paging** in operating systems:
- **Blocks** ≈ pages (fixed-size memory units)
- **Tokens** ≈ bytes
- **Sequences** ≈ processes
- **Block table** ≈ page table (logical → physical mapping)

### 1.3 Block-Based KV Cache
Instead of one contiguous buffer per sequence, the KV cache is divided into **fixed-size blocks** (default 16 tokens). Each block contains the keys and values for that token slice. During attention, a custom kernel fetches these blocks via the block table.

**Key insight**: Blocks do **not** need to be contiguous in physical memory. A sequence's logical blocks are mapped to arbitrary physical blocks through a per-sequence **block table**.

### 1.4 Benefits
1. **Near-zero external fragmentation** — all blocks are same size
2. **Minimal internal fragmentation** — only the last block of a sequence can be partially filled (<4% waste)
3. **Flexible sharing** — multiple sequences can map their logical blocks to the same physical block (Copy-on-Write)
4. **On-demand allocation** — blocks are allocated only as tokens are generated

---

## 2. Key Data Structures

### 2.1 Block Table
Each sequence maintains a **block table**: an array mapping logical block indices → physical block indices.

```
Logical:  [0] [1] [2] [3] [4] ...
              ↓   ↓   ↓   ↓   ↓
Physical:  #7  #3  #12 #45 #2  ... (non-contiguous)
```

During attention, the kernel uses this table to gather the correct KV blocks from the paged KV cache pool.

### 2.2 Physical Block Pool (Free Block Queue)
The KV cache manager maintains a pool of pre-allocated physical blocks. In vLLM V1 this is a **doubly-linked free block queue** (`free_block_queue`). The number of blocks is computed at initialization based on:
- GPU VRAM available (after model weights + activations)
- Block size (default 16 tokens)
- Per-block memory: `2 * block_size * num_kv_heads * head_size * dtype_bytes`

### 2.3 Block Table per Sequence / Request
In vLLM V1, the scheduler's KV cache manager maintains:
- `req_to_blocks: dict[request_id, list[physical_block_ids]]`
- Reference counts per physical block (for sharing / COW)
- Hash values per block (for prefix caching)

### 2.4 Sequence / Request State
The scheduler tracks requests with states:
- **WAITING** — in the waiting queue, not yet started
- **RUNNING** — actively being processed in a batch
- **FINISHED** — completed, blocks returned to free pool

### 2.5 InputBatch (V1)
CPU-side structure holding:
- `input_ids`, `positions` buffers
- Block tables for KV-cache indexing
- Sampling metadata

---

## 3. Continuous Batching with PagedAttention

### 3.1 What is Continuous Batching?
Traditional batching waits for all requests in a batch to finish before starting a new batch. **Continuous batching** (also called "in-flight batching" or "iteration-level scheduling") adds new requests and removes finished ones **between every forward step**.

### 3.2 How vLLM Enables It
Because PagedAttention uses block tables, sequences can be added/removed from a batch dynamically without requiring contiguous memory reshuffling. The forward pass:
1. **Flattens** all sequences into a single "super sequence"
2. Uses **position indices** and attention masks so each sequence only attends to its own tokens
3. Custom kernels handle non-contiguous KV blocks via the block table

This means there is **no right-padding waste** — every token position in the batch is doing real work.

### 3.3 Engine Step Loop (V1)
```
while requests exist:
  1. SCHEDULE: pick which requests to run (decode + prefill chunks)
  2. FORWARD PASS: run model, sample tokens
  3. POSTPROCESS: append tokens, check stop conditions, free finished blocks
```

New requests can be injected after each step. Finished requests have their blocks returned to the free pool immediately.

---

## 4. Prefix Caching (Automatic Prefix Caching)

### 4.1 The Idea
Many requests share common prefixes (e.g., system prompts, few-shot examples, code prefixes). Recomputing the KV cache for these shared tokens wastes GPU compute.

### 4.2 Block-Level Hashing
Each KV block is uniquely identified by:
```
hash(prefix_tokens + block_tokens) <-> KV Block
```

Where `prefix_tokens` are all tokens before this block, and `block_tokens` are the tokens within the block.

Example:
```
Block 1: "A gentle breeze stirred"  -> hash("A gentle breeze stirred")
Block 2: "the leaves as children" -> hash("A gentle breeze stirred" + "the leaves as children")
Block 3: "laughed in the distance" -> hash("A gentle breeze stirred the leaves as children" + "laughed in the distance")
```

### 4.3 Global Hash Table
Instead of per-sequence logical→physical mapping, prefix caching adds an indirection:
- Logical blocks map to their **hash value**
- A **global hash table** maps hash → physical block
- Multiple sequences sharing the same prefix map to the **same physical block**

### 4.4 Reference Counting & Copy-on-Write
When a sequence writes to a shared block, if the block's reference count > 1:
1. The block is **copied** to a new physical block (COW)
2. The sequence's block table is updated
3. The original block's reference count is decremented

### 4.5 Eviction Policy
When the cache is full:
1. Evict blocks with **reference count = 0** first
2. Among those, use **LRU** (least recently used)
3. Tie-breaker: evict the block at the end of the **longest prefix** (maximum number of blocks before it)

This effectively matches RadixAttention's leaf-node eviction policy without requiring a tree structure.

---

## 5. Chunked Prefill

### 5.1 The Problem
A long prefill (e.g., 4096 tokens) takes significant compute time. If decode requests are waiting behind it, inter-token latency (ITL) spikes because decodes are starved.

### 5.2 The Solution
**Chunked prefill** splits long prefills into smaller chunks (e.g., 512 tokens each) and interleaves them with decode steps.

In vLLM V1:
- The scheduling policy **prioritizes decode requests**
- It batches all pending decodes first
- Then fills remaining token budget with prefill chunks

### 5.3 Token Budget
The scheduler uses a **token budget** to decide how many tokens to process per step:
1. First, allocate tokens for all **running decode** requests
2. Then, allocate tokens for **prefill chunks** from waiting requests
3. If budget exhausted, remaining prefills wait

This keeps ITL low while still making progress on long prefills.

### 5.4 Budget Calculation
```
budget = max_num_batched_tokens (configurable, default large)
for each running decode: budget -= num_new_tokens (usually 1)
for each prefill request from waiting queue:
  if budget >= chunk_size: schedule chunk, budget -= chunk_size
  else: skip
```

### 5.5 Benefits
- Decode requests are never blocked by long prefills
- GPU utilization stays high (prefills are compute-bound, decodes are memory-bound — mixing them balances the two)
- Time-to-first-token (TTFT) for long prompts increases slightly, but overall throughput improves

---

## 6. Scheduler Architecture in vLLM

### 6.1 V1 Scheduler Components
The vLLM V1 scheduler lives in the **Engine Core** and has:

1. **Policy setting**: FCFS (first-come-first-served) or Priority-based
2. **Waiting queue**: requests that haven't started yet
3. **Running queue**: requests currently in a batch
4. **KV Cache Manager**: allocates/frees blocks, handles prefix caching, COW

### 6.2 Scheduling Flow per Step
```
1. For each RUNNING decode request:
   a. Compute new tokens needed
   b. Call kv_cache_manager.allocate_slots()
   c. If insufficient blocks → trigger preemption (evict waiting/low-priority)
   d. Subtract from token budget

2. For each WAITING prefill request (in order):
   a. Check prefix cache for computed blocks
   b. Call allocate_slots for remaining tokens (up to chunk size)
   c. If budget exhausted → stop scheduling prefills
   d. Move request to RUNNING
   e. Subtract from token budget

3. Return the scheduled batch
```

### 6.3 Preemption
When memory is exhausted:
- **Recompute preemption** (V1 default): evict a running request's KV blocks, return them to the free pool. The request goes back to WAITING and will be recomputed from scratch (or from cached prefix).
- **Swap preemption** (V0 only): swap blocks to CPU memory. Not used in V1.

### 6.4 allocate_slots Details
```
1. Compute num_new_blocks = ceil(new_tokens / block_size)
2. Check if free_block_queue has enough blocks
3. If not enough: return failure → scheduler may preempt or skip request
4. If enough: pop n blocks from free queue, append to req_to_blocks[request_id]
```

### 6.5 Forward Pass Execution
After scheduling:
1. **Update states**: prune finished requests, update block tables in InputBatch
2. **Prepare inputs**: CPU→GPU copy, compute positions, build `slot_mapping`, construct attention metadata
3. **Run model**: custom paged attention kernel
4. **Gather & sample**: extract last-token logits, sample next tokens
5. **Postprocess**: append tokens, check stop conditions, free blocks for finished requests

### 6.6 Multi-Process Architecture (V1)
```
API Server (HTTP)  →  Engine Core (scheduler)  →  GPU Workers (model forward)
     ↑                                              ↑
     └─────────────  EngineCoreOutputs  ────────────┘
```
- Engine Core runs the scheduler loop
- GPU Workers execute model forward passes
- Communication via shared memory / message passing

---

## 7. Reference Implementations & Papers

### 7.1 Primary Paper
**"Efficient Memory Management for Large Language Model Serving with PagedAttention"**
- Authors: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica
- Venue: SOSP 2023
- arXiv: https://arxiv.org/abs/2309.06180
- DOI: 10.1145/3600006.3613165

### 7.2 Key vLLM References
- **vLLM Blog**: https://blog.vllm.ai/2023/06/20/vllm.html
- **vLLM Architecture Docs**: https://docs.vllm.ai/en/latest/design/arch_overview/
- **Paged Attention Docs**: https://docs.vllm.ai/en/latest/design/paged_attention/
- **Prefix Caching Docs**: https://docs.vllm.ai/en/v0.7.3/design/automatic_prefix_caching.html
- **Optimization Guide**: https://docs.vllm.ai/en/stable/configuration/optimization/

### 7.3 Detailed Architecture Blog
**"Inside vLLM: Anatomy of a High-Throughput LLM Inference System"** by Aleksa Gordic
- https://www.aleksagordic.com/blog/vllm
- Covers: engine core, scheduling, chunked prefill, prefix caching, speculative decoding, disaggregated P/D, multi-GPU serving
- Based on vLLM V1 (commit 42172ad, August 2025)

### 7.4 Rust Implementations
1. **rvLLM** (m0at/rvllm) — "High-performance LLM inference in Rust. Drop-in vLLM replacement."
   - GitHub: https://github.com/m0at/rvllm
   - Very early / experimental (as of research date)

2. **mistral.rs** (EricLBuehler) — Fast, flexible LLM inference in Rust
   - GitHub: https://github.com/EricLBuehler/mistral.rs
   - Uses candle (HuggingFace's Rust ML framework)
   - Has PagedAttention-like features (variable batch size, KV cache management)

3. **Awesome Rust LLM** curated list
   - GitHub: https://github.com/jondot/awesome-rust-llm

### 7.5 Related Papers
- **Orca** (OSDI 2022) — Iteration-level scheduling for LLMs (Yu et al.)
- **FlashAttention** (NeurIPS 2022) — IO-aware exact attention
- **FlashAttention-2** — Further optimizations
- **SGLang / RadixAttention** — Prefix caching with radix tree (LMSYS, 2024)
- **vAttention** (arXiv 2024) — Dynamic memory management without pre-allocation

---

## 8. Implementation Notes for Rust

### 8.1 Core Components to Build
1. **Block Allocator**
   - Fixed-size block pool
   - Free list / queue for unused blocks
   - Reference counting for shared blocks
   - Hash-based lookup for prefix caching

2. **Block Table**
   - Per-sequence logical → physical mapping
   - Support for non-contiguous physical blocks
   - COW mechanism when writing to shared blocks

3. **Scheduler**
   - Waiting and running queues
   - Token budget management
   - FCFS or priority policy
   - Chunked prefill logic
   - Preemption (recompute strategy)

4. **Paged Attention Kernel Interface**
   - Block table → physical memory indexing
   - Position computation for flattened batches
   - Attention mask / causal mask for mixed batch

5. **KV Cache Tensor Layout**
   - `[num_blocks, block_size, num_kv_heads, head_size]` or similar
   - Block table indices used to gather KV vectors at attention time

### 8.2 Data Structure Sketch (Rust)
```rust
// Physical block pool
pub struct BlockPool {
    blocks: Vec<KvBlock>,           // all physical blocks
    free_list: Vec<BlockId>,        // available blocks
    global_hash: HashMap<BlockHash, BlockId>, // prefix cache
}

// Per-sequence block table
pub struct BlockTable {
    logical_to_physical: Vec<BlockId>,
    num_tokens: usize,              // how many tokens actually used in last block
}

// Request / Sequence
pub struct Sequence {
    seq_id: SeqId,
    block_table: BlockTable,
    status: SequenceStatus,         // Waiting / Running / Finished
    num_computed_tokens: usize,
}

// Scheduler
pub struct Scheduler {
    waiting: VecDeque<SeqId>,
    running: Vec<SeqId>,
    token_budget: usize,
    block_pool: BlockPool,
    policy: SchedulingPolicy,         // FCFS or Priority
}
```

### 8.3 Key Invariants
- A physical block is either in the free list or assigned to one or more sequences
- When `ref_count > 1`, any write triggers COW
- The scheduler must never over-commit the token budget for a step
- Block size is fixed (typically 16); last block may be partially filled
- Prefix cache hashes must include ALL preceding tokens (not just block contents)

### 8.4 Integration with Attention Kernels
The paged attention kernel needs:
- `block_tables`: `[batch_size, max_num_blocks_per_seq]` int tensor
- `slot_mapping`: flat index for each token → `[layer, block_id, slot_in_block]`
- The KV cache tensor: `[num_layers, 2, num_blocks, block_size, num_heads, head_size]`

For CPU-only Rust implementations, a similar gather-based approach can be used with standard matrix operations.
