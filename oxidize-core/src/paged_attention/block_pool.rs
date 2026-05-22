use crate::tensor::DType;

/// Unique identifier for a physical block in the pool.
pub type BlockId = usize;

/// A physical KV block managed by the [`BlockPool`].
///
/// Each physical block has a reference count so that multiple sequences can
/// share the same block (used for prefix caching). When a write is attempted
/// on a block with `ref_count > 1`, copy-on-write triggers: a new physical
/// block is allocated, the data is copied, and the sequence's block table is
/// updated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalBlock {
    pub id: BlockId,
    pub ref_count: usize,
}

impl PhysicalBlock {
    /// Create a new physical block with the given id.
    pub fn new(id: BlockId) -> Self {
        Self { id, ref_count: 0 }
    }

    /// Increment the reference count.
    pub fn inc_ref(&mut self) {
        self.ref_count = self.ref_count.saturating_add(1);
    }

    /// Decrement the reference count, returning the new count.
    pub fn dec_ref(&mut self) -> usize {
        self.ref_count = self.ref_count.saturating_sub(1);
        self.ref_count
    }
}

/// Configuration for the [`BlockPool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockPoolConfig {
    /// Number of tokens per block. Default is 16.
    pub block_size: usize,
    /// Total number of physical blocks in the pool.
    pub num_blocks: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of KV heads per layer.
    pub num_kv_heads: usize,
    /// Dimension of each KV head.
    pub head_dim: usize,
    /// Data type of KV tensors.
    pub dtype: DType,
}

impl Default for BlockPoolConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            num_blocks: 0,
            num_layers: 0,
            num_kv_heads: 0,
            head_dim: 0,
            dtype: DType::F32,
        }
    }
}

impl BlockPoolConfig {
    /// Return the number of tokens each physical block can hold.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Return the size in bytes of a single physical block.
    pub fn block_bytes(&self) -> usize {
        let tokens_per_block = self.block_size;
        let kv_pairs = 2usize; // key + value
        let elements_per_block = tokens_per_block
            .saturating_mul(self.num_layers)
            .saturating_mul(kv_pairs)
            .saturating_mul(self.num_kv_heads)
            .saturating_mul(self.head_dim);
        elements_per_block.saturating_mul(self.dtype.size_in_bytes())
    }
}

/// The block pool manages a fixed set of physical KV blocks.
///
/// Blocks are allocated on-demand from a free list. When a sequence no longer
/// needs a block, it is returned to the free list. Shared blocks (used for
/// prefix caching) are tracked via reference counting on [`PhysicalBlock`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockPool {
    config: BlockPoolConfig,
    blocks: Vec<PhysicalBlock>,
    free_list: Vec<BlockId>,
}

/// Error type for block pool operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockPoolError {
    /// No free blocks remain in the pool.
    OutOfBlocks,
    /// The requested block id is invalid.
    InvalidBlockId { id: BlockId },
    /// Attempted to free a block that is not allocated.
    BlockNotAllocated { id: BlockId },
}

impl std::fmt::Display for BlockPoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockPoolError::OutOfBlocks => write!(f, "block pool exhausted: no free blocks"),
            BlockPoolError::InvalidBlockId { id } => {
                write!(f, "invalid block id: {id}")
            }
            BlockPoolError::BlockNotAllocated { id } => {
                write!(f, "block {id} is not currently allocated")
            }
        }
    }
}

impl std::error::Error for BlockPoolError {}

impl BlockPool {
    /// Create a new block pool with the given configuration.
    ///
    /// All physical blocks are initialized and placed on the free list.
    pub fn new(config: BlockPoolConfig) -> Self {
        let num_blocks = config.num_blocks;
        let mut blocks = Vec::with_capacity(num_blocks);
        let mut free_list = Vec::with_capacity(num_blocks);
        for id in 0..num_blocks {
            blocks.push(PhysicalBlock::new(id));
            free_list.push(id);
        }
        Self {
            config,
            blocks,
            free_list,
        }
    }

    /// Return the pool configuration.
    pub fn config(&self) -> &BlockPoolConfig {
        &self.config
    }

    /// Return the number of free blocks remaining.
    pub fn free_block_count(&self) -> usize {
        self.free_list.len()
    }

    /// Return the total number of blocks in the pool.
    pub fn total_block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Return the number of currently allocated (non-free) blocks.
    pub fn allocated_block_count(&self) -> usize {
        self.blocks.len().saturating_sub(self.free_list.len())
    }

    /// Allocate a single physical block from the free list.
    ///
    /// The block's reference count is set to 1.
    pub fn allocate_block(&mut self) -> Result<BlockId, BlockPoolError> {
        let id = self
            .free_list
            .pop()
            .ok_or(BlockPoolError::OutOfBlocks)?;
        let block = self
            .blocks
            .get_mut(id)
            .ok_or(BlockPoolError::InvalidBlockId { id })?;
        block.ref_count = 1;
        Ok(id)
    }

    /// Allocate `n` physical blocks from the free list.
    ///
    /// If insufficient blocks are available, an error is returned and **no**
    /// blocks are allocated (all-or-nothing).
    pub fn allocate_blocks(&mut self, n: usize) -> Result<Vec<BlockId>, BlockPoolError> {
        if n == 0 {
            return Ok(Vec::new());
        }
        if self.free_list.len() < n {
            return Err(BlockPoolError::OutOfBlocks);
        }
        let mut ids = Vec::with_capacity(n);
        for _ in 0..n {
            let id = self.free_list.pop().expect("checked above");
            let block = self
                .blocks
                .get_mut(id)
                .ok_or(BlockPoolError::InvalidBlockId { id })?;
            block.ref_count = 1;
            ids.push(id);
        }
        Ok(ids)
    }

    /// Free a single physical block, returning it to the free list.
    ///
    /// The block's reference count must be zero (or will be set to zero).
    pub fn free_block(&mut self, id: BlockId) -> Result<(), BlockPoolError> {
        // Validate id first.
        if self.blocks.get(id).is_none() {
            return Err(BlockPoolError::InvalidBlockId { id });
        }
        let already_free = self.is_free(id);
        let block = self.blocks.get_mut(id).unwrap();
        block.ref_count = 0;
        if !already_free {
            self.free_list.push(id);
        }
        Ok(())
    }

    /// Free multiple physical blocks, returning them to the free list.
    pub fn free_blocks(&mut self, ids: &[BlockId]) -> Result<(), BlockPoolError> {
        for &id in ids {
            self.free_block(id)?;
        }
        Ok(())
    }

    /// Get a reference to a physical block by id.
    pub fn get_block(&self, id: BlockId) -> Option<&PhysicalBlock> {
        self.blocks.get(id)
    }

    /// Get a mutable reference to a physical block by id.
    pub fn get_block_mut(&mut self, id: BlockId) -> Option<&mut PhysicalBlock> {
        self.blocks.get_mut(id)
    }

    /// Check whether a block is currently on the free list.
    fn is_free(&self, id: BlockId) -> bool {
        self.free_list.contains(&id)
    }

    /// Increment the reference count of a physical block.
    ///
    /// Used when a sequence shares an already-allocated block (e.g. prefix
    /// cache hit).
    pub fn inc_ref(&mut self, id: BlockId) -> Result<(), BlockPoolError> {
        let block = self
            .blocks
            .get_mut(id)
            .ok_or(BlockPoolError::InvalidBlockId { id })?;
        if block.ref_count == 0 {
            return Err(BlockPoolError::BlockNotAllocated { id });
        }
        block.inc_ref();
        Ok(())
    }

    /// Decrement the reference count of a physical block.
    ///
    /// If the count reaches zero, the block is automatically returned to the
    /// free list.
    pub fn dec_ref(&mut self, id: BlockId) -> Result<(), BlockPoolError> {
        let block = self
            .blocks
            .get_mut(id)
            .ok_or(BlockPoolError::InvalidBlockId { id })?;
        if block.ref_count == 0 {
            return Err(BlockPoolError::BlockNotAllocated { id });
        }
        let new_count = block.dec_ref();
        if new_count == 0 && !self.is_free(id) {
            self.free_list.push(id);
        }
        Ok(())
    }

    /// Copy-on-write: when a sequence wants to write to a shared block,
    /// allocate a new block and return its id. The caller is responsible for
    /// copying KV data.
    ///
    /// The original block's reference count is decremented. If the original
    /// block is not actually shared (`ref_count == 1`), this returns `Ok(None)`
    /// because no copy is needed.
    pub fn copy_on_write(&mut self, id: BlockId) -> Result<Option<BlockId>, BlockPoolError> {
        let block = self
            .blocks
            .get(id)
            .ok_or(BlockPoolError::InvalidBlockId { id })?;
        if block.ref_count <= 1 {
            return Ok(None);
        }
        // Allocate a new block.
        let new_id = self.allocate_block()?;
        // Decrement the original block's ref count.
        self.dec_ref(id)?;
        Ok(Some(new_id))
    }
}

/// Per-sequence logical → physical block mapping.
///
/// A sequence's tokens are divided into fixed-size logical blocks. The
/// `BlockTable` maps each logical block index to a physical block id from the
/// [`BlockPool`]. Physical blocks need not be contiguous.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockTable {
    /// Mapping from logical block index → physical block id.
    logical_to_physical: Vec<BlockId>,
    /// Number of tokens currently stored in this sequence.
    num_tokens: usize,
    /// Number of tokens per physical block.
    block_size: usize,
}

impl BlockTable {
    /// Create an empty block table with the given block size.
    pub fn new(block_size: usize) -> Self {
        Self {
            logical_to_physical: Vec::new(),
            num_tokens: 0,
            block_size,
        }
    }

    /// Return the number of tokens in the sequence.
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// Return the number of logical (and therefore physical) blocks assigned.
    pub fn num_blocks(&self) -> usize {
        self.logical_to_physical.len()
    }

    /// Return the block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Return the physical block id for a given logical block index.
    pub fn get_physical_block(&self, logical_block: usize) -> Option<BlockId> {
        self.logical_to_physical.get(logical_block).copied()
    }

    /// Return a slice of all physical block ids.
    pub fn physical_blocks(&self) -> &[BlockId] {
        &self.logical_to_physical
    }

    /// Append a new physical block to the logical mapping.
    pub fn append_block(&mut self, physical_block_id: BlockId) {
        self.logical_to_physical.push(physical_block_id);
    }

    /// Remove the last `n` logical blocks, returning their physical ids.
    pub fn truncate_blocks(&mut self, n: usize) -> Vec<BlockId> {
        let split_at = self.logical_to_physical.len().saturating_sub(n);
        let removed = self.logical_to_physical.split_off(split_at);
        // Adjust token count to match remaining blocks.
        self.num_tokens = self
            .logical_to_physical
            .len()
            .saturating_mul(self.block_size);
        removed
    }

    /// Update the physical block mapping for a logical block (used during COW).
    pub fn set_physical_block(&mut self, logical_block: usize, physical_block_id: BlockId) {
        if let Some(slot) = self.logical_to_physical.get_mut(logical_block) {
            *slot = physical_block_id;
        }
    }

    /// Append a token, extending the block table if a new block is needed.
    ///
    /// Returns `true` if a new block was required (the caller should allocate
    /// one from the pool and call [`Self::append_block`]).
    pub fn append_token(&mut self) -> bool {
        let token_index_in_block = self.num_tokens % self.block_size;
        let needs_new_block = token_index_in_block == 0 && self.num_tokens > 0;
        self.num_tokens += 1;
        needs_new_block
    }

    /// Compute the number of additional blocks required to store `n` more
    /// tokens.
    pub fn blocks_needed_for_tokens(&self, n: usize) -> usize {
        let future_tokens = self.num_tokens.saturating_add(n);
        let future_blocks = future_tokens.div_ceil(self.block_size);
        let current_blocks = self.logical_to_physical.len();
        future_blocks.saturating_sub(current_blocks)
    }

    /// Compute the physical block id and slot index for a given token position.
    ///
    /// Returns `None` if the token position is beyond the current allocation.
    pub fn get_block_and_slot(&self, token_position: usize) -> Option<(BlockId, usize)> {
        if token_position >= self.num_tokens {
            return None;
        }
        let logical_block = token_position / self.block_size;
        let slot = token_position % self.block_size;
        self.get_physical_block(logical_block)
            .map(|block_id| (block_id, slot))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config(num_blocks: usize) -> BlockPoolConfig {
        BlockPoolConfig {
            block_size: 16,
            num_blocks,
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 64,
            dtype: DType::F32,
        }
    }

    #[test]
    fn block_pool_created_with_configurable_block_size() {
        let mut config = default_config(10);
        config.block_size = 8;
        let pool = BlockPool::new(config);
        assert_eq!(pool.config().block_size(), 8);
        assert_eq!(pool.total_block_count(), 10);
        assert_eq!(pool.free_block_count(), 10);
        assert_eq!(pool.allocated_block_count(), 0);
    }

    #[test]
    fn block_pool_default_block_size_is_16() {
        let pool = BlockPool::new(default_config(10));
        assert_eq!(pool.config().block_size(), 16);
    }

    #[test]
    fn blocks_allocated_from_free_list_on_demand() {
        let mut pool = BlockPool::new(default_config(5));
        let id1 = pool.allocate_block().expect("should allocate");
        let id2 = pool.allocate_block().expect("should allocate");
        assert_eq!(pool.free_block_count(), 3);
        assert_eq!(pool.allocated_block_count(), 2);
        assert_ne!(id1, id2);

        // Verify ref counts.
        assert_eq!(pool.get_block(id1).unwrap().ref_count, 1);
        assert_eq!(pool.get_block(id2).unwrap().ref_count, 1);
    }

    #[test]
    fn allocate_multiple_blocks_all_or_nothing() {
        let mut pool = BlockPool::new(default_config(3));
        let ids = pool.allocate_blocks(2).expect("should allocate 2");
        assert_eq!(ids.len(), 2);
        assert_eq!(pool.free_block_count(), 1);

        // Requesting more than available should fail.
        let err = pool.allocate_blocks(2).expect_err("should fail");
        assert_eq!(err, BlockPoolError::OutOfBlocks);

        // No blocks should have been allocated on failure.
        assert_eq!(pool.free_block_count(), 1);
    }

    #[test]
    fn out_of_blocks_error_when_empty() {
        let mut pool = BlockPool::new(default_config(1));
        pool.allocate_block().unwrap();
        let err = pool.allocate_block().expect_err("should fail");
        assert_eq!(err, BlockPoolError::OutOfBlocks);
    }

    #[test]
    fn freed_blocks_returned_to_free_list() {
        let mut pool = BlockPool::new(default_config(5));
        let id = pool.allocate_block().unwrap();
        assert_eq!(pool.free_block_count(), 4);

        pool.free_block(id).unwrap();
        assert_eq!(pool.free_block_count(), 5);
        assert_eq!(pool.allocated_block_count(), 0);
        assert_eq!(pool.get_block(id).unwrap().ref_count, 0);
    }

    #[test]
    fn freed_block_can_be_reallocated() {
        let mut pool = BlockPool::new(default_config(2));
        let id1 = pool.allocate_block().unwrap();
        pool.free_block(id1).unwrap();
        let id2 = pool.allocate_block().unwrap();
        assert_eq!(id1, id2, "free list is LIFO, same id should be reused");
    }

    #[test]
    fn physical_blocks_have_ref_count_for_sharing() {
        let mut pool = BlockPool::new(default_config(5));
        let id = pool.allocate_block().unwrap();
        assert_eq!(pool.get_block(id).unwrap().ref_count, 1);

        pool.inc_ref(id).unwrap();
        assert_eq!(pool.get_block(id).unwrap().ref_count, 2);

        pool.inc_ref(id).unwrap();
        assert_eq!(pool.get_block(id).unwrap().ref_count, 3);
    }

    #[test]
    fn dec_ref_returns_block_to_free_list_at_zero() {
        let mut pool = BlockPool::new(default_config(5));
        let id = pool.allocate_block().unwrap();
        pool.inc_ref(id).unwrap();
        assert_eq!(pool.get_block(id).unwrap().ref_count, 2);

        pool.dec_ref(id).unwrap();
        assert_eq!(pool.get_block(id).unwrap().ref_count, 1);
        assert_eq!(pool.free_block_count(), 4); // still allocated

        pool.dec_ref(id).unwrap();
        assert_eq!(pool.get_block(id).unwrap().ref_count, 0);
        assert_eq!(pool.free_block_count(), 5); // returned to free list
    }

    #[test]
    fn dec_ref_on_unallocated_block_fails() {
        let mut pool = BlockPool::new(default_config(3));
        let err = pool.dec_ref(0).expect_err("should fail");
        assert_eq!(err, BlockPoolError::BlockNotAllocated { id: 0 });
    }

    #[test]
    fn copy_on_write_allocates_new_block_when_shared() {
        let mut pool = BlockPool::new(default_config(5));
        let original = pool.allocate_block().unwrap();
        pool.inc_ref(original).unwrap(); // now shared
        assert_eq!(pool.get_block(original).unwrap().ref_count, 2);
        assert_eq!(pool.free_block_count(), 4); // 5 total - 1 allocated = 4 free

        let cow_result = pool.copy_on_write(original).unwrap();
        assert!(cow_result.is_some());
        let new_id = cow_result.unwrap();
        assert_ne!(new_id, original);
        assert_eq!(pool.get_block(original).unwrap().ref_count, 1);
        assert_eq!(pool.get_block(new_id).unwrap().ref_count, 1);
        assert_eq!(pool.free_block_count(), 3); // 5 total - 2 allocated = 3 free
    }

    #[test]
    fn copy_on_write_returns_none_when_not_shared() {
        let mut pool = BlockPool::new(default_config(3));
        let id = pool.allocate_block().unwrap();
        assert_eq!(pool.get_block(id).unwrap().ref_count, 1);

        let cow_result = pool.copy_on_write(id).unwrap();
        assert_eq!(cow_result, None);
    }

    #[test]
    fn copy_on_write_fails_when_pool_exhausted() {
        let mut pool = BlockPool::new(default_config(1));
        let id = pool.allocate_block().unwrap();
        pool.inc_ref(id).unwrap(); // shared, but pool has no spare blocks

        let err = pool.copy_on_write(id).expect_err("should fail");
        assert_eq!(err, BlockPoolError::OutOfBlocks);
    }

    #[test]
    fn block_table_maps_logical_to_physical() {
        let mut table = BlockTable::new(16);
        table.append_block(7);
        table.append_block(3);
        table.append_block(12);

        assert_eq!(table.get_physical_block(0), Some(7));
        assert_eq!(table.get_physical_block(1), Some(3));
        assert_eq!(table.get_physical_block(2), Some(12));
        assert_eq!(table.get_physical_block(3), None);
    }

    #[test]
    fn block_table_appends_token_and_requests_new_block() {
        let mut table = BlockTable::new(4);
        // First 4 tokens fit in the first block.
        assert!(!table.append_token());
        assert!(!table.append_token());
        assert!(!table.append_token());
        assert!(!table.append_token());
        assert_eq!(table.num_tokens(), 4);
        assert_eq!(table.num_blocks(), 0); // no block allocated yet

        // 5th token needs a new block.
        assert!(table.append_token());
        assert_eq!(table.num_tokens(), 5);
    }

    #[test]
    fn block_table_blocks_needed_for_tokens() {
        let mut table = BlockTable::new(16);
        table.append_block(0);
        table.append_block(1);
        table.num_tokens = 20; // 20 tokens = 2 blocks

        // Need 12 more tokens → 32 total → 2 blocks needed.
        assert_eq!(table.blocks_needed_for_tokens(12), 0);

        // Need 13 more tokens → 33 total → 3 blocks, need 1 more.
        assert_eq!(table.blocks_needed_for_tokens(13), 1);

        // Empty table, 17 tokens → 2 blocks.
        let empty = BlockTable::new(16);
        assert_eq!(empty.blocks_needed_for_tokens(17), 2);
    }

    #[test]
    fn block_table_get_block_and_slot() {
        let mut table = BlockTable::new(16);
        table.append_block(5);
        table.append_block(7);
        table.num_tokens = 20;

        assert_eq!(table.get_block_and_slot(0), Some((5, 0)));
        assert_eq!(table.get_block_and_slot(15), Some((5, 15)));
        assert_eq!(table.get_block_and_slot(16), Some((7, 0)));
        assert_eq!(table.get_block_and_slot(19), Some((7, 3)));
        assert_eq!(table.get_block_and_slot(20), None);
    }

    #[test]
    fn block_table_truncate_blocks() {
        let mut table = BlockTable::new(16);
        table.append_block(1);
        table.append_block(2);
        table.append_block(3);
        table.num_tokens = 48;

        let removed = table.truncate_blocks(2);
        assert_eq!(removed, vec![2, 3]);
        assert_eq!(table.num_blocks(), 1);
        assert_eq!(table.num_tokens(), 16);
    }

    #[test]
    fn block_table_set_physical_block() {
        let mut table = BlockTable::new(16);
        table.append_block(1);
        table.append_block(2);

        table.set_physical_block(1, 99);
        assert_eq!(table.get_physical_block(1), Some(99));
    }

    #[test]
    fn free_blocks_batch_returns_all_to_pool() {
        let mut pool = BlockPool::new(default_config(10));
        let ids = pool.allocate_blocks(5).unwrap();
        assert_eq!(pool.free_block_count(), 5);

        pool.free_blocks(&ids).unwrap();
        assert_eq!(pool.free_block_count(), 10);
        for id in &ids {
            assert_eq!(pool.get_block(*id).unwrap().ref_count, 0);
        }
    }

    #[test]
    fn block_pool_config_block_bytes() {
        let config = BlockPoolConfig {
            block_size: 16,
            num_blocks: 10,
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 64,
            dtype: DType::F32,
        };
        // 16 tokens * 2 layers * 2 (K+V) * 4 heads * 64 dims * 4 bytes
        let expected = 16usize * 2 * 2 * 4 * 64 * 4;
        assert_eq!(config.block_bytes(), expected);
    }

    #[test]
    fn block_pool_config_block_bytes_f16() {
        let config = BlockPoolConfig {
            block_size: 16,
            num_blocks: 10,
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 64,
            dtype: DType::F16,
        };
        let expected = 16usize * 2 * 2 * 4 * 64 * 2;
        assert_eq!(config.block_bytes(), expected);
    }

    #[test]
    fn invalid_block_id_errors() {
        let mut pool = BlockPool::new(default_config(3));
        let err = pool.free_block(100).expect_err("should fail");
        assert_eq!(err, BlockPoolError::InvalidBlockId { id: 100 });
    }

    #[test]
    fn inc_ref_on_unallocated_block_fails() {
        let mut pool = BlockPool::new(default_config(3));
        let err = pool.inc_ref(0).expect_err("should fail");
        assert_eq!(err, BlockPoolError::BlockNotAllocated { id: 0 });
    }
}
