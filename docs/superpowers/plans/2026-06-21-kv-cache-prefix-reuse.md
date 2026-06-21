# KV Cache Prefix Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse cached KV blocks for identical prompt prefixes across requests, skipping the prefill compute for the matched prefix and running only the unmatched suffix through the network.

**Architecture:** vLLM-style automatic prefix caching for a single active sequence at a time. Each full (`BLOCK_SIZE`-wide) KV block is content-addressed by a *chained* hash of its token ids plus its parent block's hash. A `PrefixRegistry` maps that hash to a physical block id; physical blocks gain reference counts so registry-owned and sequence-owned references coexist safely. On a new prompt we reset the active sequence, match the longest cached block-aligned prefix, reuse those physical blocks (no recompute), run only the suffix tokens through the transformer, then register any newly-completed full blocks. The existing `paged_attention` kernel already supports `seq_q < kv_len` via its `offset = kv_len - seq_q` causal mask, so **no attention-kernel changes are needed**.

**Tech Stack:** Rust, candle-core / candle-nn, `std::collections::hash_map::DefaultHasher` (no new dependencies).

## Global Constraints

- No new crate dependencies — hashing uses `std::collections::hash_map::DefaultHasher`.
- `BLOCK_SIZE = 16` (defined in `src/models/common/cache.rs:8`) — never hardcode `16` elsewhere; read the table's `block_size()`.
- Prefix reuse must be **numerically identical** to a full prefill (reused KV equals freshly-computed KV for the same tokens at the same absolute positions). No test may tolerate output drift from enabling reuse.
- Reuse operates on a **single active sequence**; concurrency / multi-sequence sharing and block eviction are explicitly out of scope (documented as future work).
- Run tests with: `cargo test` (crate name: `flour`). Filter a single test with `cargo test <test_name>`.

---

### Task 1: Reference-counted block allocator

**Files:**
- Modify: `src/models/common/paged.rs:3-28` (struct `BlockAllocator` and its impl)
- Modify: `src/models/common/paged.rs:198-209` (existing allocator test that uses the removed `free_block`)
- Test: `src/models/common/paged.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: nothing (leaf change).
- Produces:
  - `BlockAllocator::new(num_blocks: usize) -> BlockAllocator`
  - `BlockAllocator::allocate(&mut self) -> Option<usize>` (sets the returned block's refcount to 1)
  - `BlockAllocator::incref(&mut self, block_id: usize)`
  - `BlockAllocator::decref(&mut self, block_id: usize) -> bool` (returns `true` iff refcount reached 0 and the block was returned to the free list)
  - `BlockAllocator::ref_count(&self, block_id: usize) -> usize`
  - `BlockAllocator::num_free(&self) -> usize`

- [ ] **Step 1: Write the failing test**

Add to `mod tests` in `src/models/common/paged.rs`:

```rust
#[test]
fn allocator_refcounts_govern_when_a_block_is_freed() {
    let mut alloc = BlockAllocator::new(2);
    assert_eq!(alloc.num_free(), 2);

    let a = alloc.allocate().unwrap();
    assert_eq!(alloc.ref_count(a), 1);

    // A second reference keeps the block live across one decref.
    alloc.incref(a);
    assert_eq!(alloc.ref_count(a), 2);
    assert_eq!(alloc.decref(a), false); // still referenced
    assert_eq!(alloc.num_free(), 1);

    // Final decref frees it back to the pool.
    assert_eq!(alloc.decref(a), true);
    assert_eq!(alloc.ref_count(a), 0);
    assert_eq!(alloc.num_free(), 2);
    // Freed block can be hander out again.
    assert_eq!(alloc.allocate(), Some(a));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test allocator_refcounts_govern_when_a_block_is_freed`
Expected: FAIL to compile — `incref`, `decref`, `ref_count` do not exist.

- [ ] **Step 3: Write minimal implementation**

Replace the `BlockAllocator` struct and impl at `src/models/common/paged.rs:3-28` with:

```rust
/// Hands out and reclaims physical KV block ids from a free list, tracking a reference
/// count per block so registry-owned and sequence-owned references can coexist. A block is
/// returned to the free list only when its last reference is dropped.
#[derive(Debug)]
pub struct BlockAllocator {
    free: Vec<usize>,
    ref_counts: Vec<usize>,
}

impl BlockAllocator {
    pub fn new(num_blocks: usize) -> Self {
        // Reverse so the first `allocate` returns block 0 (nicer for debugging).
        Self {
            free: (0..num_blocks).rev().collect(),
            ref_counts: vec![0; num_blocks],
        }
    }

    pub fn allocate(&mut self) -> Option<usize> {
        let id = self.free.pop()?;
        self.ref_counts[id] = 1;
        Some(id)
    }

    pub fn incref(&mut self, block_id: usize) {
        self.ref_counts[block_id] += 1;
    }

    /// Drop one reference; returns `true` if the block hit zero references and was freed.
    pub fn decref(&mut self, block_id: usize) -> bool {
        debug_assert!(self.ref_counts[block_id] > 0, "decref of unreferenced block");
        self.ref_counts[block_id] -= 1;
        if self.ref_counts[block_id] == 0 {
            self.free.push(block_id);
            true
        } else {
            false
        }
    }

    pub fn ref_count(&self, block_id: usize) -> usize {
        self.ref_counts[block_id]
    }

    pub fn num_free(&self) -> usize {
        self.free.len()
    }
}
```

- [ ] **Step 4: Update the existing allocator test that referenced `free_block`**

Replace the test at `src/models/common/paged.rs:198-209` (`allocator_hands_out_distinct_blocks_then_exhausts`) with:

```rust
#[test]
fn allocator_hands_out_distinct_blocks_then_exhausts() {
    let mut alloc = BlockAllocator::new(2);
    assert_eq!(alloc.num_free(), 2);
    let a = alloc.allocate().unwrap();
    let b = alloc.allocate().unwrap();
    assert_ne!(a, b);
    assert_eq!(alloc.allocate(), None);
    assert_eq!(alloc.decref(a), true);
    assert_eq!(alloc.num_free(), 1);
    assert_eq!(alloc.allocate(), Some(a));
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --lib models::common::paged`
Expected: PASS (all `paged` tests, including the two above).

- [ ] **Step 6: Commit**

```bash
git add src/models/common/paged.rs
git commit -m "feat: add reference counting to BlockAllocator"
```

---

### Task 2: Chained block hash and prefix registry

**Files:**
- Create: `src/models/common/prefix.rs`
- Modify: `src/models/common/mod.rs:1-14` (declare the module)
- Test: `src/models/common/prefix.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `pub const PREFIX_HASH_SEED: u64` — parent hash used for the very first block.
  - `pub fn block_hash(parent: u64, tokens: &[u32]) -> u64`
  - `pub struct PrefixRegistry` with:
    - `PrefixRegistry::new() -> PrefixRegistry`
    - `PrefixRegistry::get(&self, hash: u64, tokens: &[u32]) -> Option<usize>` (collision-safe: returns the physical block id only if stored tokens equal `tokens`)
    - `PrefixRegistry::contains(&self, hash: u64) -> bool`
    - `PrefixRegistry::insert(&mut self, hash: u64, tokens: Vec<u32>, block_id: usize)`
    - `PrefixRegistry::clear(&mut self)`

- [ ] **Step 1: Write the failing test**

Create `src/models/common/prefix.rs` with only the test module first:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_hash_is_deterministic_and_parent_sensitive() {
        let toks = [1u32, 2, 3];
        assert_eq!(block_hash(PREFIX_HASH_SEED, &toks), block_hash(PREFIX_HASH_SEED, &toks));
        // Different parent => different child hash (chaining).
        assert_ne!(block_hash(PREFIX_HASH_SEED, &toks), block_hash(42, &toks));
        // Different tokens => different hash.
        assert_ne!(block_hash(PREFIX_HASH_SEED, &toks), block_hash(PREFIX_HASH_SEED, &[1, 2, 4]));
    }

    #[test]
    fn registry_get_verifies_tokens_against_collisions() {
        let mut reg = PrefixRegistry::new();
        let h = block_hash(PREFIX_HASH_SEED, &[1, 2]);
        reg.insert(h, vec![1, 2], 7);

        // Correct tokens at the stored hash -> hit.
        assert_eq!(reg.get(h, &[1, 2]), Some(7));
        assert!(reg.contains(h));
        // Same hash but mismatched tokens (simulated collision) -> miss, not a wrong reuse.
        assert_eq!(reg.get(h, &[9, 9]), None);
        // Unknown hash -> miss.
        assert_eq!(reg.get(block_hash(PREFIX_HASH_SEED, &[3, 4]), &[3, 4]), None);

        reg.clear();
        assert!(!reg.contains(h));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib models::common::prefix`
Expected: FAIL to compile — module not declared / symbols missing.

- [ ] **Step 3: Declare the module**

In `src/models/common/mod.rs`, add the module declaration alphabetically after `pub mod paged;` (it is fine to keep ordering loose — match the existing list). Add this line in the `pub mod` block (around `src/models/common/mod.rs:6`):

```rust
pub mod prefix;
```

- [ ] **Step 4: Write minimal implementation**

Prepend to `src/models/common/prefix.rs` (above the test module):

```rust
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Parent hash for the first block in a sequence; chains every subsequent block hash.
pub const PREFIX_HASH_SEED: u64 = 0;

/// Chained content hash: folds the parent block's hash together with this block's token ids,
/// so the result identifies a full prefix path (positions `0..end`), not just one block's
/// contents. Two sequences share a block hash iff every token from position 0 matches.
pub fn block_hash(parent: u64, tokens: &[u32]) -> u64 {
    let mut h = DefaultHasher::new();
    parent.hash(&mut h);
    tokens.hash(&mut h);
    h.finish()
}

/// Maps a full block's chained hash to the physical block holding its KV. The block's token
/// ids are retained so lookups can verify a match and reject hash collisions.
#[derive(Debug, Default)]
pub struct PrefixRegistry {
    entries: HashMap<u64, (Vec<u32>, usize)>,
}

impl PrefixRegistry {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Physical block id stored under `hash`, but only when the stored tokens equal `tokens`
    /// (guards against hash collisions returning the wrong KV).
    pub fn get(&self, hash: u64, tokens: &[u32]) -> Option<usize> {
        self.entries
            .get(&hash)
            .and_then(|(stored, id)| (stored.as_slice() == tokens).then_some(*id))
    }

    pub fn contains(&self, hash: u64) -> bool {
        self.entries.contains_key(&hash)
    }

    pub fn insert(&mut self, hash: u64, tokens: Vec<u32>, block_id: usize) {
        self.entries.insert(hash, (tokens, block_id));
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --lib models::common::prefix`
Expected: PASS (both tests).

- [ ] **Step 6: Commit**

```bash
git add src/models/common/prefix.rs src/models/common/mod.rs
git commit -m "feat: add chained block hash and prefix registry"
```

---

### Task 3: BlockTable accessors for reuse orchestration

**Files:**
- Modify: `src/models/common/paged.rs:38-81` (`BlockTable` impl)
- Test: `src/models/common/paged.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: `BlockTable` (existing).
- Produces:
  - `BlockTable::blocks(&self) -> &[usize]` — physical block ids in logical order.
  - `BlockTable::block_at(&self, logical_block: usize) -> usize` — physical id of logical block index.
  - `BlockTable::clear(&mut self)` — empty the table (no blocks, `len = 0`); keeps `block_size`.

- [ ] **Step 1: Write the failing test**

Add to `mod tests` in `src/models/common/paged.rs`:

```rust
#[test]
fn block_table_exposes_blocks_and_clears() {
    let mut table = BlockTable::new(2);
    table.push_block(5);
    table.push_block(3);
    table.advance(4);
    assert_eq!(table.blocks(), &[5, 3]);
    assert_eq!(table.block_at(0), 5);
    assert_eq!(table.block_at(1), 3);

    table.clear();
    assert!(table.is_empty());
    assert_eq!(table.len(), 0);
    assert_eq!(table.blocks(), &[] as &[usize]);
    assert_eq!(table.block_size(), 2); // block_size preserved
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test block_table_exposes_blocks_and_clears`
Expected: FAIL to compile — `blocks`, `block_at`, `clear` do not exist.

- [ ] **Step 3: Write minimal implementation**

Add these methods inside `impl BlockTable` in `src/models/common/paged.rs` (e.g. right after `push_block`, around `src/models/common/paged.rs:66`):

```rust
    /// Physical block ids currently mapped, in logical order.
    pub fn blocks(&self) -> &[usize] {
        &self.blocks
    }

    /// Physical block id for a logical block index.
    pub fn block_at(&self, logical_block: usize) -> usize {
        self.blocks[logical_block]
    }

    /// Drop all blocks and reset the live length, keeping the configured block size.
    pub fn clear(&mut self) {
        self.blocks.clear();
        self.len = 0;
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib models::common::paged`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/common/paged.rs
git commit -m "feat: add BlockTable accessors for prefix reuse"
```

---

### Task 4: KvCache + Cache prefix orchestration

**Files:**
- Modify: `src/models/common/cache.rs:1-15` (imports, `KvCache` struct)
- Modify: `src/models/common/cache.rs:17-92` (`KvCache` impl — add registry field init + methods)
- Modify: `src/models/common/cache.rs:104-183` (`Cache` impl — delegating methods)
- Test: `src/models/common/cache.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: `BlockAllocator::{incref, decref}` (Task 1), `PrefixRegistry`, `block_hash`, `PREFIX_HASH_SEED` (Task 2), `BlockTable::{blocks, block_at, clear}` (Task 3).
- Produces (on `KvCache`, and delegated identically on `Cache`):
  - `reset_sequence(&mut self)` — decref every block held by the current sequence and empty its block table; the prefix registry is preserved.
  - `match_prefix(&mut self, token_ids: &[u32]) -> usize` — reuse the longest cached block-aligned prefix into the (freshly reset) sequence; returns matched token count (a multiple of `block_size`). Always leaves ≥1 token unmatched so a suffix exists.
  - `register_prefix(&mut self, token_ids: &[u32])` — register every full block of the completed prefill so future sequences can reuse it.
  - `clear_prefix_cache(&mut self)` — drop all registry entries (escape hatch; does not affect the live sequence).

- [ ] **Step 1: Write the failing tests**

Add to `mod tests` in `src/models/common/cache.rs` (the module already imports `super::*` and defines `test_config()` with `max_seq_len: 16`, `BLOCK_SIZE = 16` → exactly 1 block). To exercise multi-block reuse, these tests build their own larger config:

```rust
fn prefix_config() -> Config {
    // max_seq_len 64 with BLOCK_SIZE 16 => 4 physical blocks.
    Config {
        max_seq_len: 64,
        ..test_config()
    }
}

/// Fill the live suffix [matched..matched+seq_len) of layer 0 with deterministic KV so the
/// pool has real, distinguishable contents to reuse.
fn write_zeros_for_current_batch(cache: &mut Cache, seq_len: usize) {
    let k = candle_core::Tensor::zeros(
        (1, prefix_config().num_key_value_heads, seq_len, prefix_config().head_dim),
        candle_core::DType::F32,
        &Device::Cpu,
    )
    .unwrap();
    cache.write_kv(0, &k, &k).unwrap();
}

#[test]
fn match_prefix_is_empty_before_anything_is_registered() {
    let mut cache = Cache::new(&prefix_config(), &Device::Cpu).unwrap();
    let ids: Vec<u32> = (0..40).collect(); // 2 full blocks + partial
    cache.reset_sequence();
    assert_eq!(cache.match_prefix(&ids), 0);
}

#[test]
fn register_then_match_reuses_full_blocks_and_leaves_a_suffix() {
    let cfg = prefix_config();
    let mut cache = Cache::new(&cfg, &Device::Cpu).unwrap();
    let ids: Vec<u32> = (0..40).collect(); // 40 tokens => 2 full blocks (32) + 8 partial

    // First request: nothing cached, allocate + fill all 40, then register full blocks.
    cache.reset_sequence();
    assert_eq!(cache.match_prefix(&ids), 0);
    cache.allocate_kv(40).unwrap();
    write_zeros_for_current_batch(&mut cache, 40);
    cache.register_prefix(&ids);

    // Second request, identical prompt: the 2 full blocks (32 tokens) are reused; the partial
    // tail (8 tokens) stays as suffix.
    cache.reset_sequence();
    let matched = cache.match_prefix(&ids);
    assert_eq!(matched, 32);
}

#[test]
fn match_prefix_leaves_last_block_when_prompt_is_block_aligned() {
    let cfg = prefix_config();
    let mut cache = Cache::new(&cfg, &Device::Cpu).unwrap();
    let ids: Vec<u32> = (0..32).collect(); // exactly 2 full blocks, no partial tail

    cache.reset_sequence();
    cache.match_prefix(&ids);
    cache.allocate_kv(32).unwrap();
    write_zeros_for_current_batch(&mut cache, 32);
    cache.register_prefix(&ids);

    cache.reset_sequence();
    // Block-aligned prompt: at most num_full - 1 blocks reused so a suffix always remains.
    assert_eq!(cache.match_prefix(&ids), 16);
}

#[test]
fn reset_sequence_frees_unregistered_blocks_but_keeps_registered_ones() {
    let cfg = prefix_config();
    let mut cache = Cache::new(&cfg, &Device::Cpu).unwrap();
    let ids: Vec<u32> = (0..40).collect();

    cache.reset_sequence();
    cache.match_prefix(&ids);
    cache.allocate_kv(40).unwrap(); // 3 blocks: 2 full + 1 partial
    write_zeros_for_current_batch(&mut cache, 40);
    cache.register_prefix(&ids);

    let free_after_first = cache.free_blocks_for_test();
    cache.reset_sequence();
    // The partial 3rd block is freed; the 2 registered blocks stay live.
    assert_eq!(cache.free_blocks_for_test(), free_after_first + 1);
}

#[test]
fn clear_prefix_cache_disables_reuse() {
    let cfg = prefix_config();
    let mut cache = Cache::new(&cfg, &Device::Cpu).unwrap();
    let ids: Vec<u32> = (0..40).collect();

    cache.reset_sequence();
    cache.match_prefix(&ids);
    cache.allocate_kv(40).unwrap();
    write_zeros_for_current_batch(&mut cache, 40);
    cache.register_prefix(&ids);

    cache.clear_prefix_cache();
    cache.reset_sequence();
    assert_eq!(cache.match_prefix(&ids), 0);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib models::common::cache`
Expected: FAIL to compile — `reset_sequence`, `match_prefix`, `register_prefix`, `clear_prefix_cache`, `free_blocks_for_test` do not exist.

- [ ] **Step 3: Wire the registry into KvCache**

In `src/models/common/cache.rs`, update the imports at the top (around `src/models/common/cache.rs:5`) to also pull in the prefix module:

```rust
use super::paged::{BlockAllocator, BlockTable, PagedKvPool};
use super::prefix::{block_hash, PrefixRegistry, PREFIX_HASH_SEED};
use super::Config;
```

Add a `registry` field to `KvCache` (struct at `src/models/common/cache.rs:10-15`):

```rust
#[derive(Debug)]
pub struct KvCache {
    pool: PagedKvPool,
    allocator: BlockAllocator,
    table: BlockTable,
    registry: PrefixRegistry,
}
```

Initialize it in `KvCache::new` (the `Ok(Self { ... })` at `src/models/common/cache.rs:31-35`):

```rust
        Ok(Self {
            pool,
            allocator,
            table,
            registry: PrefixRegistry::new(),
        })
```

- [ ] **Step 4: Add the orchestration methods to KvCache**

Add inside `impl KvCache` (e.g. after `gather_blocks`, around `src/models/common/cache.rs:91`):

```rust
    /// Drop the current sequence's references and start an empty sequence. Registry-owned
    /// blocks (refcount > 1) survive; sequence-only blocks return to the free list.
    pub fn reset_sequence(&mut self) {
        let blocks: Vec<usize> = self.table.blocks().to_vec();
        for block in blocks {
            self.allocator.decref(block);
        }
        self.table.clear();
    }

    /// Reuse cached full blocks that form a prefix of `token_ids`, pushing them into the
    /// current (just-reset) sequence. Returns the number of tokens served from cache (a
    /// multiple of `block_size`). Always leaves at least one block unmatched so the caller has
    /// a suffix to run — guaranteeing last-position logits.
    pub fn match_prefix(&mut self, token_ids: &[u32]) -> usize {
        let bs = self.table.block_size();
        let num_full = token_ids.len() / bs;
        // If the prompt is exactly block-aligned, never reuse its final full block, or the
        // suffix would be empty.
        let max_reusable = if token_ids.len() % bs == 0 {
            num_full.saturating_sub(1)
        } else {
            num_full
        };

        let mut parent = PREFIX_HASH_SEED;
        let mut matched = 0usize;
        for b in 0..max_reusable {
            let chunk = &token_ids[b * bs..(b + 1) * bs];
            let h = block_hash(parent, chunk);
            match self.registry.get(h, chunk) {
                Some(block_id) => {
                    self.allocator.incref(block_id);
                    self.table.push_block(block_id);
                    matched += bs;
                    parent = h;
                }
                None => break,
            }
        }
        self.table.advance(matched);
        matched
    }

    /// Register every full block of the just-completed prefill so later sequences can reuse it.
    /// `token_ids` is the entire prompt (reused prefix + freshly computed suffix). Blocks
    /// already in the registry are skipped (their reference is already held).
    pub fn register_prefix(&mut self, token_ids: &[u32]) {
        let bs = self.table.block_size();
        let num_full = token_ids.len() / bs;
        let mut parent = PREFIX_HASH_SEED;
        for b in 0..num_full {
            let chunk = &token_ids[b * bs..(b + 1) * bs];
            let h = block_hash(parent, chunk);
            if !self.registry.contains(h) {
                let block_id = self.table.block_at(b);
                self.allocator.incref(block_id); // registry takes its own reference
                self.registry.insert(h, chunk.to_vec(), block_id);
            }
            parent = h;
        }
    }

    /// Drop all cached prefixes. The live sequence's blocks keep their sequence references, so
    /// the in-flight request is unaffected; only future reuse is disabled until re-registered.
    pub fn clear_prefix_cache(&mut self) {
        self.registry.clear();
    }

    #[cfg(test)]
    pub fn free_blocks(&self) -> usize {
        self.allocator.num_free()
    }
```

- [ ] **Step 5: Add delegating methods to `Cache`**

Add inside `impl Cache` (after `kv_blocks`, around `src/models/common/cache.rs:182`):

```rust
    /// Begin a fresh sequence, releasing the previous sequence's non-cached blocks. The prefix
    /// registry persists across sequences (this is what enables cross-request reuse).
    pub fn reset_sequence(&mut self) {
        self.kvs.reset_sequence();
    }

    /// Reuse the longest cached block-aligned prefix of `token_ids`; returns matched token
    /// count. Call on a freshly reset sequence, before `allocate_kv`/`write_kv` for the suffix.
    pub fn match_prefix(&mut self, token_ids: &[u32]) -> usize {
        self.kvs.match_prefix(token_ids)
    }

    /// Register the completed prefill's full blocks for future reuse.
    pub fn register_prefix(&mut self, token_ids: &[u32]) {
        self.kvs.register_prefix(token_ids);
    }

    /// Drop all cached prefixes (escape hatch; does not disturb the live sequence).
    pub fn clear_prefix_cache(&mut self) {
        self.kvs.clear_prefix_cache();
    }

    #[cfg(test)]
    pub fn free_blocks_for_test(&self) -> usize {
        self.kvs.free_blocks()
    }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo test --lib models::common::cache`
Expected: PASS (the five new tests plus all pre-existing cache tests).

- [ ] **Step 7: Commit**

```bash
git add src/models/common/cache.rs
git commit -m "feat: add prefix match/register orchestration to KvCache"
```

---

### Task 5: CausalLM::prefill_cached

**Files:**
- Modify: `src/models/common/model.rs:43-59` (add method beside `forward`)
- Test: `src/models/common/model.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: `Cache::{reset_sequence, match_prefix, register_prefix}` (Task 4); existing `CausalLM::forward(&self, input_ids: &Tensor, index_pos: usize, cache: &mut Cache) -> Result<Tensor>`.
- Produces:
  - `CausalLM::prefill_cached(&self, input_ids: &Tensor, cache: &mut Cache) -> Result<(Tensor, usize)>` — returns `(logits, reused_prefix_len)`. `logits` covers the suffix positions (shape `(1, suffix_len, vocab_size)`); its final row is the last prompt token, ready to sample. `reused_prefix_len` is the number of prompt tokens served from cache.

- [ ] **Step 1: Write the failing tests**

The `model.rs` test module already has `test_config()` (with `max_seq_len: 16`) and `make_vb`. Add a larger config plus tests. Add to `mod tests` in `src/models/common/model.rs`:

```rust
fn prefix_test_config() -> Config {
    Config {
        max_seq_len: 64,
        ..test_config()
    }
}

#[test]
fn prefill_cached_with_cold_cache_matches_plain_forward() {
    let cfg = prefix_test_config();
    let model = CausalLM::load(make_vb(&cfg), cfg.clone()).unwrap();

    // Plain forward over the whole prompt (reference).
    let ids_vec: Vec<u32> = (0..20u32).map(|i| i % cfg.vocab_size as u32).collect();
    let ids = Tensor::from_vec(ids_vec.clone(), (1, ids_vec.len()), &Device::Cpu).unwrap();
    let mut ref_cache = super::super::Cache::new(&cfg, &Device::Cpu).unwrap();
    let reference = model.forward(&ids, 0, &mut ref_cache).unwrap();
    let ref_last: Vec<f32> = reference
        .i((0, ids_vec.len() - 1)).unwrap()
        .to_vec1().unwrap();

    // Cold prefill_cached: nothing cached yet => no reuse, identical last-row logits.
    let mut cache = super::super::Cache::new(&cfg, &Device::Cpu).unwrap();
    let (logits, reused) = model.prefill_cached(&ids, &mut cache).unwrap();
    assert_eq!(reused, 0);
    let got_last: Vec<f32> = logits
        .i((0, logits.dim(1).unwrap() - 1)).unwrap()
        .to_vec1().unwrap();
    for (a, b) in ref_last.iter().zip(got_last.iter()) {
        assert!((a - b).abs() < 1e-4, "logit mismatch {a} vs {b}");
    }
}

#[test]
fn prefill_cached_reuses_prefix_and_preserves_last_logits() {
    let cfg = prefix_test_config();
    let model = CausalLM::load(make_vb(&cfg), cfg.clone()).unwrap();

    let ids_vec: Vec<u32> = (0..40u32).map(|i| i % cfg.vocab_size as u32).collect();
    let ids = Tensor::from_vec(ids_vec.clone(), (1, ids_vec.len()), &Device::Cpu).unwrap();

    // Reference: last-row logits from a cold cache.
    let mut ref_cache = super::super::Cache::new(&cfg, &Device::Cpu).unwrap();
    let (ref_logits, _) = model.prefill_cached(&ids, &mut ref_cache).unwrap();
    let ref_last: Vec<f32> = ref_logits
        .i((0, ref_logits.dim(1).unwrap() - 1)).unwrap()
        .to_vec1().unwrap();

    // Same cache, run the identical prompt again: prefix now reused, suffix shorter.
    let (logits, reused) = model.prefill_cached(&ids, &mut ref_cache).unwrap();
    assert!(reused > 0, "expected prefix reuse on the second identical prompt");
    assert_eq!(logits.dim(1).unwrap(), ids_vec.len() - reused);

    let got_last: Vec<f32> = logits
        .i((0, logits.dim(1).unwrap() - 1)).unwrap()
        .to_vec1().unwrap();
    for (a, b) in ref_last.iter().zip(got_last.iter()) {
        assert!((a - b).abs() < 1e-4, "reuse changed logits: {a} vs {b}");
    }
}
```

Note: this test uses `candle_core::IndexOp` (`.i(..)`). Ensure the test module's `use` brings it in — add `use candle_core::IndexOp;` to the imports inside `mod tests` if not already present (the file currently imports `candle_core::{DType, Device, Tensor}` at `src/models/common/model.rs:70`; extend it).

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib models::common::model`
Expected: FAIL to compile — `prefill_cached` does not exist (and possibly `IndexOp` unimported).

- [ ] **Step 3: Write minimal implementation**

Add the `IndexOp` import is only needed in tests. In the non-test code, add the method inside `impl CausalLM`, right after `forward` (after `src/models/common/model.rs:59`):

```rust
    /// Prefill that reuses any cached prefix. Matched leading blocks are read straight from the
    /// KV pool (no recompute); only the unmatched suffix runs through the network. Returns the
    /// suffix logits (final row = last prompt token) and how many prompt tokens were reused.
    pub fn prefill_cached(
        &self,
        input_ids: &Tensor,
        cache: &mut Cache,
    ) -> Result<(Tensor, usize)> {
        let ids: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;

        cache.reset_sequence();
        let matched = cache.match_prefix(&ids);

        let suffix = &ids[matched..];
        let suffix_ids = Tensor::from_vec(suffix.to_vec(), (1, suffix.len()), input_ids.device())?;

        // `forward` allocates KV slots for the suffix, writes its K/V into fresh blocks, and
        // runs paged attention over ALL live blocks (reused prefix + suffix). RoPE uses
        // `index_pos = matched`, so the suffix sits at its true absolute positions.
        let logits = self.forward(&suffix_ids, matched, cache)?;

        cache.register_prefix(&ids);
        Ok((logits, matched))
    }
```

- [ ] **Step 4: Add the test import**

In `src/models/common/model.rs`, change the test-module import line (`src/models/common/model.rs:70`) from:

```rust
    use candle_core::{DType, Device, Tensor};
```

to:

```rust
    use candle_core::{DType, Device, IndexOp, Tensor};
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --lib models::common::model`
Expected: PASS (the two new tests plus existing model tests).

- [ ] **Step 6: Commit**

```bash
git add src/models/common/model.rs
git commit -m "feat: add prefix-cached prefill to CausalLM"
```

---

### Task 6: Engine integration (cross-request reuse)

**Files:**
- Modify: `src/engine.rs:14-26` (`GenerationStats`, `Engine` struct)
- Modify: `src/engine.rs:28-66` (`Engine::load` — create the persistent cache)
- Modify: `src/engine.rs:72-120` (`Engine::generate` — use `prefill_cached` against the persistent cache)
- Test: `src/engine.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: `CausalLM::prefill_cached` (Task 5); `Cache::new` (existing).
- Produces:
  - `GenerationStats { prompt_tokens: usize, completion_tokens: usize, reused_prefix_tokens: usize }`
  - `Engine` holds `cache: std::sync::Mutex<Cache>` so the prefix registry persists across `generate` calls. `generate` keeps its `&self` signature (the mutex provides interior mutability; the single-sequence scope means calls serialize on the lock).

- [ ] **Step 1: Write the failing test**

Add to `pub(crate) mod tests` in `src/engine.rs`:

```rust
#[test]
fn second_identical_prompt_reuses_prefix_and_is_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    write_fixture_model(dir.path());
    let engine = Engine::load(dir.path()).unwrap();
    let messages = vec![ChatMessage {
        role: "user".into(),
        content: "hi".into(),
    }];
    let params = crate::sampling::SamplingParams {
        max_tokens: 4,
        seed: 7,
        temperature: 0.0,
        ..Default::default()
    };

    let mut first = String::new();
    let stats1 = engine
        .generate(&messages, &params, |tok| first.push_str(tok))
        .unwrap();
    // Cold cache: the first call computes the whole prompt.
    assert_eq!(stats1.reused_prefix_tokens, 0);

    let mut second = String::new();
    let stats2 = engine
        .generate(&messages, &params, |tok| second.push_str(tok))
        .unwrap();
    // Second identical prompt reuses at least one cached block, and output is unchanged.
    assert!(stats2.reused_prefix_tokens > 0);
    assert_eq!(first, second);
    assert_eq!(stats2.prompt_tokens, stats1.prompt_tokens);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test second_identical_prompt_reuses_prefix_and_is_deterministic`
Expected: FAIL to compile — `reused_prefix_tokens` field missing, and `Engine` has no persistent cache.

- [ ] **Step 3: Add the persistent cache and stats field**

In `src/engine.rs`, extend `GenerationStats` (at `src/engine.rs:14-17`):

```rust
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub reused_prefix_tokens: usize,
}
```

Add a `cache` field to `Engine` (struct at `src/engine.rs:19-26`):

```rust
pub struct Engine {
    model: CausalLM,
    tokenizer: Tokenizer,
    chat_template: ChatTemplate,
    eos_token_id: EosTokenId,
    device: Device,
    model_id: String,
    cache: std::sync::Mutex<Cache>,
}
```

- [ ] **Step 4: Build the cache in `load`**

In `Engine::load`, build the cache after the model is loaded and include it in the returned struct. Change the construction (currently `src/engine.rs:40` builds `model`, and `src/engine.rs:58-65` returns `Self`). After `let model = CausalLM::load(vb, cfg.clone())?;` add:

```rust
        let cache = std::sync::Mutex::new(Cache::new(&cfg, &device)?);
```

and add `cache,` to the `Ok(Self { ... })` field list:

```rust
        Ok(Self {
            model,
            tokenizer,
            chat_template,
            eos_token_id,
            device,
            model_id,
            cache,
        })
```

- [ ] **Step 5: Use prefix-cached prefill in `generate`**

Replace the body of `generate` from the cache creation through the prefill call. Specifically, replace `src/engine.rs:82-90`:

```rust
        let mut cache = Cache::new(self.model.config(), &self.device)?;
        let mut sampler = LogitsSampler::new(params.seed);
        let mut all_tokens = prompt_tokens.clone();

        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut logits = self.model.forward(&input, 0, &mut cache)?;
        let mut completion_tokens = 0usize;

        tracing::info!("Finished prefill, input token count: {} ", all_tokens.len());
```

with:

```rust
        let mut cache = self.cache.lock().unwrap();
        let mut sampler = LogitsSampler::new(params.seed);
        let mut all_tokens = prompt_tokens.clone();

        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let (mut logits, reused_prefix_tokens) =
            self.model.prefill_cached(&input, &mut cache)?;
        let mut completion_tokens = 0usize;

        tracing::info!(
            "Finished prefill, prompt tokens: {}, reused prefix tokens: {}",
            all_tokens.len(),
            reused_prefix_tokens
        );
```

The decode loop at `src/engine.rs:92-114` is unchanged except it now references the locked `cache`; since `cache` is a `MutexGuard<Cache>`, `&mut cache` still yields `&mut Cache` (deref coercion), so `self.model.forward(&next_input, index_pos, &mut cache)?` compiles as-is.

Finally, update the returned stats (`src/engine.rs:116-119`):

```rust
        Ok(GenerationStats {
            prompt_tokens: prompt_len,
            completion_tokens,
            reused_prefix_tokens,
        })
```

- [ ] **Step 6: Run the new test and the existing engine tests**

Run: `cargo test --lib engine`
Expected: PASS — the new reuse test, plus `load_and_generate_end_to_end_with_tiny_random_model` and `generate_is_deterministic_for_a_fixed_seed` (the latter now also exercises reuse on its second call and must still produce identical output).

- [ ] **Step 7: Run the full suite**

Run: `cargo test`
Expected: PASS (whole workspace).

- [ ] **Step 8: Commit**

```bash
git add src/engine.rs
git commit -m "feat: persist KV cache across requests for prefix reuse"
```

---

## Out of Scope (documented future work)

- **Block eviction.** Registered blocks are held by the registry forever, so a long-lived `Engine` that serves many *distinct* prompts can exhaust the pool. `Cache::clear_prefix_cache()` is the manual escape hatch. An LRU/refcount-aware eviction policy is future work.
- **Multi-sequence / concurrent sharing.** The cache models a single active sequence; the engine serializes `generate` on the cache mutex. Concurrent requests sharing prefixes would require per-sequence block tables.
- **Registering decode-generated tokens.** Only the prompt's full blocks are registered (during `prefill_cached`). Tokens produced during decode are not added to the registry.

## Self-Review

**1. Spec coverage:**
- vLLM-style content-hash auto matching → Task 2 (`block_hash` chained hash, `PrefixRegistry`) + Task 4 (`match_prefix`). ✔
- Skip prefix compute → Task 5 (`prefill_cached` runs only the suffix through `forward`). ✔
- Single-sequence cross-request reuse → Task 4 (`reset_sequence` preserves registry) + Task 6 (persistent `Cache` on `Engine`). ✔
- Correctness/no-drift constraint → reuse tests assert last-row logit equality (Task 5) and identical generated text (Task 6). ✔
- Refcounted lifetime (don't free shared blocks) → Task 1 + `reset_sequence`/`register_prefix` refcount discipline (Task 4). ✔

**2. Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to Task N" — every code step contains full code. ✔

**3. Type consistency:**
- `match_prefix`/`register_prefix`/`reset_sequence`/`clear_prefix_cache` names identical across `KvCache` and `Cache` delegates and call sites. ✔
- `block_hash(parent, tokens)`, `PrefixRegistry::{get,contains,insert,clear}` signatures match between Task 2 definition and Task 4 usage. ✔
- `prefill_cached -> Result<(Tensor, usize)>` matches Task 6 destructuring `let (mut logits, reused_prefix_tokens) = ...`. ✔
- `BlockTable::{blocks, block_at, clear}` defined in Task 3, used in Task 4. ✔
- `BlockAllocator::{incref, decref, ref_count, num_free}` defined in Task 1, used in Task 4. ✔
