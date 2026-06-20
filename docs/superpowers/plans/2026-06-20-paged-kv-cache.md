# Paged KV Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the contiguous per-layer KV cache with a block-paged KV memory subsystem (allocator + block table + physical pool), wired into the existing single-sequence inference path with unchanged end-to-end behavior.

**Architecture:** Introduce a fixed-size physical KV pool sliced into blocks. A global `BlockAllocator` hands out physical block ids; a per-sequence `BlockTable` maps logical token positions to physical slots; a `PagedKvCache` owns the per-layer pool tensors and does slot-addressed writes plus gather-reads. The existing `Cache::append_kv` keeps its signature — it now writes new KV into paged slots and gathers the full sequence back out, so `CausalSelfAttention::forward` and the dense `causal_attention` kernel are untouched. This delivers the memory-management half of PagedAttention; a fused paged-attention kernel, the continuous-batching scheduler, and prefix-sharing/CoW are separate follow-up plans.

**Tech Stack:** Rust, `candle_core` (Tensor ops: `slice_assign`, `index_select`, `narrow`, `transpose`), `cargo test`.

## Global Constraints

- Block size is fixed at `BLOCK_SIZE = 16` tokens.
- The pool is sized to exactly cover `cfg.max_seq_len`: `num_blocks = max_seq_len.div_ceil(BLOCK_SIZE)`. Appending past `max_seq_len` returns an error (`candle_core::bail!`) — this plan does **not** reproduce the old sliding-window trim; that is a named follow-up (see "Out of Scope").
- All tensors are `DType::F32` on the `Device` passed to `Cache::new` (currently `Device::Cpu`).
- `append_kv` must remain a drop-in replacement: same signature `(&mut self, layer_idx: usize, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)>`, where `k`/`v` arrive shaped `(1, kv_heads, seq_len, head_dim)` and the returned pair is the full `(1, kv_heads, total_len, head_dim)` KV.
- Paged write/advance bookkeeping happens only on `layer_idx == 0`; layers iterate ascending from 0 (verified in `src/models/common/model.rs:50`).
- Keep clippy clean and `cargo fmt`-formatted (repo convention, see commit `8d32381`).

**Out of Scope (follow-up plans):** fused paged-attention kernel (avoid the gather), continuous-batching scheduler / multi-sequence shared pool, prefix sharing + copy-on-write, sliding-window eviction parity.

---

## File Structure

- **Create** `src/models/common/paged.rs` — `BlockAllocator`, `BlockTable`, `PagedKvCache`. One file: these three types are the paged-memory subsystem and change together.
- **Modify** `src/models/common/mod.rs` — declare the new module.
- **Modify** `src/models/common/cache.rs` — swap `kvs: Vec<Option<(Tensor, Tensor)>>` for the paged structures; rewrite `append_kv`. `cos`/`sin`/`masks`/`rope_*`/`causal_mask` are unchanged.

No changes to `attention.rs`, `backend/cpu.rs`, `engine.rs`, or model files — that is the point of preserving `append_kv`'s contract.

---

### Task 1: BlockAllocator

**Files:**
- Create: `src/models/common/paged.rs`
- Modify: `src/models/common/mod.rs:1-6` (add module declaration)
- Test: in `src/models/common/paged.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: nothing.
- Produces: `BlockAllocator::new(num_blocks: usize) -> Self`, `allocate(&mut self) -> Option<usize>`, `free_block(&mut self, block_id: usize)`, `num_free(&self) -> usize`.

- [ ] **Step 1: Declare the module**

In `src/models/common/mod.rs`, add a line under the existing `pub mod` block:

```rust
pub mod attention;
pub mod cache;
pub mod config;
pub mod mlp;
pub mod model;
pub mod paged;
pub mod transformer;
```

- [ ] **Step 2: Write the failing test**

Create `src/models/common/paged.rs` with only the test module and an empty type so it compiles to a failing assertion:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocator_hands_out_distinct_blocks_then_exhausts() {
        let mut alloc = BlockAllocator::new(2);
        assert_eq!(alloc.num_free(), 2);
        let a = alloc.allocate().unwrap();
        let b = alloc.allocate().unwrap();
        assert_ne!(a, b);
        assert_eq!(alloc.allocate(), None);
        alloc.free_block(a);
        assert_eq!(alloc.num_free(), 1);
        assert_eq!(alloc.allocate(), Some(a));
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test --lib paged::tests::allocator -- --nocapture`
Expected: FAIL — `cannot find type BlockAllocator in this scope`.

- [ ] **Step 4: Write minimal implementation**

Add to the top of `src/models/common/paged.rs` (above the test module):

```rust
/// Hands out and reclaims physical KV block ids from a free list.
#[derive(Debug)]
pub struct BlockAllocator {
    free: Vec<usize>,
}

impl BlockAllocator {
    pub fn new(num_blocks: usize) -> Self {
        // Reverse so the first `allocate` returns block 0 (nicer for debugging).
        Self {
            free: (0..num_blocks).rev().collect(),
        }
    }

    pub fn allocate(&mut self) -> Option<usize> {
        self.free.pop()
    }

    pub fn free_block(&mut self, block_id: usize) {
        self.free.push(block_id);
    }

    pub fn num_free(&self) -> usize {
        self.free.len()
    }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test --lib paged::tests::allocator`
Expected: PASS (1 passed).

- [ ] **Step 6: Commit**

```bash
git add src/models/common/paged.rs src/models/common/mod.rs
git commit -m "feat: add BlockAllocator for paged KV cache"
```

---

### Task 2: BlockTable

**Files:**
- Modify: `src/models/common/paged.rs`
- Test: `src/models/common/paged.rs` (same test module)

**Interfaces:**
- Consumes: nothing.
- Produces: `BlockTable::new(block_size: usize) -> Self`, `len(&self) -> usize`, `is_empty(&self) -> bool`, `capacity(&self) -> usize`, `push_block(&mut self, block_id: usize)`, `advance(&mut self, n: usize)`, `slot(&self, pos: usize) -> usize`, `slots(&self, start: usize, end: usize) -> Vec<u32>`.

- [ ] **Step 1: Write the failing test**

Add inside `mod tests` in `src/models/common/paged.rs`:

```rust
#[test]
fn block_table_maps_logical_positions_to_physical_slots() {
    let mut table = BlockTable::new(2); // block_size = 2
    assert!(table.is_empty());
    table.push_block(5); // logical block 0 -> physical block 5
    table.push_block(3); // logical block 1 -> physical block 3
    assert_eq!(table.capacity(), 4);

    // pos 0,1 live in physical block 5 -> slots 10,11; pos 2,3 in block 3 -> slots 6,7
    assert_eq!(table.slot(0), 10);
    assert_eq!(table.slot(1), 11);
    assert_eq!(table.slot(2), 6);
    assert_eq!(table.slot(3), 7);

    table.advance(3);
    assert_eq!(table.len(), 3);
    assert_eq!(table.slots(0, 3), vec![10u32, 11, 6]);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib paged::tests::block_table`
Expected: FAIL — `cannot find type BlockTable in this scope`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/models/common/paged.rs` (above the test module, below `BlockAllocator`):

```rust
/// Per-sequence mapping from logical token positions to physical pool slots.
#[derive(Debug)]
pub struct BlockTable {
    blocks: Vec<usize>,
    block_size: usize,
    len: usize,
}

impl BlockTable {
    pub fn new(block_size: usize) -> Self {
        Self {
            blocks: Vec::new(),
            block_size,
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total token slots currently backed by allocated blocks.
    pub fn capacity(&self) -> usize {
        self.blocks.len() * self.block_size
    }

    pub fn push_block(&mut self, block_id: usize) {
        self.blocks.push(block_id);
    }

    pub fn advance(&mut self, n: usize) {
        self.len += n;
    }

    /// Physical pool slot index for a logical token position.
    pub fn slot(&self, pos: usize) -> usize {
        self.blocks[pos / self.block_size] * self.block_size + pos % self.block_size
    }

    /// Physical slots for logical positions `start..end`, as u32 (index_select dtype).
    pub fn slots(&self, start: usize, end: usize) -> Vec<u32> {
        (start..end).map(|p| self.slot(p) as u32).collect()
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib paged::tests::block_table`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/common/paged.rs
git commit -m "feat: add BlockTable mapping logical positions to physical slots"
```

---

### Task 3: PagedKvCache (slot-addressed write + gather)

**Files:**
- Modify: `src/models/common/paged.rs`
- Test: `src/models/common/paged.rs` (same test module)

**Interfaces:**
- Consumes: nothing (operates on raw slot indices).
- Produces:
  - `PagedKvCache::new(num_layers: usize, num_slots: usize, kv_heads: usize, head_dim: usize, device: &Device) -> Result<Self>`
  - `write(&mut self, layer_idx: usize, slots: &[u32], k: &Tensor, v: &Tensor) -> Result<()>` — `k`/`v` shaped `(1, kv_heads, slots.len(), head_dim)`.
  - `gather(&self, layer_idx: usize, slots: &[u32]) -> Result<(Tensor, Tensor)>` — returns `(1, kv_heads, slots.len(), head_dim)` for each.

- [ ] **Step 1: Add imports**

At the very top of `src/models/common/paged.rs`, add:

```rust
use candle_core::{DType, Device, Result, Tensor};
```

- [ ] **Step 2: Write the failing test**

Add inside `mod tests`:

```rust
use candle_core::{Device, Tensor};

#[test]
fn paged_pool_write_then_gather_round_trips() {
    // 1 layer, 4 slots, 1 kv head, head_dim 2.
    let dev = Device::Cpu;
    let mut pool = PagedKvCache::new(1, 4, 1, 2, &dev).unwrap();

    // Two tokens written to slots 2 and 0 (deliberately out of order).
    // k shape: (1, kv_heads=1, seq=2, head_dim=2)
    let k = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 1, 2, 2), &dev).unwrap();
    let v = Tensor::from_vec(vec![5f32, 6.0, 7.0, 8.0], (1, 1, 2, 2), &dev).unwrap();
    pool.write(0, &[2u32, 0], &k, &v).unwrap();

    // Gather slots 0 then 2 -> rows should be [3,4] then [1,2] for k.
    let (gk, gv) = pool.gather(0, &[0u32, 2]).unwrap();
    assert_eq!(gk.dims(), &[1, 1, 2, 2]);
    let gk: Vec<f32> = gk.flatten_all().unwrap().to_vec1().unwrap();
    let gv: Vec<f32> = gv.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(gk, vec![3.0, 4.0, 1.0, 2.0]);
    assert_eq!(gv, vec![7.0, 8.0, 5.0, 6.0]);
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test --lib paged::tests::paged_pool`
Expected: FAIL — `cannot find type PagedKvCache in this scope`.

- [ ] **Step 4: Write minimal implementation**

Add to `src/models/common/paged.rs` (above the test module, below `BlockTable`):

```rust
/// Physical KV storage: per layer, a flat pool of `(num_slots, kv_heads, head_dim)`
/// for K and V. Writes address individual slots; gather reads a slot list back into
/// a contiguous `(1, kv_heads, n, head_dim)` tensor for the dense attention kernel.
#[derive(Debug)]
pub struct PagedKvCache {
    k_pools: Vec<Tensor>,
    v_pools: Vec<Tensor>,
    kv_heads: usize,
    head_dim: usize,
    device: Device,
}

impl PagedKvCache {
    pub fn new(
        num_layers: usize,
        num_slots: usize,
        kv_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let mut k_pools = Vec::with_capacity(num_layers);
        let mut v_pools = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k_pools.push(Tensor::zeros((num_slots, kv_heads, head_dim), DType::F32, device)?);
            v_pools.push(Tensor::zeros((num_slots, kv_heads, head_dim), DType::F32, device)?);
        }
        Ok(Self {
            k_pools,
            v_pools,
            kv_heads,
            head_dim,
            device: device.clone(),
        })
    }

    pub fn write(&mut self, layer_idx: usize, slots: &[u32], k: &Tensor, v: &Tensor) -> Result<()> {
        // (1, kv_heads, seq, head_dim) -> (seq, kv_heads, head_dim)
        let kt = k.squeeze(0)?.transpose(0, 1)?.contiguous()?;
        let vt = v.squeeze(0)?.transpose(0, 1)?.contiguous()?;
        for (i, &slot) in slots.iter().enumerate() {
            let slot = slot as usize;
            let k_tok = kt.narrow(0, i, 1)?; // (1, kv_heads, head_dim)
            let v_tok = vt.narrow(0, i, 1)?;
            let ranges = [slot..slot + 1, 0..self.kv_heads, 0..self.head_dim];
            self.k_pools[layer_idx] = self.k_pools[layer_idx].slice_assign(&ranges, &k_tok)?;
            self.v_pools[layer_idx] = self.v_pools[layer_idx].slice_assign(&ranges, &v_tok)?;
        }
        Ok(())
    }

    pub fn gather(&self, layer_idx: usize, slots: &[u32]) -> Result<(Tensor, Tensor)> {
        let n = slots.len();
        let idx = Tensor::new(slots, &self.device)?;
        let k = self.k_pools[layer_idx].index_select(&idx, 0)?; // (n, kv_heads, head_dim)
        let v = self.v_pools[layer_idx].index_select(&idx, 0)?;
        // (n, kv_heads, head_dim) -> (1, kv_heads, n, head_dim)
        let k = k
            .reshape((1, n, self.kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((1, n, self.kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        Ok((k, v))
    }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test --lib paged::tests::paged_pool`
Expected: PASS.

- [ ] **Step 6: Run the whole paged module and format**

Run: `cargo test --lib paged && cargo fmt && cargo clippy --lib -- -D warnings`
Expected: all paged tests PASS; clippy clean.

- [ ] **Step 7: Commit**

```bash
git add src/models/common/paged.rs
git commit -m "feat: add PagedKvCache slot-addressed write and gather"
```

---

### Task 4: Wire Cache onto the paged subsystem

**Files:**
- Modify: `src/models/common/cache.rs:1-86` (struct fields, `new`, `append_kv`; keep `rope_*` and `causal_mask` as-is)
- Test: `src/models/common/cache.rs` (existing `mod tests` stay; add one paged-specific test)

**Interfaces:**
- Consumes: `BlockAllocator`, `BlockTable`, `PagedKvCache` (Task 1–3).
- Produces: `Cache::append_kv(&mut self, layer_idx: usize, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)>` — unchanged signature; `Cache::new(cfg: &Config, device: &Device) -> Result<Self>` — unchanged signature.

- [ ] **Step 1: Update imports and the `BLOCK_SIZE` constant**

In `src/models/common/cache.rs`, replace the import block (lines 1-5) with:

```rust
use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};

use super::paged::{BlockAllocator, BlockTable, PagedKvCache};
use super::Config;

const BLOCK_SIZE: usize = 16;
```

- [ ] **Step 2: Replace the struct fields**

Replace the `pub struct Cache { ... }` block (lines 7-15) with:

```rust
#[derive(Debug)]
pub struct Cache {
    cos: Tensor,
    sin: Tensor,
    masks: HashMap<(usize, usize), Tensor>,
    allocator: BlockAllocator,
    table: BlockTable,
    pool: PagedKvCache,
    max_seq_len: usize,
    device: Device,
    // Per-forward scratch, (re)computed on layer 0 and reused by later layers.
    pending_write_slots: Vec<u32>,
    pending_all_slots: Vec<u32>,
}
```

- [ ] **Step 3: Replace the tail of `new` that builds the struct**

In `Cache::new`, replace the `Ok(Self { ... })` block (currently lines 33-40, the one that sets `kvs: vec![None; ...]`) with:

```rust
        let num_blocks = cfg.max_seq_len.div_ceil(BLOCK_SIZE);
        let num_slots = num_blocks * BLOCK_SIZE;
        let allocator = BlockAllocator::new(num_blocks);
        let table = BlockTable::new(BLOCK_SIZE);
        let pool = PagedKvCache::new(
            cfg.num_hidden_layers,
            num_slots,
            cfg.num_key_value_heads,
            cfg.head_dim,
            device,
        )?;

        Ok(Self {
            cos,
            sin,
            masks: HashMap::new(),
            allocator,
            table,
            pool,
            max_seq_len: cfg.max_seq_len,
            device: device.clone(),
            pending_write_slots: Vec::new(),
            pending_all_slots: Vec::new(),
        })
```

- [ ] **Step 4: Replace `append_kv`**

Replace the entire `append_kv` method (lines 66-85) with:

```rust
    pub fn append_kv(
        &mut self,
        layer_idx: usize,
        k: Tensor,
        v: Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = k.dims()[2];

        // Allocation + position bookkeeping happen once per forward, on the first layer.
        if layer_idx == 0 {
            let start = self.table.len();
            let end = start + seq_len;
            if end > self.max_seq_len {
                candle_core::bail!(
                    "paged kv cache overflow: {end} tokens exceeds max_seq_len {}",
                    self.max_seq_len
                );
            }
            while self.table.capacity() < end {
                let block = self
                    .allocator
                    .allocate()
                    .ok_or_else(|| candle_core::Error::Msg("out of kv blocks".into()))?;
                self.table.push_block(block);
            }
            self.pending_write_slots = self.table.slots(start, end);
            self.table.advance(seq_len);
            self.pending_all_slots = self.table.slots(0, self.table.len());
        }

        // Clone the slot lists to release the immutable borrow of `self` before
        // taking the mutable borrow of `self.pool`.
        let write_slots = self.pending_write_slots.clone();
        let all_slots = self.pending_all_slots.clone();
        self.pool.write(layer_idx, &write_slots, &k, &v)?;
        self.pool.gather(layer_idx, &all_slots)
    }
```

- [ ] **Step 5: Add a paged-behavior test**

Add inside the existing `mod tests` in `src/models/common/cache.rs` (the `test_config` there uses `num_hidden_layers: 2`, `num_key_value_heads: 2`, `head_dim: 4`, `max_seq_len: 16`):

```rust
#[test]
fn append_kv_overflow_past_max_seq_len_errors() {
    let mut cache = Cache::new(&test_config(), &Device::Cpu).unwrap();
    // max_seq_len = 16; one shot of 17 tokens must error.
    let k = candle_core::Tensor::zeros((1, 2, 17, 4), candle_core::DType::F32, &Device::Cpu)
        .unwrap();
    let v = k.clone();
    assert!(cache.append_kv(0, k, v).is_err());
}

#[test]
fn append_kv_across_two_layers_shares_positions() {
    let mut cache = Cache::new(&test_config(), &Device::Cpu).unwrap();
    let k = candle_core::Tensor::zeros((1, 2, 3, 4), candle_core::DType::F32, &Device::Cpu)
        .unwrap();
    let v = k.clone();
    // Layer 0 allocates and advances; layer 1 reuses the same slots, no double-advance.
    let (k0, _) = cache.append_kv(0, k.clone(), v.clone()).unwrap();
    let (k1, _) = cache.append_kv(1, k, v).unwrap();
    assert_eq!(k0.dims(), &[1, 2, 3, 4]);
    assert_eq!(k1.dims(), &[1, 2, 3, 4]);
}
```

- [ ] **Step 6: Run the cache tests (existing + new) to verify they pass**

Run: `cargo test --lib models::common::cache`
Expected: PASS — including the pre-existing `append_kv_concatenates_along_seq_dim`, `causal_mask_*`, `rope_*`, plus the two new tests.

- [ ] **Step 7: Commit**

```bash
git add src/models/common/cache.rs
git commit -m "feat: back Cache with paged KV pool instead of contiguous cat"
```

---

### Task 5: End-to-end parity verification

**Files:**
- Test only (no production changes). Exercises `attention.rs`, `model.rs`, and `engine.rs` tests that drive `Cache` through the real forward path.

**Interfaces:**
- Consumes: the rewritten `Cache` (Task 4). Produces: nothing new.

- [ ] **Step 1: Run the attention + model forward tests**

Run: `cargo test --lib models::common`
Expected: PASS — `forward_preserves_shape_prefill`, `forward_works_with_qkv_bias_and_qk_norm`, `forward_single_token_decode_step_after_prefill`, and the model `forward_*` tests all still pass through the paged cache.

- [ ] **Step 2: Run the engine end-to-end tests**

Run: `cargo test --lib engine::tests`
Expected: PASS — `load_and_generate_end_to_end_with_tiny_random_model` and `generate_is_deterministic_for_a_fixed_seed`. Determinism passing confirms the paged write/gather is order-stable.

- [ ] **Step 3: Run the full suite, format, and lint**

Run: `cargo test && cargo fmt --check && cargo clippy --all-targets -- -D warnings`
Expected: entire suite PASS; formatting clean; no clippy warnings.

- [ ] **Step 4: Commit (only if Step 3 surfaced a fmt/clippy fix; otherwise skip)**

```bash
git add -A
git commit -m "chore: fmt and clippy clean for paged KV cache"
```

---

## Self-Review

**Spec coverage:**
- Paged physical storage → Task 3 (`PagedKvCache` pool). ✓
- Block allocator with free list → Task 1. ✓
- Block table (logical→physical mapping) → Task 2. ✓
- Drop-in `append_kv` preserving the attention contract → Task 4. ✓
- No regression to existing forward/engine behavior → Task 5. ✓
- Fused paged kernel / scheduler / CoW / eviction → explicitly Out of Scope, named as follow-up plans. ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to" — every code step is complete. ✓

**Type consistency:** `BlockAllocator::{new, allocate, free_block, num_free}`, `BlockTable::{new, len, is_empty, capacity, push_block, advance, slot, slots}`, `PagedKvCache::{new, write, gather}` are defined in Tasks 1–3 and used with identical names/signatures in Task 4. `append_kv` and `Cache::new` keep their original signatures. `slots()` returns `Vec<u32>`, matching `index_select`'s integer-index requirement and `write`/`gather`'s `&[u32]` params. ✓

---

Plan complete and saved to `docs/superpowers/plans/2026-06-20-paged-kv-cache.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
