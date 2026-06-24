use candle_core::{DType, Device, Result, Tensor};

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
        debug_assert!(
            self.ref_counts[block_id] > 0,
            "decref of unreferenced block"
        );
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

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Total token slots currently backed by allocated blocks.
    pub fn capacity(&self) -> usize {
        self.blocks.len() * self.block_size
    }

    pub fn push_block(&mut self, block_id: usize) {
        self.blocks.push(block_id);
    }

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

/// Physical KV storage: per layer, a flat pool of `(kv_heads, num_slots, head_dim)`
/// for K and V. The slot axis sits in the middle so the attention-ready layout
/// `(1, kv_heads, n, head_dim)` is reachable by `narrow` + `unsqueeze` alone — reading a
/// contiguous run of slots (one paged block) is a zero-copy view, never a gather.
#[derive(Debug)]
pub struct PagedKvPool {
    k_pools: Vec<Tensor>,
    v_pools: Vec<Tensor>,
    kv_heads: usize,
    head_dim: usize,
    device: Device,
}

impl PagedKvPool {
    pub fn new(
        num_layers: usize,
        num_slots: usize,
        kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let mut k_pools = Vec::with_capacity(num_layers);
        let mut v_pools = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k_pools.push(Tensor::zeros(
                (kv_heads, num_slots, head_dim),
                dtype,
                device,
            )?);
            v_pools.push(Tensor::zeros(
                (kv_heads, num_slots, head_dim),
                dtype,
                device,
            )?);
        }
        Ok(Self {
            k_pools,
            v_pools,
            kv_heads,
            head_dim,
            device: device.clone(),
        })
    }

    pub fn num_layers(&self) -> usize {
        self.k_pools.len()
    }

    pub fn kv_heads(&self) -> usize {
        self.kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Element dtype of the pooled K/V tensors (all layers share one dtype).
    pub fn dtype(&self) -> DType {
        self.k_pools[0].dtype()
    }

    pub fn write(&mut self, layer_idx: usize, slots: &[u32], k: &Tensor, v: &Tensor) -> Result<()> {
        // (1, kv_heads, seq, head_dim) -> (kv_heads, seq, head_dim); slot axis already aligned.
        let kt = k.squeeze(0)?;
        let vt = v.squeeze(0)?;
        let k_pool = &self.k_pools[layer_idx];
        let v_pool = &self.v_pools[layer_idx];
        let mut i = 0;
        while i < slots.len() {
            let mut n = 1;
            let start_slot = slots[i] as usize;
            while i + n < slots.len() && i + n == start_slot + n {
                n += 1;
            }
            k_pool.slice_set(&kt.narrow(1, i, n)?.contiguous()?, 1, start_slot)?;
            v_pool.slice_set(&vt.narrow(1, i, n)?.contiguous()?, 1, start_slot)?;
            i += n;
        }
        Ok(())
    }

    /// Overwrite one full physical block (`block_id`) with imported K/V. Accepts the
    /// attention-ready `(1, kv_heads, block_size, head_dim)` layout (squeezed to
    /// `(kv_heads, block_size, head_dim)`) or that 3-D shape directly. The block occupies the
    /// contiguous slot run `block_id * block_size .. + block_size`, so the write is a single
    /// `slice_set` per layer. Used by the remote-KV import path.
    pub fn write_block(
        &mut self,
        layer_idx: usize,
        block_id: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<()> {
        let kt = Self::normalize_block(k)?;
        let vt = Self::normalize_block(v)?;
        let (kh, block_size, hd) = kt.dims3()?;
        let (vkh, vbs, vhd) = vt.dims3()?;
        if (kh, block_size, hd) != (vkh, vbs, vhd) {
            candle_core::bail!(
                "write_block k/v shape mismatch: k={:?} v={:?}",
                kt.dims(),
                vt.dims()
            );
        }
        if kh != self.kv_heads || hd != self.head_dim {
            candle_core::bail!(
                "write_block expects (.., kv_heads={}, block_size, head_dim={}), got kv_heads={kh} head_dim={hd}",
                self.kv_heads,
                self.head_dim
            );
        }
        let start_slot = block_id * block_size;
        self.k_pools[layer_idx].slice_set(&kt.contiguous()?, 1, start_slot)?;
        self.v_pools[layer_idx].slice_set(&vt.contiguous()?, 1, start_slot)?;
        Ok(())
    }

    /// Normalize a block tensor to `(kv_heads, block_size, head_dim)`: squeeze a leading batch
    /// axis of 1 if present, accept the 3-D shape as-is, reject anything else.
    fn normalize_block(t: &Tensor) -> Result<Tensor> {
        match t.rank() {
            4 => t.squeeze(0),
            3 => Ok(t.clone()),
            r => candle_core::bail!("write_block expects rank 3 or 4 tensor, got rank {r}"),
        }
    }

    /// Read an arbitrary slot list back as `(1, kv_heads, n, head_dim)`. The gather (one
    /// `index_select`) is unavoidable for non-contiguous slots; prefer [`Self::block_view`]
    /// when the slots form a contiguous run.
    pub fn gather(&self, layer_idx: usize, slots: &[u32]) -> Result<(Tensor, Tensor)> {
        let idx = Tensor::new(slots, &self.device)?;
        let k = self.k_pools[layer_idx].index_select(&idx, 1)?; // (kv_heads, n, head_dim)
        let v = self.v_pools[layer_idx].index_select(&idx, 1)?;
        Ok((k.unsqueeze(0)?, v.unsqueeze(0)?))
    }

    /// Zero-copy view of `n` contiguous slots starting at `start_slot`, as
    /// `(1, kv_heads, n, head_dim)`. Used to feed one paged block straight into the
    /// PagedAttention kernel without materializing a copy. Slots within a single paged block
    /// are always physically contiguous (see `BlockTable::slot`), so this is the read path.
    pub fn block_view(
        &self,
        layer_idx: usize,
        start_slot: usize,
        n: usize,
    ) -> Result<(Tensor, Tensor)> {
        let k = self.k_pools[layer_idx]
            .narrow(1, start_slot, n)?
            .unsqueeze(0)?;
        let v = self.v_pools[layer_idx]
            .narrow(1, start_slot, n)?
            .unsqueeze(0)?;
        Ok((k, v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn paged_pool_write_then_gather_round_trips() {
        // 1 layer, 4 slots, 1 kv head, head_dim 2.
        let dev = Device::Cpu;
        let mut pool = PagedKvPool::new(1, 4, 1, 2, DType::F32, &dev).unwrap();

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

    #[test]
    fn paged_pool_write_block_round_trips_full_block() {
        // 1 layer, 4 slots (= 2 physical blocks of size 2), 1 kv head, head_dim 2.
        let dev = Device::Cpu;
        let mut pool = PagedKvPool::new(1, 4, 1, 2, DType::F32, &dev).unwrap();

        // Write a full block of 2 slots into physical block 1 (slots 2..4).
        let k = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 1, 2, 2), &dev).unwrap();
        let v = Tensor::from_vec(vec![5f32, 6.0, 7.0, 8.0], (1, 1, 2, 2), &dev).unwrap();
        pool.write_block(0, 1, &k, &v).unwrap();

        // block_view reads back the contiguous run at start_slot = block_id * block_size = 2.
        let (rk, rv) = pool.block_view(0, 2, 2).unwrap();
        assert_eq!(rk.dims(), &[1, 1, 2, 2]);
        let rk: Vec<f32> = rk.flatten_all().unwrap().to_vec1().unwrap();
        let rv: Vec<f32> = rv.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(rk, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(rv, vec![5.0, 6.0, 7.0, 8.0]);

        // Physical block 0 (slots 0..2) is untouched and still zero.
        let (zk, _) = pool.block_view(0, 0, 2).unwrap();
        let zk: Vec<f32> = zk.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(zk, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn write_block_rejects_mismatched_kv_heads() {
        let dev = Device::Cpu;
        let mut pool = PagedKvPool::new(1, 4, 1, 2, DType::F32, &dev).unwrap();
        // kv_heads = 2 but the pool was built with 1.
        let k = Tensor::zeros((1, 2, 2, 2), DType::F32, &dev).unwrap();
        assert!(pool.write_block(0, 0, &k, &k).is_err());
    }

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
}
