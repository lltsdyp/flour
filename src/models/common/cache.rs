use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};

use super::paged::{BlockAllocator, BlockTable, PagedKvPool};
use super::prefix::{block_hash, PrefixRegistry, PREFIX_HASH_SEED};
use super::Config;
use crate::kv_cache::bundle::{BundleDType, KvBundle, KvBundleBlock, KvBundleMeta, KvLayerBlock};

const BLOCK_SIZE: usize = 16;

#[derive(Debug)]
pub struct KvCache {
    pool: PagedKvPool,
    allocator: BlockAllocator,
    table: BlockTable,
    registry: PrefixRegistry,
}

impl KvCache {
    pub fn new(cfg: &Config, dtype: DType, device: &Device) -> Result<Self> {
        let num_blocks = cfg.max_seq_len.div_ceil(BLOCK_SIZE);
        let num_slots = num_blocks * BLOCK_SIZE;
        let allocator = BlockAllocator::new(num_blocks);
        let table = BlockTable::new(BLOCK_SIZE);
        let pool = PagedKvPool::new(
            cfg.num_hidden_layers,
            num_slots,
            cfg.num_key_value_heads,
            cfg.head_dim,
            dtype,
            device,
        )?;

        Ok(Self {
            pool,
            allocator,
            table,
            registry: PrefixRegistry::new(),
        })
    }

    pub fn allocate(&mut self) -> Option<usize> {
        let block = self.allocator.allocate()?;
        self.table.push_block(block);
        Some(block)
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.table.capacity()
    }

    pub fn slots(&self, start: usize, end: usize) -> Vec<u32> {
        self.table.slots(start, end)
    }

    pub fn advance(&mut self, seq_len: usize) {
        self.table.advance(seq_len);
    }

    /// Write the trailing `seq_len` tokens of this batch into their reserved slots.
    /// Blocks must already be reserved by `Cache::allocate_kv` before calling.
    pub fn write_current(&mut self, layer_idx: usize, k: &Tensor, v: &Tensor) -> Result<()> {
        let seq_len = k.dims()[2];
        let end = self.table.len();
        let start = end - seq_len;
        let slots = self.table.slots(start, end);
        self.pool.write(layer_idx, &slots, k, v)
    }

    /// View the live KV (logical positions `0..len`) one physical block at a time, in
    /// logical order. Each entry is a zero-copy `(k, v)` view shaped
    /// `(1, kv_heads, block_n, head_dim)`: slots inside a block are physically contiguous,
    /// so each block is a `narrow` view rather than a gather.
    pub fn gather_blocks(&self, layer_idx: usize) -> Result<Vec<(Tensor, Tensor)>> {
        let len = self.table.len();
        let block_size = self.table.block_size();
        let num_blocks = len.div_ceil(block_size);
        let mut blocks = Vec::with_capacity(num_blocks);
        for b in 0..num_blocks {
            let start = b * block_size;
            let n = block_size.min(len - start);
            // The first token of a logical block maps to the start of its physical block.
            let start_slot = self.table.slot(start);
            blocks.push(self.pool.block_view(layer_idx, start_slot, n)?);
        }
        Ok(blocks)
    }

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
        let max_reusable = if token_ids.len().is_multiple_of(bs) {
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

    /// Export the first `token_count` live tokens as a portable [`KvBundle`]. `token_count` must
    /// be block-aligned and within the live sequence length. Reads the real K/V for every layer
    /// of every covered logical block straight from the pool (zero-copy views, then a contiguous
    /// copy into the bundle). Returns an error — not a panic — for misaligned counts or
    /// unrepresentable dtypes so the caller can degrade to a remote miss.
    pub fn export_prefix_bundle(
        &self,
        model_id: &str,
        token_ids: &[u32],
        token_count: usize,
    ) -> Result<KvBundle> {
        let bs = self.table.block_size();
        if token_count == 0 || !token_count.is_multiple_of(bs) {
            candle_core::bail!("export token_count {token_count} not aligned to block size {bs}");
        }
        if token_count > self.table.len() {
            candle_core::bail!(
                "export token_count {token_count} exceeds live len {}",
                self.table.len()
            );
        }
        if token_ids.len() < token_count {
            candle_core::bail!(
                "token_ids len {} shorter than token_count {token_count}",
                token_ids.len()
            );
        }
        let dtype = BundleDType::from_candle(self.pool.dtype()).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "dtype {:?} is not representable in a kv bundle",
                self.pool.dtype()
            ))
        })?;

        let num_layers = self.pool.num_layers();
        let num_blocks = token_count / bs;
        let mut blocks = Vec::with_capacity(num_blocks);
        for b in 0..num_blocks {
            // The first token of logical block `b` maps to the start of its physical block.
            let start_slot = self.table.slot(b * bs);
            let mut layers = Vec::with_capacity(num_layers);
            for layer in 0..num_layers {
                let (k, v) = self.pool.block_view(layer, start_slot, bs)?;
                layers.push(KvLayerBlock {
                    k: k.squeeze(0)?.contiguous()?,
                    v: v.squeeze(0)?.contiguous()?,
                });
            }
            blocks.push(KvBundleBlock {
                logical_block_idx: b,
                layers,
            });
        }

        Ok(KvBundle {
            meta: KvBundleMeta {
                model_id: model_id.to_string(),
                token_count,
                token_ids: token_ids[..token_count].to_vec(),
                block_size: bs,
                num_layers,
                num_kv_heads: self.pool.kv_heads(),
                head_dim: self.pool.head_dim(),
                dtype,
            },
            blocks,
        })
    }

    /// Import a remote [`KvBundle`] into a fresh sequence: validate it against the local model and
    /// cache config, allocate one physical block per bundle block, write every layer's K/V, push
    /// the blocks into the table, and register them in the prefix registry so the suffix forward
    /// and later requests can reuse them. Returns the imported token count.
    ///
    /// Rejects (without leaving partial state) a bundle whose config, dtype, block alignment, or
    /// token prefix does not match `prompt_tokens`. On a mid-write failure, releases everything it
    /// allocated and resets the sequence, so a caller can safely fall back to cold prefill.
    pub fn import_prefix_bundle(
        &mut self,
        bundle: &KvBundle,
        prompt_tokens: &[u32],
    ) -> Result<usize> {
        let meta = &bundle.meta;
        let bs = self.table.block_size();

        if meta.block_size != bs {
            candle_core::bail!(
                "bundle block_size {} != cache block_size {bs}",
                meta.block_size
            );
        }
        if meta.num_layers != self.pool.num_layers() {
            candle_core::bail!(
                "bundle num_layers {} != model {}",
                meta.num_layers,
                self.pool.num_layers()
            );
        }
        if meta.num_kv_heads != self.pool.kv_heads() {
            candle_core::bail!(
                "bundle num_kv_heads {} != model {}",
                meta.num_kv_heads,
                self.pool.kv_heads()
            );
        }
        if meta.head_dim != self.pool.head_dim() {
            candle_core::bail!(
                "bundle head_dim {} != model {}",
                meta.head_dim,
                self.pool.head_dim()
            );
        }
        if BundleDType::from_candle(self.pool.dtype()) != Some(meta.dtype) {
            candle_core::bail!("bundle dtype {:?} != model dtype", meta.dtype);
        }
        if meta.token_count == 0 || !meta.token_count.is_multiple_of(bs) {
            candle_core::bail!("bundle token_count {} not block aligned", meta.token_count);
        }
        if meta.token_ids.len() != meta.token_count {
            candle_core::bail!("bundle token_ids len != token_count");
        }
        if prompt_tokens.len() < meta.token_count
            || prompt_tokens[..meta.token_count] != meta.token_ids[..]
        {
            candle_core::bail!("prompt does not start with bundle token ids");
        }
        let num_blocks = meta.token_count / bs;
        if bundle.blocks.len() != num_blocks {
            candle_core::bail!(
                "bundle has {} blocks, expected {num_blocks}",
                bundle.blocks.len()
            );
        }

        // Start a clean sequence, then ensure there is room for every imported block.
        self.reset_sequence();
        if self.allocator.num_free() < num_blocks {
            candle_core::bail!(
                "not enough free kv blocks to import ({} < {num_blocks})",
                self.allocator.num_free()
            );
        }

        let mut allocated: Vec<usize> = Vec::with_capacity(num_blocks);
        let write_result = (|| -> Result<()> {
            for block in &bundle.blocks {
                if block.layers.len() != meta.num_layers {
                    candle_core::bail!("bundle block has wrong layer count");
                }
                let block_id = self
                    .allocator
                    .allocate()
                    .ok_or_else(|| candle_core::Error::Msg("out of kv blocks".into()))?;
                allocated.push(block_id);
                for (layer, lb) in block.layers.iter().enumerate() {
                    self.pool.write_block(layer, block_id, &lb.k, &lb.v)?;
                }
                self.table.push_block(block_id);
            }
            Ok(())
        })();

        if let Err(e) = write_result {
            // Roll back: drop every block we allocated, clear the half-built table.
            for id in &allocated {
                self.allocator.decref(*id);
            }
            self.table.clear();
            return Err(e);
        }

        self.table.advance(meta.token_count);

        // Register imported blocks with the same chained hash + extra refcount as local prefill,
        // so reset + match_prefix reuses them exactly like a locally-computed prefix.
        let mut parent = PREFIX_HASH_SEED;
        for b in 0..num_blocks {
            let chunk = &meta.token_ids[b * bs..(b + 1) * bs];
            let h = block_hash(parent, chunk);
            if !self.registry.contains(h) {
                let block_id = self.table.block_at(b);
                self.allocator.incref(block_id);
                self.registry.insert(h, chunk.to_vec(), block_id);
            }
            parent = h;
        }

        Ok(meta.token_count)
    }

    /// Drop all cached prefixes, releasing the registry's reference on each cached block. The
    /// live sequence's own references are untouched, so an in-flight request is unaffected;
    /// blocks referenced only by the registry return to the free list.
    pub fn clear_prefix_cache(&mut self) {
        for block_id in self.registry.drain_block_ids() {
            self.allocator.decref(block_id);
        }
    }

    #[cfg(test)]
    pub fn free_blocks(&self) -> usize {
        self.allocator.num_free()
    }
}

#[derive(Debug)]
pub struct Cache {
    cos: Tensor,
    sin: Tensor,
    masks: HashMap<(usize, usize), Tensor>,
    kvs: KvCache,
    max_seq_len: usize,
    device: Device,
}

impl Cache {
    pub fn new(cfg: &Config, dtype: DType, device: &Device) -> Result<Self> {
        let half_dim = cfg.head_dim / 2;
        let theta: Vec<f32> = (0..half_dim)
            .map(|i| 1f32 / cfg.rope_theta.powf((2 * i) as f32 / cfg.head_dim as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;

        let idx_theta = Tensor::arange(0, cfg.max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_seq_len, 1))?
            .matmul(&theta.reshape((1, half_dim))?)?;

        // RoPE tables are computed in f32 for accuracy, then cast to the model dtype so
        // `candle_nn::rotary_emb::rope` sees q/k/cos/sin all in the same dtype.
        let cos = idx_theta.cos()?.to_dtype(dtype)?.contiguous()?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?.contiguous()?;

        Ok(Self {
            cos,
            sin,
            masks: HashMap::new(),
            kvs: KvCache::new(cfg, dtype, device)?,
            max_seq_len: cfg.max_seq_len,
            device: device.clone(),
        })
    }

    pub fn rope_cos(&self, index_pos: usize, seq_len: usize) -> Result<Tensor> {
        self.cos.narrow(0, index_pos, seq_len)?.contiguous()
    }

    pub fn rope_sin(&self, index_pos: usize, seq_len: usize) -> Result<Tensor> {
        self.sin.narrow(0, index_pos, seq_len)?.contiguous()
    }

    /// Build a `(seq_len, kv_seq_len)` mask where 1 = masked out. New query positions occupy
    /// the last `seq_len` kv columns; any earlier (already-cached) kv columns are always visible.
    pub fn causal_mask(&mut self, seq_len: usize, kv_seq_len: usize) -> Result<Tensor> {
        let key = (seq_len, kv_seq_len);
        if let Some(mask) = self.masks.get(&key) {
            return Ok(mask.clone());
        }

        let offset = kv_seq_len - seq_len;
        let data: Vec<u8> = (0..seq_len)
            .flat_map(|i| (0..kv_seq_len).map(move |j| u8::from(j > i + offset)))
            .collect();
        let mask = Tensor::from_slice(&data, (seq_len, kv_seq_len), &self.device)?;
        self.masks.insert(key, mask.clone());
        Ok(mask)
    }

    pub fn allocate_kv(&mut self, seq_len: usize) -> Result<()> {
        let start = self.kvs.len();
        let end = start + seq_len;

        if end > self.max_seq_len {
            candle_core::bail!(
                "paged kv cache overflow: {end} tokens exceeds max_seq_len {}",
                self.max_seq_len
            );
        }

        while self.kvs.capacity() < end {
            self.kvs
                .allocate()
                .ok_or_else(|| candle_core::Error::Msg("out of kv blocks".into()))?;
        }
        self.kvs.advance(seq_len);
        Ok(())
    }

    /// Persist this batch's K/V into the paged pool without gathering it back. Pair with
    /// [`Cache::kv_blocks`] + `backend::cpu::paged_attention` for the streaming kernel path.
    pub fn write_kv(&mut self, layer_idx: usize, k: &Tensor, v: &Tensor) -> Result<()> {
        self.kvs.write_current(layer_idx, k, v)
    }

    /// The full live KV for a layer, returned block-by-block in logical order.
    pub fn kv_blocks(&self, layer_idx: usize) -> Result<Vec<(Tensor, Tensor)>> {
        self.kvs.gather_blocks(layer_idx)
    }

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

    /// Export the first `token_count` live tokens as a portable [`KvBundle`] for cross-node reuse.
    pub fn export_prefix_bundle(
        &self,
        model_id: &str,
        token_ids: &[u32],
        token_count: usize,
    ) -> Result<KvBundle> {
        self.kvs
            .export_prefix_bundle(model_id, token_ids, token_count)
    }

    /// Import a remote [`KvBundle`] into a fresh sequence and register its blocks for reuse.
    /// Returns the imported token count, or an error (leaving no partial state) when the bundle
    /// is incompatible with this model/cache or with `prompt_tokens`.
    pub fn import_prefix_bundle(
        &mut self,
        bundle: &KvBundle,
        prompt_tokens: &[u32],
    ) -> Result<usize> {
        self.kvs.import_prefix_bundle(bundle, prompt_tokens)
    }

    /// Drop all cached prefixes (escape hatch; does not disturb the live sequence).
    pub fn clear_prefix_cache(&mut self) {
        self.kvs.clear_prefix_cache();
    }

    #[cfg(test)]
    pub fn free_blocks_for_test(&self) -> usize {
        self.kvs.free_blocks()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::common::Config;
    use candle_core::Device;

    fn test_config() -> Config {
        Config {
            hidden_size: 8,
            intermediate_size: 16,
            vocab_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            max_seq_len: 16,
            use_qkv_bias: false,
            use_qk_norm: false,
            tie_word_embeddings: false,
            eos_token_id: None,
        }
    }

    #[test]
    fn rope_tables_have_expected_shape() {
        let cache = Cache::new(&test_config(), DType::F32, &Device::Cpu).unwrap();
        let cos = cache.rope_cos(0, 5).unwrap();
        assert_eq!(cos.dims(), &[5, 2]); // head_dim/2 = 2
    }

    #[test]
    fn rope_tables_respect_index_pos_offset() {
        let cache = Cache::new(&test_config(), DType::F32, &Device::Cpu).unwrap();
        let cos_a = cache
            .rope_cos(0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let cos_b = cache
            .rope_cos(3, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_ne!(cos_a, cos_b);
    }

    #[test]
    fn causal_mask_is_upper_triangular() {
        let mut cache = Cache::new(&test_config(), DType::F32, &Device::Cpu).unwrap();
        let mask = cache.causal_mask(3, 3).unwrap();
        let mask: Vec<u8> = mask.flatten_all().unwrap().to_vec1().unwrap();
        // row 0: only col 0 visible -> [0,1,1]; row 1: [0,0,1]; row 2: [0,0,0]
        assert_eq!(mask, vec![0, 1, 1, 0, 0, 1, 0, 0, 0]);
    }

    #[test]
    fn causal_mask_pads_for_existing_kv_prefix() {
        let mut cache = Cache::new(&test_config(), DType::F32, &Device::Cpu).unwrap();
        // 1 new query position, 3 total kv positions (2 already cached) -> fully visible
        let mask = cache.causal_mask(1, 3).unwrap();
        let mask: Vec<u8> = mask.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(mask, vec![0, 0, 0]);
    }

    /// Total kv length recovered from the block views.
    fn block_kv_len(blocks: &[(candle_core::Tensor, candle_core::Tensor)]) -> usize {
        blocks.iter().map(|(k, _)| k.dims()[2]).sum()
    }

    #[test]
    fn write_kv_then_blocks_spans_appended_batches() {
        let mut cache = Cache::new(&test_config(), DType::F32, &Device::Cpu).unwrap();
        cache.allocate_kv(2).unwrap();
        let k1 = candle_core::Tensor::zeros((1, 2, 2, 4), candle_core::DType::F32, &Device::Cpu)
            .unwrap();
        cache.write_kv(0, &k1, &k1).unwrap();
        let blocks = cache.kv_blocks(0).unwrap();
        assert_eq!(block_kv_len(&blocks), 2);
        assert_eq!(blocks[0].0.dims(), &[1, 2, 2, 4]);

        // A second batch extends the live window; block views now cover 2 + 1 = 3 positions.
        cache.allocate_kv(1).unwrap();
        let k2 = candle_core::Tensor::zeros((1, 2, 1, 4), candle_core::DType::F32, &Device::Cpu)
            .unwrap();
        cache.write_kv(0, &k2, &k2).unwrap();
        assert_eq!(block_kv_len(&cache.kv_blocks(0).unwrap()), 3);
    }

    #[test]
    fn allocate_kv_overflow_past_max_seq_len_errors() {
        let mut cache = Cache::new(&test_config(), DType::F32, &Device::Cpu).unwrap();
        // max_seq_len = 16; reserving 17 tokens in one shot must error.
        assert!(cache.allocate_kv(17).is_err());
    }

    #[test]
    fn write_kv_across_two_layers_shares_positions() {
        let mut cache = Cache::new(&test_config(), DType::F32, &Device::Cpu).unwrap();
        let k = candle_core::Tensor::zeros((1, 2, 3, 4), candle_core::DType::F32, &Device::Cpu)
            .unwrap();
        // allocate_kv advances once for the batch; both layers reuse the same slots.
        cache.allocate_kv(3).unwrap();
        cache.write_kv(0, &k, &k).unwrap();
        cache.write_kv(1, &k, &k).unwrap();
        assert_eq!(block_kv_len(&cache.kv_blocks(0).unwrap()), 3);
        assert_eq!(block_kv_len(&cache.kv_blocks(1).unwrap()), 3);
    }

    #[test]
    fn kv_blocks_round_trips_written_values() {
        // Two-token write across a single block must read back identically (zero-copy view).
        let cfg = test_config();
        let mut cache = Cache::new(&cfg, DType::F32, &Device::Cpu).unwrap();
        cache.allocate_kv(2).unwrap();
        let data: Vec<f32> = (0..(2 * 2 * 4)).map(|i| i as f32).collect();
        let k = candle_core::Tensor::from_vec(data.clone(), (1, 2, 2, 4), &Device::Cpu).unwrap();
        cache.write_kv(0, &k, &k).unwrap();
        let blocks = cache.kv_blocks(0).unwrap();
        let got: Vec<f32> = blocks[0].0.flatten_all().unwrap().to_vec1().unwrap();
        let want: Vec<f32> = k.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(got, want);
    }

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
            (
                1,
                prefix_config().num_key_value_heads,
                seq_len,
                prefix_config().head_dim,
            ),
            candle_core::DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        cache.write_kv(0, &k, &k).unwrap();
    }

    #[test]
    fn match_prefix_is_empty_before_anything_is_registered() {
        let mut cache = Cache::new(&prefix_config(), DType::F32, &Device::Cpu).unwrap();
        let ids: Vec<u32> = (0..40).collect(); // 2 full blocks + partial
        cache.reset_sequence();
        assert_eq!(cache.match_prefix(&ids), 0);
    }

    #[test]
    fn register_then_match_reuses_full_blocks_and_leaves_a_suffix() {
        let cfg = prefix_config();
        let mut cache = Cache::new(&cfg, DType::F32, &Device::Cpu).unwrap();
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
        let mut cache = Cache::new(&cfg, DType::F32, &Device::Cpu).unwrap();
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
        let mut cache = Cache::new(&cfg, DType::F32, &Device::Cpu).unwrap();
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
    fn export_prefix_bundle_captures_real_block_kv() {
        let cfg = prefix_config(); // kv_heads 2, head_dim 4, 2 layers, block_size 16
        let mut cache = Cache::new(&cfg, DType::F32, &Device::Cpu).unwrap();
        let ids: Vec<u32> = (0..32u32).collect();
        let kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        cache.reset_sequence();
        cache.allocate_kv(32).unwrap();
        // Distinct, deterministic K/V per layer so values are checkable against the pool.
        for layer in 0..cfg.num_hidden_layers {
            let n = kv_heads * 32 * head_dim;
            let kdata: Vec<f32> = (0..n).map(|i| (layer * 1000 + i) as f32).collect();
            let vdata: Vec<f32> = kdata.iter().map(|x| x + 0.25).collect();
            let k = candle_core::Tensor::from_vec(kdata, (1, kv_heads, 32, head_dim), &Device::Cpu)
                .unwrap();
            let v = candle_core::Tensor::from_vec(vdata, (1, kv_heads, 32, head_dim), &Device::Cpu)
                .unwrap();
            cache.write_kv(layer, &k, &v).unwrap();
        }

        let bundle = cache.export_prefix_bundle("m", &ids, 32).unwrap();
        assert_eq!(bundle.meta.model_id, "m");
        assert_eq!(bundle.meta.token_count, 32);
        assert_eq!(bundle.meta.block_size, 16);
        assert_eq!(bundle.blocks.len(), 2);
        assert_eq!(bundle.meta.num_layers, cfg.num_hidden_layers);
        assert_eq!(bundle.meta.num_kv_heads, kv_heads);
        assert_eq!(bundle.meta.head_dim, head_dim);
        assert_eq!(bundle.meta.dtype, BundleDType::F32);
        assert_eq!(bundle.meta.token_ids, ids);

        // Exported K/V values must equal what the pool block views return.
        for layer in 0..cfg.num_hidden_layers {
            let pool_blocks = cache.kv_blocks(layer).unwrap();
            assert_eq!(pool_blocks.len(), 2);
            for (b, (pk, pv)) in pool_blocks.iter().enumerate() {
                let bk: Vec<f32> = bundle.blocks[b].layers[layer]
                    .k
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap();
                let want_k: Vec<f32> = pk.flatten_all().unwrap().to_vec1().unwrap();
                assert_eq!(bk, want_k, "k mismatch layer {layer} block {b}");
                let bv: Vec<f32> = bundle.blocks[b].layers[layer]
                    .v
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap();
                let want_v: Vec<f32> = pv.flatten_all().unwrap().to_vec1().unwrap();
                assert_eq!(bv, want_v, "v mismatch layer {layer} block {b}");
            }
        }
    }

    /// Build cache A, fill 32 tokens with deterministic K/V, return (cache, bundle, ids 0..32).
    fn cache_with_bundle() -> (Cache, crate::kv_cache::bundle::KvBundle, Vec<u32>) {
        let cfg = prefix_config();
        let mut cache_a = Cache::new(&cfg, DType::F32, &Device::Cpu).unwrap();
        let kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        cache_a.reset_sequence();
        cache_a.allocate_kv(32).unwrap();
        for layer in 0..cfg.num_hidden_layers {
            let n = kv_heads * 32 * head_dim;
            let kdata: Vec<f32> = (0..n).map(|i| (layer * 1000 + i) as f32 + 0.125).collect();
            let vdata: Vec<f32> = kdata.iter().map(|x| x + 0.5).collect();
            let k = candle_core::Tensor::from_vec(kdata, (1, kv_heads, 32, head_dim), &Device::Cpu)
                .unwrap();
            let v = candle_core::Tensor::from_vec(vdata, (1, kv_heads, 32, head_dim), &Device::Cpu)
                .unwrap();
            cache_a.write_kv(layer, &k, &v).unwrap();
        }
        let ids: Vec<u32> = (0..32u32).collect();
        let bundle = cache_a.export_prefix_bundle("m", &ids, 32).unwrap();
        (cache_a, bundle, ids)
    }

    #[test]
    fn import_prefix_bundle_round_trips_values_and_enables_reuse() {
        let cfg = prefix_config();
        let (cache_a, bundle, _ids) = cache_with_bundle();

        let mut cache_b = Cache::new(&cfg, DType::F32, &Device::Cpu).unwrap();
        let prompt: Vec<u32> = (0..40u32).collect(); // first 32 tokens == bundle ids
        let imported = cache_b.import_prefix_bundle(&bundle, &prompt).unwrap();
        assert_eq!(imported, 32);

        // Imported K/V matches cache A's live blocks, layer by layer, block by block.
        for layer in 0..cfg.num_hidden_layers {
            let a = cache_a.kv_blocks(layer).unwrap();
            let b = cache_b.kv_blocks(layer).unwrap();
            assert_eq!(a.len(), b.len());
            for i in 0..a.len() {
                let av: Vec<f32> = a[i].0.flatten_all().unwrap().to_vec1().unwrap();
                let bv: Vec<f32> = b[i].0.flatten_all().unwrap().to_vec1().unwrap();
                assert_eq!(av, bv, "k mismatch layer {layer} block {i}");
                let av2: Vec<f32> = a[i].1.flatten_all().unwrap().to_vec1().unwrap();
                let bv2: Vec<f32> = b[i].1.flatten_all().unwrap().to_vec1().unwrap();
                assert_eq!(av2, bv2, "v mismatch layer {layer} block {i}");
            }
        }

        // After reset, the registry lets the same prompt reuse the imported blocks.
        cache_b.reset_sequence();
        assert_eq!(cache_b.match_prefix(&prompt), 32);
    }

    #[test]
    fn import_prefix_bundle_rejects_mismatched_prompt_prefix() {
        let cfg = prefix_config();
        let (_a, bundle, _ids) = cache_with_bundle();
        let mut cache_b = Cache::new(&cfg, DType::F32, &Device::Cpu).unwrap();
        let wrong: Vec<u32> = (100..140u32).collect(); // does not start with bundle ids
        assert!(cache_b.import_prefix_bundle(&bundle, &wrong).is_err());
        // No partial state survives a rejected import.
        cache_b.reset_sequence();
        assert_eq!(cache_b.match_prefix(&wrong), 0);
    }

    #[test]
    fn import_prefix_bundle_rejects_dimension_mismatch() {
        let (_a, bundle, _ids) = cache_with_bundle(); // head_dim 4
        let cfg2 = Config {
            head_dim: 8,
            ..prefix_config()
        };
        let mut cache_b = Cache::new(&cfg2, DType::F32, &Device::Cpu).unwrap();
        let prompt: Vec<u32> = (0..40u32).collect();
        assert!(cache_b.import_prefix_bundle(&bundle, &prompt).is_err());
    }

    #[test]
    fn export_prefix_bundle_rejects_unaligned_or_oversized_count() {
        let cfg = prefix_config();
        let mut cache = Cache::new(&cfg, DType::F32, &Device::Cpu).unwrap();
        cache.reset_sequence();
        cache.allocate_kv(32).unwrap();
        write_zeros_for_current_batch(&mut cache, 32);
        // Not block aligned.
        assert!(cache.export_prefix_bundle("m", &[0; 32], 20).is_err());
        // Exceeds live length.
        assert!(cache.export_prefix_bundle("m", &[0; 48], 48).is_err());
    }

    #[test]
    fn clear_prefix_cache_disables_reuse() {
        let cfg = prefix_config();
        let mut cache = Cache::new(&cfg, DType::F32, &Device::Cpu).unwrap();
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

    #[test]
    fn clear_prefix_cache_releases_registry_only_blocks_to_the_pool() {
        let cfg = prefix_config();
        let mut cache = Cache::new(&cfg, DType::F32, &Device::Cpu).unwrap();
        let ids: Vec<u32> = (0..40).collect(); // 2 full blocks registered + partial tail

        cache.reset_sequence();
        cache.match_prefix(&ids);
        cache.allocate_kv(40).unwrap();
        write_zeros_for_current_batch(&mut cache, 40);
        cache.register_prefix(&ids); // blocks 0,1 held by registry (refcount 2)

        // End the sequence: registered blocks drop to refcount 1 (registry only).
        cache.reset_sequence();
        let free_before = cache.free_blocks_for_test();

        // Clearing the prefix cache must return the 2 registry-only blocks to the pool.
        cache.clear_prefix_cache();
        assert_eq!(cache.free_blocks_for_test(), free_before + 2);

        // Reuse is also disabled afterwards.
        cache.reset_sequence();
        assert_eq!(cache.match_prefix(&ids), 0);
    }
}
