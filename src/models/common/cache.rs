use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};

use super::paged::{BlockAllocator, BlockTable, PagedKvPool};
use super::Config;

const BLOCK_SIZE: usize = 16;

#[derive(Debug)]
pub struct KvCache{
    pool: PagedKvPool,
    allocator: BlockAllocator,
    table: BlockTable,
}

impl KvCache{
    pub fn new(cfg: &Config, device: &Device) -> Result<Self> {
        let num_blocks = cfg.max_seq_len.div_ceil(BLOCK_SIZE);
        let num_slots = num_blocks * BLOCK_SIZE;
        let allocator = BlockAllocator::new(num_blocks);
        let table = BlockTable::new(BLOCK_SIZE);
        let pool = PagedKvPool::new(
            cfg.num_hidden_layers,
            num_slots,
            cfg.num_key_value_heads,
            cfg.head_dim,
            device,
        )?;

        Ok(Self {
            pool,
            allocator,
            table
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

    pub fn capacity(&self) -> usize{
        self.table.capacity()
    }

    pub fn slots(&self,start: usize,end: usize) -> Vec<u32>{
        self.table.slots(start, end)
    }

    pub fn write(&mut self, layer_idx: usize, slots: &[u32], k: &Tensor, v: &Tensor) -> Result<()>{
        self.pool.write(layer_idx, slots, k, v)
    }

    pub fn gather(&self, layer_idx: usize, slots: &[u32]) -> Result<(Tensor, Tensor)> {
        self.pool.gather(layer_idx, slots)
    }
    pub fn advance(&mut self, seq_len: usize) {
        self.table.advance(seq_len);
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
    pub fn new(cfg: &Config, device: &Device) -> Result<Self> {
        let half_dim = cfg.head_dim / 2;
        let theta: Vec<f32> = (0..half_dim)
            .map(|i| 1f32 / cfg.rope_theta.powf((2 * i) as f32 / cfg.head_dim as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;

        let idx_theta = Tensor::arange(0, cfg.max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_seq_len, 1))?
            .matmul(&theta.reshape((1, half_dim))?)?;

        let cos = idx_theta.cos()?.contiguous()?;
        let sin = idx_theta.sin()?.contiguous()?;

        Ok(Self {
            cos,
            sin,
            masks: HashMap::new(),
            kvs: KvCache::new(cfg, device)?,
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
        if !self.masks.contains_key(&key) {
            let offset = kv_seq_len - seq_len;
            let mask: Vec<u8> = (0..seq_len)
                .flat_map(|i| (0..kv_seq_len).map(move |j| u8::from(j > i + offset)))
                .collect();
            let mask = Tensor::from_slice(&mask, (seq_len, kv_seq_len), &self.device)?;
            self.masks.insert(key, mask);
        }
        Ok(self.masks.get(&key).unwrap().clone())
    }

    pub fn allocate_kv(&mut self,seq_len: usize) -> Result<()>{
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

    pub fn append_kv(
        &mut self,
        layer_idx: usize,
        k: Tensor,
        v: Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Blocks are reserved and the logical length advanced once per batch by
        // `allocate_kv` (called from the model's forward, before any layer runs). Here `len`
        // already includes this batch's tokens, so the new window is the trailing `seq_len`.
        let seq_len = k.dims()[2];
        let end = self.kvs.len();
        let start = end - seq_len;
        let write_slots = self.kvs.slots(start, end);
        let all_slots = self.kvs.slots(0, end);

        self.kvs.write(layer_idx, &write_slots, &k, &v)?;
        self.kvs.gather(layer_idx, &all_slots)
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
        let cache = Cache::new(&test_config(), &Device::Cpu).unwrap();
        let cos = cache.rope_cos(0, 5).unwrap();
        assert_eq!(cos.dims(), &[5, 2]); // head_dim/2 = 2
    }

    #[test]
    fn rope_tables_respect_index_pos_offset() {
        let cache = Cache::new(&test_config(), &Device::Cpu).unwrap();
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
        let mut cache = Cache::new(&test_config(), &Device::Cpu).unwrap();
        let mask = cache.causal_mask(3, 3).unwrap();
        let mask: Vec<u8> = mask.flatten_all().unwrap().to_vec1().unwrap();
        // row 0: only col 0 visible -> [0,1,1]; row 1: [0,0,1]; row 2: [0,0,0]
        assert_eq!(mask, vec![0, 1, 1, 0, 0, 1, 0, 0, 0]);
    }

    #[test]
    fn causal_mask_pads_for_existing_kv_prefix() {
        let mut cache = Cache::new(&test_config(), &Device::Cpu).unwrap();
        // 1 new query position, 3 total kv positions (2 already cached) -> fully visible
        let mask = cache.causal_mask(1, 3).unwrap();
        let mask: Vec<u8> = mask.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(mask, vec![0, 0, 0]);
    }

    #[test]
    fn append_kv_concatenates_along_seq_dim() {
        let mut cache = Cache::new(&test_config(), &Device::Cpu).unwrap();
        cache.allocate_kv(2).unwrap();
        let k1 = candle_core::Tensor::zeros((1, 2, 2, 4), candle_core::DType::F32, &Device::Cpu)
            .unwrap();
        let v1 = k1.clone();
        let (k, _v) = cache.append_kv(0, k1, v1).unwrap();
        assert_eq!(k.dims(), &[1, 2, 2, 4]);

        cache.allocate_kv(1).unwrap();
        let k2 = candle_core::Tensor::zeros((1, 2, 1, 4), candle_core::DType::F32, &Device::Cpu)
            .unwrap();
        let v2 = k2.clone();
        let (k, _v) = cache.append_kv(0, k2, v2).unwrap();
        assert_eq!(k.dims(), &[1, 2, 3, 4]); // 2 + 1 = 3
    }

    #[test]
    fn append_kv_overflow_past_max_seq_len_errors() {
        let mut cache = Cache::new(&test_config(), &Device::Cpu).unwrap();
        // max_seq_len = 16; reserving 17 tokens in one shot must error.
        assert!(cache.allocate_kv(17).is_err());
    }

    #[test]
    fn append_kv_across_two_layers_shares_positions() {
        let mut cache = Cache::new(&test_config(), &Device::Cpu).unwrap();
        let k = candle_core::Tensor::zeros((1, 2, 3, 4), candle_core::DType::F32, &Device::Cpu)
            .unwrap();
        let v = k.clone();
        // allocate_kv advances once for the batch; both layers reuse the same slots.
        cache.allocate_kv(3).unwrap();
        let (k0, _) = cache.append_kv(0, k.clone(), v.clone()).unwrap();
        let (k1, _) = cache.append_kv(1, k, v).unwrap();
        assert_eq!(k0.dims(), &[1, 2, 3, 4]);
        assert_eq!(k1.dims(), &[1, 2, 3, 4]);
    }
}
