use candle_core::{Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, Module, RmsNorm, VarBuilder};

use super::{Cache, Config, DecoderLayer};

#[derive(Debug)]
pub struct CausalLM {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    cfg: Config,
}

impl CausalLM {
    pub fn load(vb: VarBuilder, cfg: Config) -> Result<Self> {
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb.pp("model.layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::load(vb_layers.pp(i), &cfg)?);
        }

        let norm_w = vb.pp("model.norm").get(cfg.hidden_size, "weight")?;
        let norm = RmsNorm::new(norm_w, cfg.rms_norm_eps);

        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            cfg,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let mut x = self.embed_tokens.forward(input_ids)?;
        // Reserve KV slots for this batch's tokens once, before any layer writes. All layers
        // share these logical positions, so allocation/advance happens here, not per layer.
        let (_b_sz, seq_len) = input_ids.dims2()?;
        cache.allocate_kv(seq_len)?;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, index_pos, i, cache)?;
        }
        let x = self.norm.forward(&x)?;
        self.lm_head.forward(&x)
    }

    /// Run the network over only the unmatched suffix `input_ids[reused_prefix_tokens..]`, given
    /// that the first `reused_prefix_tokens` tokens already sit in `cache` (from a local match or
    /// a remote import). Does NOT reset the sequence, do local prefix matching, or register the
    /// prefix — the caller owns that lifecycle. `forward` allocates KV slots for the suffix,
    /// writes its K/V into fresh blocks, and runs paged attention over ALL live blocks (reused
    /// prefix + suffix). RoPE uses `index_pos = reused_prefix_tokens`, so the suffix sits at its
    /// true absolute positions. Returns the suffix logits (final row = last prompt token).
    pub fn prefill_suffix(
        &self,
        input_ids: &Tensor,
        reused_prefix_tokens: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let ids: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;
        let prompt_len = ids.len();
        if reused_prefix_tokens >= prompt_len {
            candle_core::bail!(
                "reused_prefix_tokens {reused_prefix_tokens} must be < prompt_len {prompt_len}; \
                 cached KV cannot produce the last prompt token's logits"
            );
        }
        let suffix = &ids[reused_prefix_tokens..];
        let suffix_ids = Tensor::from_vec(suffix.to_vec(), (1, suffix.len()), input_ids.device())?;
        self.forward(&suffix_ids, reused_prefix_tokens, cache)
    }

    /// Prefill that reuses any cached prefix. Matched leading blocks are read straight from the
    /// KV pool (no recompute); only the unmatched suffix runs through the network. Returns the
    /// suffix logits (final row = last prompt token) and how many prompt tokens were reused.
    pub fn prefill_cached(&self, input_ids: &Tensor, cache: &mut Cache) -> Result<(Tensor, usize)> {
        let ids: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;

        cache.reset_sequence();
        let matched = cache.match_prefix(&ids);
        let logits = self.prefill_suffix(input_ids, matched, cache)?;
        cache.register_prefix(&ids);
        Ok((logits, matched))
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }
}

/// Tiny, deterministic model/config builders shared by unit tests across the crate (e.g. the
/// kv_cache manager's remote-import test needs a real `CausalLM`). Test-only.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    pub(crate) fn test_config() -> Config {
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
            tie_word_embeddings: true,
            eos_token_id: None,
        }
    }

    /// Like [`test_config`] but with a 64-token window (4 blocks), for prefix-reuse tests.
    pub(crate) fn prefix_test_config() -> Config {
        Config {
            max_seq_len: 64,
            ..test_config()
        }
    }

    fn make_tensor(shape: &[usize], seed: u64) -> Tensor {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let numel: usize = shape.iter().product();
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<f32> = (0..numel).map(|_| rng.gen_range(-0.1..0.1)).collect();
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }

    pub(crate) fn make_vb(cfg: &Config) -> VarBuilder<'static> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let size_q = cfg.head_dim * cfg.num_attention_heads;
        let size_kv = cfg.head_dim * cfg.num_key_value_heads;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert(
            "model.embed_tokens.weight".into(),
            make_tensor(&[cfg.vocab_size, h], 1),
        );
        map.insert(
            "model.norm.weight".into(),
            Tensor::ones(h, DType::F32, &Device::Cpu).unwrap(),
        );
        for layer in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{layer}");
            map.insert(
                format!("{p}.input_layernorm.weight"),
                Tensor::ones(h, DType::F32, &Device::Cpu).unwrap(),
            );
            map.insert(
                format!("{p}.post_attention_layernorm.weight"),
                Tensor::ones(h, DType::F32, &Device::Cpu).unwrap(),
            );
            map.insert(
                format!("{p}.self_attn.q_proj.weight"),
                make_tensor(&[size_q, h], 10 + layer as u64),
            );
            map.insert(
                format!("{p}.self_attn.k_proj.weight"),
                make_tensor(&[size_kv, h], 20 + layer as u64),
            );
            map.insert(
                format!("{p}.self_attn.v_proj.weight"),
                make_tensor(&[size_kv, h], 30 + layer as u64),
            );
            map.insert(
                format!("{p}.self_attn.o_proj.weight"),
                make_tensor(&[h, size_q], 40 + layer as u64),
            );
            map.insert(
                format!("{p}.mlp.gate_proj.weight"),
                make_tensor(&[i, h], 50 + layer as u64),
            );
            map.insert(
                format!("{p}.mlp.up_proj.weight"),
                make_tensor(&[i, h], 60 + layer as u64),
            );
            map.insert(
                format!("{p}.mlp.down_proj.weight"),
                make_tensor(&[h, i], 70 + layer as u64),
            );
        }
        VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{make_vb, prefix_test_config, test_config};
    use super::*;
    use candle_core::{DType, Device, IndexOp, Tensor};

    #[test]
    fn forward_prefill_returns_logits_for_every_position() {
        let cfg = test_config();
        let model = CausalLM::load(make_vb(&cfg), cfg.clone()).unwrap();
        let mut cache = super::super::Cache::new(&cfg, DType::F32, &Device::Cpu).unwrap();
        let ids = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &Device::Cpu).unwrap();
        let logits = model.forward(&ids, 0, &mut cache).unwrap();
        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
    }

    #[test]
    fn forward_decode_step_returns_logits_for_one_position() {
        let cfg = test_config();
        let model = CausalLM::load(make_vb(&cfg), cfg.clone()).unwrap();
        let mut cache = super::super::Cache::new(&cfg, DType::F32, &Device::Cpu).unwrap();
        let prefill = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &Device::Cpu).unwrap();
        model.forward(&prefill, 0, &mut cache).unwrap();
        let step = Tensor::from_vec(vec![4u32], (1, 1), &Device::Cpu).unwrap();
        let logits = model.forward(&step, 3, &mut cache).unwrap();
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }

    #[test]
    fn tied_embeddings_use_embed_tokens_as_lm_head() {
        let cfg = test_config();
        let model = CausalLM::load(make_vb(&cfg), cfg.clone()).unwrap();
        // No "lm_head.weight" tensor was provided, so loading only succeeds if
        // tie_word_embeddings correctly reused embed_tokens instead of fetching lm_head.
        assert_eq!(model.config().vocab_size, cfg.vocab_size);
    }

    #[test]
    fn prefill_cached_with_cold_cache_matches_plain_forward() {
        let cfg = prefix_test_config();
        let model = CausalLM::load(make_vb(&cfg), cfg.clone()).unwrap();

        // Plain forward over the whole prompt (reference).
        let ids_vec: Vec<u32> = (0..20u32).map(|i| i % cfg.vocab_size as u32).collect();
        let ids = Tensor::from_vec(ids_vec.clone(), (1, ids_vec.len()), &Device::Cpu).unwrap();
        let mut ref_cache = super::super::Cache::new(&cfg, DType::F32, &Device::Cpu).unwrap();
        let reference = model.forward(&ids, 0, &mut ref_cache).unwrap();
        let ref_last: Vec<f32> = reference
            .i((0, ids_vec.len() - 1))
            .unwrap()
            .to_vec1()
            .unwrap();

        // Cold prefill_cached: nothing cached yet => no reuse, identical last-row logits.
        let mut cache = super::super::Cache::new(&cfg, DType::F32, &Device::Cpu).unwrap();
        let (logits, reused) = model.prefill_cached(&ids, &mut cache).unwrap();
        assert_eq!(reused, 0);
        let got_last: Vec<f32> = logits
            .i((0, logits.dim(1).unwrap() - 1))
            .unwrap()
            .to_vec1()
            .unwrap();
        for (a, b) in ref_last.iter().zip(got_last.iter()) {
            assert!((a - b).abs() < 1e-4, "logit mismatch {a} vs {b}");
        }
    }

    #[test]
    fn prefill_suffix_consumes_imported_prefix_and_matches_cold_logits() {
        let cfg = prefix_test_config();
        let model = CausalLM::load(make_vb(&cfg), cfg.clone()).unwrap();

        let ids_vec: Vec<u32> = (0..40u32).map(|i| i % cfg.vocab_size as u32).collect();
        let ids = Tensor::from_vec(ids_vec.clone(), (1, 40), &Device::Cpu).unwrap();

        // Cold reference: full prefill over all 40 tokens.
        let mut ref_cache = super::super::Cache::new(&cfg, DType::F32, &Device::Cpu).unwrap();
        let (ref_logits, reused) = model.prefill_cached(&ids, &mut ref_cache).unwrap();
        assert_eq!(reused, 0);
        let ref_last: Vec<f32> = ref_logits
            .i((0, ref_logits.dim(1).unwrap() - 1))
            .unwrap()
            .to_vec1()
            .unwrap();

        // Cache A: full prefill, then export the first 32 tokens (2 blocks) as a bundle.
        let mut cache_a = super::super::Cache::new(&cfg, DType::F32, &Device::Cpu).unwrap();
        model.prefill_cached(&ids, &mut cache_a).unwrap();
        let bundle = cache_a.export_prefix_bundle("m", &ids_vec, 32).unwrap();

        // Cache B: import the prefix, then run suffix-only prefill over the remaining 8 tokens.
        let mut cache_b = super::super::Cache::new(&cfg, DType::F32, &Device::Cpu).unwrap();
        let imported = cache_b.import_prefix_bundle(&bundle, &ids_vec).unwrap();
        assert_eq!(imported, 32);

        let logits = model.prefill_suffix(&ids, imported, &mut cache_b).unwrap();
        assert_eq!(logits.dim(1).unwrap(), 40 - 32);
        let got_last: Vec<f32> = logits
            .i((0, logits.dim(1).unwrap() - 1))
            .unwrap()
            .to_vec1()
            .unwrap();
        for (a, b) in ref_last.iter().zip(got_last.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "suffix prefill over imported prefix changed logits: {a} vs {b}"
            );
        }
    }

    #[test]
    fn prefill_suffix_rejects_reuse_covering_whole_prompt() {
        let cfg = prefix_test_config();
        let model = CausalLM::load(make_vb(&cfg), cfg.clone()).unwrap();
        let ids = Tensor::from_vec((0..16u32).collect::<Vec<_>>(), (1, 16), &Device::Cpu).unwrap();
        let mut cache = super::super::Cache::new(&cfg, DType::F32, &Device::Cpu).unwrap();
        // reused == prompt_len leaves no suffix to produce last-token logits -> error.
        assert!(model.prefill_suffix(&ids, 16, &mut cache).is_err());
    }

    #[test]
    fn prefill_cached_reuses_prefix_and_preserves_last_logits() {
        let cfg = prefix_test_config();
        let model = CausalLM::load(make_vb(&cfg), cfg.clone()).unwrap();

        let ids_vec: Vec<u32> = (0..40u32).map(|i| i % cfg.vocab_size as u32).collect();
        let ids = Tensor::from_vec(ids_vec.clone(), (1, ids_vec.len()), &Device::Cpu).unwrap();

        // Reference: last-row logits from a cold cache.
        let mut ref_cache = super::super::Cache::new(&cfg, DType::F32, &Device::Cpu).unwrap();
        let (ref_logits, _) = model.prefill_cached(&ids, &mut ref_cache).unwrap();
        let ref_last: Vec<f32> = ref_logits
            .i((0, ref_logits.dim(1).unwrap() - 1))
            .unwrap()
            .to_vec1()
            .unwrap();

        // Same cache, run the identical prompt again: prefix now reused, suffix shorter.
        let (logits, reused) = model.prefill_cached(&ids, &mut ref_cache).unwrap();
        assert!(
            reused > 0,
            "expected prefix reuse on the second identical prompt"
        );
        assert_eq!(logits.dim(1).unwrap(), ids_vec.len() - reused);

        let got_last: Vec<f32> = logits
            .i((0, logits.dim(1).unwrap() - 1))
            .unwrap()
            .to_vec1()
            .unwrap();
        for (a, b) in ref_last.iter().zip(got_last.iter()) {
            assert!((a - b).abs() < 1e-4, "reuse changed logits: {a} vs {b}");
        }
    }
}
