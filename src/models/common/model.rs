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

        Ok(Self { embed_tokens, layers, norm, lm_head, cfg })
    }

    pub fn forward(&self, input_ids: &Tensor, index_pos: usize, cache: &mut Cache) -> Result<Tensor> {
        let mut x = self.embed_tokens.forward(input_ids)?;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, index_pos, i, cache)?;
        }
        let x = self.norm.forward(&x)?;
        self.lm_head.forward(&x)
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::common::Config;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

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
            tie_word_embeddings: true,
            eos_token_id: None,
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

    fn make_vb(cfg: &Config) -> VarBuilder<'static> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let size_q = cfg.head_dim * cfg.num_attention_heads;
        let size_kv = cfg.head_dim * cfg.num_key_value_heads;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("model.embed_tokens.weight".into(), make_tensor(&[cfg.vocab_size, h], 1));
        map.insert("model.norm.weight".into(), Tensor::ones(h, DType::F32, &Device::Cpu).unwrap());
        for layer in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{layer}");
            map.insert(format!("{p}.input_layernorm.weight"), Tensor::ones(h, DType::F32, &Device::Cpu).unwrap());
            map.insert(format!("{p}.post_attention_layernorm.weight"), Tensor::ones(h, DType::F32, &Device::Cpu).unwrap());
            map.insert(format!("{p}.self_attn.q_proj.weight"), make_tensor(&[size_q, h], 10 + layer as u64));
            map.insert(format!("{p}.self_attn.k_proj.weight"), make_tensor(&[size_kv, h], 20 + layer as u64));
            map.insert(format!("{p}.self_attn.v_proj.weight"), make_tensor(&[size_kv, h], 30 + layer as u64));
            map.insert(format!("{p}.self_attn.o_proj.weight"), make_tensor(&[h, size_q], 40 + layer as u64));
            map.insert(format!("{p}.mlp.gate_proj.weight"), make_tensor(&[i, h], 50 + layer as u64));
            map.insert(format!("{p}.mlp.up_proj.weight"), make_tensor(&[i, h], 60 + layer as u64));
            map.insert(format!("{p}.mlp.down_proj.weight"), make_tensor(&[h, i], 70 + layer as u64));
        }
        VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
    }

    #[test]
    fn forward_prefill_returns_logits_for_every_position() {
        let cfg = test_config();
        let model = CausalLM::load(make_vb(&cfg), cfg.clone()).unwrap();
        let mut cache = super::super::Cache::new(&cfg, &Device::Cpu).unwrap();
        let ids = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &Device::Cpu).unwrap();
        let logits = model.forward(&ids, 0, &mut cache).unwrap();
        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
    }

    #[test]
    fn forward_decode_step_returns_logits_for_one_position() {
        let cfg = test_config();
        let model = CausalLM::load(make_vb(&cfg), cfg.clone()).unwrap();
        let mut cache = super::super::Cache::new(&cfg, &Device::Cpu).unwrap();
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
}
