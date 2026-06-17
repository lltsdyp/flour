use candle_core::{Result, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, RmsNorm, VarBuilder};

use crate::backend::cpu::{causal_attention, repeat_kv};

use super::{Cache, Config};

#[derive(Debug)]
pub struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
}

impl CausalSelfAttention {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let h = cfg.hidden_size;
        let size_q = cfg.head_dim * cfg.num_attention_heads;
        let size_kv = cfg.head_dim * cfg.num_key_value_heads;

        let (q_proj, k_proj, v_proj) = if cfg.use_qkv_bias {
            (
                candle_nn::linear(h, size_q, vb.pp("q_proj"))?,
                candle_nn::linear(h, size_kv, vb.pp("k_proj"))?,
                candle_nn::linear(h, size_kv, vb.pp("v_proj"))?,
            )
        } else {
            (
                linear_no_bias(h, size_q, vb.pp("q_proj"))?,
                linear_no_bias(h, size_kv, vb.pp("k_proj"))?,
                linear_no_bias(h, size_kv, vb.pp("v_proj"))?,
            )
        };
        let o_proj = linear_no_bias(size_q, h, vb.pp("o_proj"))?;

        let (q_norm, k_norm) = if cfg.use_qk_norm {
            let qw = vb.pp("q_norm").get(cfg.head_dim, "weight")?;
            let kw = vb.pp("k_norm").get(cfg.head_dim, "weight")?;
            (
                Some(RmsNorm::new(qw, cfg.rms_norm_eps)),
                Some(RmsNorm::new(kw, cfg.rms_norm_eps)),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        layer_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?;
        let k = k.reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?;
        let v = v.reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?;

        let q = match &self.q_norm {
            Some(n) => n.forward(&q.contiguous()?)?,
            None => q,
        };
        let k = match &self.k_norm {
            Some(n) => n.forward(&k.contiguous()?)?,
            None => k,
        };

        let q = q.transpose(1, 2)?.contiguous()?; // (b, heads, seq, head_dim)
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let cos = cache.rope_cos(index_pos, seq_len)?;
        let sin = cache.rope_sin(index_pos, seq_len)?;
        let q = candle_nn::rotary_emb::rope(&q, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k, &cos, &sin)?;

        let (k, v) = cache.append_kv(layer_idx, k, v)?;
        let kv_seq_len = k.dims()[2];

        let n_rep = self.num_attention_heads / self.num_key_value_heads;
        let k = repeat_kv(k, n_rep)?;
        let v = repeat_kv(v, n_rep)?;

        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let y = if seq_len > 1 || kv_seq_len > 1 {
            let mask = cache.causal_mask(seq_len, kv_seq_len)?;
            causal_attention(&q, &k, &v, Some(&mask), scale)?
        } else {
            causal_attention(&q, &k, &v, None, scale)?
        };

        let y = y.transpose(1, 2)?.reshape((b_sz, seq_len, self.num_attention_heads * self.head_dim))?;
        self.o_proj.forward(&y)
    }
}

#[cfg(test)]
mod tests {
    use super::CausalSelfAttention;
    use crate::models::common::Config;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    fn test_config(use_qkv_bias: bool, use_qk_norm: bool) -> Config {
        Config {
            hidden_size: 8,
            intermediate_size: 16,
            vocab_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1, // GQA: 2 query heads share 1 kv head
            head_dim: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            max_seq_len: 16,
            use_qkv_bias,
            use_qk_norm,
            tie_word_embeddings: false,
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
        let size_q = cfg.head_dim * cfg.num_attention_heads;
        let size_kv = cfg.head_dim * cfg.num_key_value_heads;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("q_proj.weight".into(), make_tensor(&[size_q, h], 1));
        map.insert("k_proj.weight".into(), make_tensor(&[size_kv, h], 2));
        map.insert("v_proj.weight".into(), make_tensor(&[size_kv, h], 3));
        map.insert("o_proj.weight".into(), make_tensor(&[h, size_q], 4));
        if cfg.use_qkv_bias {
            map.insert("q_proj.bias".into(), make_tensor(&[size_q], 5));
            map.insert("k_proj.bias".into(), make_tensor(&[size_kv], 6));
            map.insert("v_proj.bias".into(), make_tensor(&[size_kv], 7));
        }
        if cfg.use_qk_norm {
            map.insert("q_norm.weight".into(), Tensor::ones(cfg.head_dim, DType::F32, &Device::Cpu).unwrap());
            map.insert("k_norm.weight".into(), Tensor::ones(cfg.head_dim, DType::F32, &Device::Cpu).unwrap());
        }
        VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
    }

    #[test]
    fn forward_preserves_shape_prefill() {
        let cfg = test_config(false, false);
        let attn = CausalSelfAttention::load(make_vb(&cfg), &cfg).unwrap();
        let mut cache = super::super::Cache::new(&cfg, &Device::Cpu).unwrap();
        let x = make_tensor(&[1, 5, cfg.hidden_size], 100);
        let y = attn.forward(&x, 0, 0, &mut cache).unwrap();
        assert_eq!(y.dims(), &[1, 5, cfg.hidden_size]);
    }

    #[test]
    fn forward_works_with_qkv_bias_and_qk_norm() {
        let cfg = test_config(true, true);
        let attn = CausalSelfAttention::load(make_vb(&cfg), &cfg).unwrap();
        let mut cache = super::super::Cache::new(&cfg, &Device::Cpu).unwrap();
        let x = make_tensor(&[1, 5, cfg.hidden_size], 101);
        let y = attn.forward(&x, 0, 0, &mut cache).unwrap();
        assert_eq!(y.dims(), &[1, 5, cfg.hidden_size]);
    }

    #[test]
    fn forward_single_token_decode_step_after_prefill() {
        let cfg = test_config(false, false);
        let attn = CausalSelfAttention::load(make_vb(&cfg), &cfg).unwrap();
        let mut cache = super::super::Cache::new(&cfg, &Device::Cpu).unwrap();
        let prefill = make_tensor(&[1, 3, cfg.hidden_size], 102);
        attn.forward(&prefill, 0, 0, &mut cache).unwrap();
        let step = make_tensor(&[1, 1, cfg.hidden_size], 103);
        let y = attn.forward(&step, 3, 0, &mut cache).unwrap();
        assert_eq!(y.dims(), &[1, 1, cfg.hidden_size]);
    }
}
