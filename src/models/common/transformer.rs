use candle_core::{Result, Tensor};
use candle_nn::{Module, RmsNorm, VarBuilder};

use super::{Cache, CausalSelfAttention, Config, MLP};

#[derive(Debug)]
pub struct DecoderLayer {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    self_attn: CausalSelfAttention,
    mlp: MLP,
}

impl DecoderLayer {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let input_w = vb.pp("input_layernorm").get(cfg.hidden_size, "weight")?;
        let post_w = vb
            .pp("post_attention_layernorm")
            .get(cfg.hidden_size, "weight")?;
        Ok(Self {
            input_layernorm: RmsNorm::new(input_w, cfg.rms_norm_eps),
            post_attention_layernorm: RmsNorm::new(post_w, cfg.rms_norm_eps),
            self_attn: CausalSelfAttention::load(vb.pp("self_attn"), cfg)?,
            mlp: MLP::load(vb.pp("mlp"), cfg)?,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        layer_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let residual = x;
        let h = self.input_layernorm.forward(x)?;
        let h = self.self_attn.forward(&h, index_pos, layer_idx, cache)?;
        let h = (residual + h)?;

        let residual = &h;
        let y = self.post_attention_layernorm.forward(&h)?;
        let y = self.mlp.forward(&y)?;
        residual + y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::common::{Cache, Config};
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    fn test_config() -> Config {
        Config {
            hidden_size: 8,
            intermediate_size: 16,
            vocab_size: 32,
            num_hidden_layers: 1,
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
        map.insert(
            "input_layernorm.weight".into(),
            Tensor::ones(h, DType::F32, &Device::Cpu).unwrap(),
        );
        map.insert(
            "post_attention_layernorm.weight".into(),
            Tensor::ones(h, DType::F32, &Device::Cpu).unwrap(),
        );
        map.insert(
            "self_attn.q_proj.weight".into(),
            make_tensor(&[size_q, h], 30),
        );
        map.insert(
            "self_attn.k_proj.weight".into(),
            make_tensor(&[size_kv, h], 31),
        );
        map.insert(
            "self_attn.v_proj.weight".into(),
            make_tensor(&[size_kv, h], 32),
        );
        map.insert(
            "self_attn.o_proj.weight".into(),
            make_tensor(&[h, size_q], 33),
        );
        map.insert("mlp.gate_proj.weight".into(), make_tensor(&[i, h], 40));
        map.insert("mlp.up_proj.weight".into(), make_tensor(&[i, h], 41));
        map.insert("mlp.down_proj.weight".into(), make_tensor(&[h, i], 42));
        VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
    }

    #[test]
    fn forward_preserves_shape() {
        let cfg = test_config();
        let layer = DecoderLayer::load(make_vb(&cfg), &cfg).unwrap();
        let mut cache = Cache::new(&cfg, &Device::Cpu).unwrap();
        let x = make_tensor(&[1, 4, cfg.hidden_size], 50);
        cache.allocate_kv(4).unwrap();
        let y = layer.forward(&x, 0, 0, &mut cache).unwrap();
        assert_eq!(y.dims(), &[1, 4, cfg.hidden_size]);
    }
}
