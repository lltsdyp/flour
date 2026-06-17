use candle_core::{Result, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

use super::Config;

#[derive(Debug)]
pub struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MLP {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        Ok(Self {
            gate_proj: linear_no_bias(h, i, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(h, i, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(i, h, vb.pp("down_proj"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
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
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("gate_proj.weight".into(), make_tensor(&[i, h], 10));
        map.insert("up_proj.weight".into(), make_tensor(&[i, h], 11));
        map.insert("down_proj.weight".into(), make_tensor(&[h, i], 12));
        VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
    }

    #[test]
    fn forward_preserves_hidden_size() {
        let cfg = test_config();
        let mlp = MLP::load(make_vb(&cfg), &cfg).unwrap();
        let x = make_tensor(&[1, 3, cfg.hidden_size], 20);
        let y = mlp.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 3, cfg.hidden_size]);
    }
}
