use serde::Deserialize;

use crate::models::common::{Config, EosTokenId};

fn default_rope_theta() -> f32 {
    1_000_000.0
}

fn default_max_position_embeddings() -> usize {
    40960
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub head_dim: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub eos_token_id: Option<EosTokenId>,
    pub tie_word_embeddings: bool,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
}

impl Qwen3Config {
    pub fn into_config(self) -> Config {
        let num_key_value_heads = self.num_key_value_heads.unwrap_or(self.num_attention_heads);
        let head_dim = self
            .head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads);
        let config = Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            max_seq_len: self.max_position_embeddings,
            use_qkv_bias: false,
            use_qk_norm: true,
            tie_word_embeddings: self.tie_word_embeddings,
            eos_token_id: self.eos_token_id,
        };
        tracing::info!("Successfully loading qwen3 model: {:#?}", config);
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "vocab_size": 151936,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "eos_token_id": [151643, 151645],
        "tie_word_embeddings": true,
        "max_position_embeddings": 40960
    }"#;

    #[test]
    fn parses_sample_config_with_explicit_head_dim() {
        let raw: Qwen3Config = serde_json::from_str(SAMPLE).unwrap();
        let cfg = raw.into_config();
        assert_eq!(cfg.head_dim, 128); // explicit, not hidden_size/num_attention_heads (=64)
        assert!(cfg.use_qk_norm);
        assert!(!cfg.use_qkv_bias);
        assert!(matches!(
            cfg.eos_token_id,
            Some(crate::models::common::EosTokenId::Multiple(ref v)) if v == &vec![151643, 151645]
        ));
    }

    #[test]
    fn head_dim_falls_back_to_hidden_over_heads_when_absent() {
        let raw = r#"{
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 64, "intermediate_size": 128, "vocab_size": 256,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4,
            "rms_norm_eps": 1e-6, "tie_word_embeddings": true
        }"#;
        let raw: Qwen3Config = serde_json::from_str(raw).unwrap();
        assert_eq!(raw.into_config().head_dim, 16); // 64 / 4
    }
}
