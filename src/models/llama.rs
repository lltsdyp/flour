use serde::Deserialize;

use crate::models::common::{Config, EosTokenId};

fn default_rope_theta() -> f32 {
    500_000.0
}

fn default_max_position_embeddings() -> usize {
    4096
}

#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub eos_token_id: Option<EosTokenId>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
}

impl LlamaConfig {
    pub fn into_config(self) -> Config {
        let num_key_value_heads = self.num_key_value_heads.unwrap_or(self.num_attention_heads);
        let head_dim = self.hidden_size / self.num_attention_heads;
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
            use_qk_norm: false,
            tie_word_embeddings: self.tie_word_embeddings,
            eos_token_id: self.eos_token_id,
        };
        tracing::info!("Successfully loading llama model: {:#?}",config);
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "vocab_size": 128256,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "eos_token_id": 128001,
        "tie_word_embeddings": false,
        "max_position_embeddings": 8192
    }"#;

    #[test]
    fn parses_sample_config_into_shared_config() {
        let raw: LlamaConfig = serde_json::from_str(SAMPLE).unwrap();
        let cfg = raw.into_config();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 2048 / 32);
        assert!(!cfg.use_qkv_bias);
        assert!(!cfg.use_qk_norm);
        assert!(!cfg.tie_word_embeddings);
        assert_eq!(cfg.max_seq_len, 8192);
        assert!(matches!(
            cfg.eos_token_id,
            Some(crate::models::common::EosTokenId::Single(128001))
        ));
    }

    #[test]
    fn defaults_apply_when_optional_fields_missing() {
        let minimal = r#"{
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 64,
            "intermediate_size": 128,
            "vocab_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "rms_norm_eps": 1e-5
        }"#;
        let raw: LlamaConfig = serde_json::from_str(minimal).unwrap();
        let cfg = raw.into_config();
        assert_eq!(cfg.num_key_value_heads, 4); // defaults to num_attention_heads
        assert_eq!(cfg.rope_theta, 500_000.0);
        assert_eq!(cfg.max_seq_len, 4096);
        assert!(!cfg.tie_word_embeddings);
    }
}
