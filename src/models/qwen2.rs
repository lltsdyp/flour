use serde::Deserialize;

use crate::models::common::{Config, EosTokenId};

fn default_rope_theta() -> f32 {
    1_000_000.0
}

fn default_true() -> bool {
    true
}

fn default_max_position_embeddings() -> usize {
    32768
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen2Config {
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
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
}

impl Qwen2Config {
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
            use_qkv_bias: true,
            use_qk_norm: false,
            tie_word_embeddings: self.tie_word_embeddings,
            eos_token_id: self.eos_token_id,
        };
        tracing::info!("Successfully loading qwen2 model: {:#?}",config);
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 896,
        "intermediate_size": 4864,
        "vocab_size": 151936,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "eos_token_id": 151643,
        "max_position_embeddings": 32768
    }"#;

    #[test]
    fn parses_sample_config_and_sets_qkv_bias() {
        let raw: Qwen2Config = serde_json::from_str(SAMPLE).unwrap();
        let cfg = raw.into_config();
        assert_eq!(cfg.hidden_size, 896);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert!(cfg.use_qkv_bias);
        assert!(!cfg.use_qk_norm);
        assert!(cfg.tie_word_embeddings); // default true, not present in SAMPLE
        assert_eq!(cfg.max_seq_len, 32768);
    }

    #[test]
    fn tie_word_embeddings_can_be_overridden_false() {
        let raw = r#"{
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 64, "intermediate_size": 128, "vocab_size": 256,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "rms_norm_eps": 1e-6, "tie_word_embeddings": false
        }"#;
        let raw: Qwen2Config = serde_json::from_str(raw).unwrap();
        assert!(!raw.into_config().tie_word_embeddings);
    }
}
