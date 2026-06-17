#[derive(Debug, Clone, serde::Deserialize)]
#[serde(untagged)]
pub enum EosTokenId {
    Single(u32),
    Multiple(Vec<u32>),
}

impl EosTokenId {
    pub fn is_eos(&self, token_id: u32) -> bool {
        match self {
            EosTokenId::Single(id) => *id == token_id,
            EosTokenId::Multiple(ids) => ids.contains(&token_id),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_seq_len: usize,
    pub use_qkv_bias: bool,
    pub use_qk_norm: bool,
    pub tie_word_embeddings: bool,
    pub eos_token_id: Option<EosTokenId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eos_single_matches_only_that_id() {
        let e = EosTokenId::Single(2);
        assert!(e.is_eos(2));
        assert!(!e.is_eos(3));
    }

    #[test]
    fn eos_multiple_matches_any_listed_id() {
        let e = EosTokenId::Multiple(vec![1, 2, 3]);
        assert!(e.is_eos(2));
        assert!(!e.is_eos(4));
    }

    #[test]
    fn eos_token_id_deserializes_from_number_or_array() {
        let single: EosTokenId = serde_json::from_str("2").unwrap();
        assert!(matches!(single, EosTokenId::Single(2)));
        let multi: EosTokenId = serde_json::from_str("[1,2,3]").unwrap();
        assert!(matches!(multi, EosTokenId::Multiple(ref v) if v == &vec![1,2,3]));
    }
}
