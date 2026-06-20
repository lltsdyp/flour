pub mod common;
pub mod llama;
pub mod qwen2;
pub mod qwen3;

use anyhow::{anyhow, bail, Context};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Llama,
    Qwen2,
    Qwen3,
}

pub fn detect_family(raw: &serde_json::Value) -> anyhow::Result<ModelFamily> {
    let arch = raw
        .get("architectures")
        .and_then(|a| a.as_array())
        .and_then(|a| a.first())
        .and_then(|a| a.as_str())
        .ok_or_else(|| anyhow!("config.json missing 'architectures' field"))?;

    tracing::info!("Architecture field: {}", arch);

    if arch.starts_with("Llama") {
        Ok(ModelFamily::Llama)
    } else if arch.starts_with("Qwen3") {
        Ok(ModelFamily::Qwen3)
    } else if arch.starts_with("Qwen2") {
        Ok(ModelFamily::Qwen2)
    } else {
        bail!("unsupported model architecture: {arch} (supported: Llama*, Qwen2*, Qwen3*)")
    }
}

pub fn load_config(family: ModelFamily, raw: &serde_json::Value) -> anyhow::Result<common::Config> {
    match family {
        ModelFamily::Llama => Ok(serde_json::from_value::<llama::LlamaConfig>(raw.clone())
            .context("parsing Llama config.json")?
            .into_config()),
        ModelFamily::Qwen2 => Ok(serde_json::from_value::<qwen2::Qwen2Config>(raw.clone())
            .context("parsing Qwen2 config.json")?
            .into_config()),
        ModelFamily::Qwen3 => Ok(serde_json::from_value::<qwen3::Qwen3Config>(raw.clone())
            .context("parsing Qwen3 config.json")?
            .into_config()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_llama_from_architectures() {
        let raw: serde_json::Value =
            serde_json::from_str(r#"{"architectures":["LlamaForCausalLM"]}"#).unwrap();
        assert_eq!(detect_family(&raw).unwrap(), ModelFamily::Llama);
    }

    #[test]
    fn detects_qwen2_from_architectures() {
        let raw: serde_json::Value =
            serde_json::from_str(r#"{"architectures":["Qwen2ForCausalLM"]}"#).unwrap();
        assert_eq!(detect_family(&raw).unwrap(), ModelFamily::Qwen2);
    }

    #[test]
    fn detects_qwen3_from_architectures() {
        let raw: serde_json::Value =
            serde_json::from_str(r#"{"architectures":["Qwen3ForCausalLM"]}"#).unwrap();
        assert_eq!(detect_family(&raw).unwrap(), ModelFamily::Qwen3);
    }

    #[test]
    fn rejects_unsupported_architecture() {
        let raw: serde_json::Value =
            serde_json::from_str(r#"{"architectures":["MixtralForCausalLM"]}"#).unwrap();
        assert!(detect_family(&raw).is_err());
    }

    #[test]
    fn rejects_missing_architectures_field() {
        let raw: serde_json::Value = serde_json::from_str(r#"{}"#).unwrap();
        assert!(detect_family(&raw).is_err());
    }

    #[test]
    fn load_config_dispatches_to_the_right_parser() {
        let raw: serde_json::Value = serde_json::from_str(
            r#"{"architectures":["Qwen3ForCausalLM"],"hidden_size":64,"intermediate_size":128,
               "vocab_size":256,"num_hidden_layers":2,"num_attention_heads":4,
               "num_key_value_heads":4,"rms_norm_eps":1e-6,"tie_word_embeddings":true}"#,
        )
        .unwrap();
        let cfg = load_config(ModelFamily::Qwen3, &raw).unwrap();
        assert!(cfg.use_qk_norm); // proves the Qwen3 parser ran, not Llama/Qwen2
    }
}
