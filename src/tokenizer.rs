use std::path::Path;

use anyhow::anyhow;
use serde::{Deserialize, Serialize};

use crate::models::ModelFamily;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Copy)]
pub enum ChatTemplate {
    Llama3,
    ChatMl,
}

impl ChatTemplate {
    pub fn for_family(family: ModelFamily) -> Self {
        match family {
            ModelFamily::Llama => ChatTemplate::Llama3,
            ModelFamily::Qwen2 | ModelFamily::Qwen3 => ChatTemplate::ChatMl,
        }
    }

    pub fn render(&self, messages: &[ChatMessage]) -> String {
        match self {
            ChatTemplate::Llama3 => {
                let mut out = String::from("<|begin_of_text|>");
                for m in messages {
                    out.push_str(&format!(
                        "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                        m.role, m.content
                    ));
                }
                out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                out
            }
            ChatTemplate::ChatMl => {
                let mut out = String::new();
                for m in messages {
                    out.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content));
                }
                out.push_str("<|im_start|>assistant\n");
                out
            }
        }
    }
}

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow!("failed to load tokenizer from {}: {e}", path.display()))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow!("tokenizer encode failed: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| anyhow!("tokenizer decode failed: {e}"))
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ModelFamily;

    fn msgs() -> Vec<ChatMessage> {
        vec![
            ChatMessage { role: "system".into(), content: "You are helpful.".into() },
            ChatMessage { role: "user".into(), content: "Hi".into() },
        ]
    }

    #[test]
    fn llama_template_wraps_each_turn_and_opens_assistant_turn() {
        let out = ChatTemplate::Llama3.render(&msgs());
        assert!(out.starts_with("<|begin_of_text|>"));
        assert!(out.contains("<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>"));
        assert!(out.contains("<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"));
        assert!(out.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn chatml_template_wraps_each_turn_and_opens_assistant_turn() {
        let out = ChatTemplate::ChatMl.render(&msgs());
        assert!(out.contains("<|im_start|>system\nYou are helpful.<|im_end|>\n"));
        assert!(out.contains("<|im_start|>user\nHi<|im_end|>\n"));
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn for_family_picks_llama3_for_llama_and_chatml_for_qwen() {
        assert!(matches!(ChatTemplate::for_family(ModelFamily::Llama), ChatTemplate::Llama3));
        assert!(matches!(ChatTemplate::for_family(ModelFamily::Qwen2), ChatTemplate::ChatMl));
        assert!(matches!(ChatTemplate::for_family(ModelFamily::Qwen3), ChatTemplate::ChatMl));
    }
}
