use serde::{Deserialize, Serialize};

use crate::engine::GenerationStats;
use crate::sampling::SamplingParams;
use crate::tokenizer::ChatMessage;

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
}

impl From<&ChatCompletionRequest> for SamplingParams {
    fn from(req: &ChatCompletionRequest) -> Self {
        let defaults = SamplingParams::default();
        SamplingParams {
            temperature: req.temperature.unwrap_or(defaults.temperature),
            top_p: req.top_p.or(defaults.top_p),
            top_k: req.top_k.or(defaults.top_k),
            max_tokens: req.max_tokens.unwrap_or(defaults.max_tokens),
            repeat_penalty: defaults.repeat_penalty,
            repeat_last_n: defaults.repeat_last_n,
            seed: defaults.seed,
        }
    }
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

impl ChatCompletionResponse {
    pub fn new(model: String, content: String, stats: &GenerationStats) -> Self {
        Self {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".into(),
            created: now_unix(),
            model,
            choices: vec![Choice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content,
                },
                finish_reason: "stop".into(),
            }],
            usage: Usage {
                prompt_tokens: stats.prompt_tokens,
                completion_tokens: stats.completion_tokens,
                total_tokens: stats.prompt_tokens + stats.completion_tokens,
            },
        }
    }
}

#[derive(Debug, Serialize, Default)]
pub struct StreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: StreamDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

impl ChatCompletionChunk {
    pub fn role_chunk(id: &str, created: u64, model: &str) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".into(),
            created,
            model: model.to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: StreamDelta {
                    role: Some("assistant".into()),
                    content: None,
                },
                finish_reason: None,
            }],
        }
    }

    pub fn delta_chunk(id: &str, created: u64, model: &str, content: &str) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".into(),
            created,
            model: model.to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: StreamDelta {
                    role: None,
                    content: Some(content.to_string()),
                },
                finish_reason: None,
            }],
        }
    }

    pub fn finish_chunk(id: &str, created: u64, model: &str) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".into(),
            created,
            model: model.to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: StreamDelta::default(),
                finish_reason: Some("stop".into()),
            }],
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Serialize)]
pub struct ModelsListResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::GenerationStats;

    #[test]
    fn request_deserializes_with_only_required_fields() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"flour","messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.stream, None);
        assert_eq!(req.temperature, None);
    }

    #[test]
    fn sampling_params_take_request_overrides_and_otherwise_default() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":"hi"}],"temperature":0.0,"max_tokens":16}"#,
        )
        .unwrap();
        let params: crate::sampling::SamplingParams = (&req).into();
        assert_eq!(params.temperature, 0.0);
        assert_eq!(params.max_tokens, 16);
    }

    #[test]
    fn response_serializes_with_expected_shape() {
        let stats = GenerationStats {
            prompt_tokens: 3,
            completion_tokens: 2,
        };
        let resp = ChatCompletionResponse::new("flour".into(), "hello".into(), &stats);
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["choices"][0]["message"]["content"], "hello");
        assert_eq!(json["usage"]["total_tokens"], 5);
    }

    #[test]
    fn chunk_constructors_serialize_with_delta_shape() {
        let role = ChatCompletionChunk::role_chunk("id1", 0, "flour");
        let role_json = serde_json::to_value(&role).unwrap();
        assert_eq!(role_json["choices"][0]["delta"]["role"], "assistant");
        assert!(role_json["choices"][0]["delta"]["content"].is_null());

        let delta = ChatCompletionChunk::delta_chunk("id1", 0, "flour", "hi");
        let delta_json = serde_json::to_value(&delta).unwrap();
        assert_eq!(delta_json["choices"][0]["delta"]["content"], "hi");

        let fin = ChatCompletionChunk::finish_chunk("id1", 0, "flour");
        let fin_json = serde_json::to_value(&fin).unwrap();
        assert_eq!(fin_json["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn models_list_response_serializes() {
        let resp = ModelsListResponse {
            object: "list".into(),
            data: vec![ModelObject {
                id: "flour".into(),
                object: "model".into(),
                created: 0,
                owned_by: "flour".into(),
            }],
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["data"][0]["id"], "flour");
    }
}
