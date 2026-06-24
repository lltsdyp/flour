use serde::{Deserialize, Serialize};

use crate::engine::GenerationStats;
use crate::sampling::SamplingParams;
use crate::tokenizer::ChatMessage;

/// OpenAI `stop` accepts either a single string or an array of strings.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopSequences {
    One(String),
    Many(Vec<String>),
}

impl StopSequences {
    fn to_vec(&self) -> Vec<String> {
        match self {
            StopSequences::One(s) => vec![s.clone()],
            StopSequences::Many(v) => v.clone(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: Option<bool>,
}

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
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<StopSequences>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
}

impl ChatCompletionRequest {
    /// Whether the client requested token usage in the streamed response via
    /// `stream_options.include_usage`.
    pub fn wants_usage(&self) -> bool {
        self.stream_options
            .as_ref()
            .and_then(|o| o.include_usage)
            .unwrap_or(false)
    }
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
            seed: req.seed.unwrap_or(defaults.seed),
            stop: req
                .stop
                .as_ref()
                .map(StopSequences::to_vec)
                .unwrap_or_default(),
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
                finish_reason: stats.finish_reason.as_str().into(),
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
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
            usage: None,
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
            usage: None,
        }
    }

    pub fn finish_chunk(id: &str, created: u64, model: &str, finish_reason: &str) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".into(),
            created,
            model: model.to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: StreamDelta::default(),
                finish_reason: Some(finish_reason.to_string()),
            }],
            usage: None,
        }
    }

    /// Final chunk carrying token usage, emitted when the client requests
    /// `stream_options.include_usage`. Per the OpenAI spec it carries no choices.
    pub fn usage_chunk(id: &str, created: u64, model: &str, usage: Usage) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".into(),
            created,
            model: model.to_string(),
            choices: vec![],
            usage: Some(usage),
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
    use crate::engine::{FinishReason, GenerationStats};

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
            reused_prefix_tokens: 0,
            finish_reason: FinishReason::Stop,
            remote_cache_hit: None,
            remote_key: None,
        };
        let resp = ChatCompletionResponse::new("flour".into(), "hello".into(), &stats);
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["choices"][0]["message"]["content"], "hello");
        assert_eq!(json["usage"]["total_tokens"], 5);
    }

    #[test]
    fn response_reports_finish_reason_from_stats() {
        let stats = GenerationStats {
            prompt_tokens: 3,
            completion_tokens: 2,
            reused_prefix_tokens: 0,
            finish_reason: FinishReason::Length,
            remote_cache_hit: None,
            remote_key: None,
        };
        let resp = ChatCompletionResponse::new("flour".into(), "hi".into(), &stats);
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["choices"][0]["finish_reason"], "length");
    }

    #[test]
    fn request_parses_seed_stop_array_and_stream_options() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":"hi"}],"seed":42,
                "stop":["<|im_end|>"],"stream":true,
                "stream_options":{"include_usage":true}}"#,
        )
        .unwrap();
        assert_eq!(req.seed, Some(42));
        assert!(req.wants_usage());
        let params: SamplingParams = (&req).into();
        assert_eq!(params.seed, 42);
        assert_eq!(params.stop, vec!["<|im_end|>".to_string()]);
    }

    #[test]
    fn request_parses_stop_as_single_string() {
        let req: ChatCompletionRequest =
            serde_json::from_str(r#"{"messages":[{"role":"user","content":"hi"}],"stop":"END"}"#)
                .unwrap();
        let params: SamplingParams = (&req).into();
        assert_eq!(params.stop, vec!["END".to_string()]);
        // No stream_options means usage is not requested.
        assert!(!req.wants_usage());
    }

    #[test]
    fn finish_chunk_serializes_given_finish_reason() {
        let fin = ChatCompletionChunk::finish_chunk("id1", 0, "flour", "length");
        let json = serde_json::to_value(&fin).unwrap();
        assert_eq!(json["choices"][0]["finish_reason"], "length");
        assert!(json.get("usage").is_none());
    }

    #[test]
    fn usage_chunk_serializes_with_usage_and_empty_choices() {
        let chunk = ChatCompletionChunk::usage_chunk(
            "id1",
            0,
            "flour",
            Usage {
                prompt_tokens: 7,
                completion_tokens: 3,
                total_tokens: 10,
            },
        );
        let json = serde_json::to_value(&chunk).unwrap();
        assert_eq!(json["usage"]["prompt_tokens"], 7);
        assert_eq!(json["usage"]["total_tokens"], 10);
        assert_eq!(json["choices"].as_array().unwrap().len(), 0);
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

        let fin = ChatCompletionChunk::finish_chunk("id1", 0, "flour", "stop");
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
