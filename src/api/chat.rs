use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;

use super::error::ApiError;
use super::openai::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse};
use super::AppState;
use crate::sampling::SamplingParams;

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    if req.messages.is_empty() {
        return Err(ApiError::BadRequest("messages must not be empty".into()));
    }
    let params: SamplingParams = (&req).into();
    let messages = req.messages.clone();
    let model_id = state.engine.lock().unwrap().model_id().to_string();

    if req.stream.unwrap_or(false) {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let engine = state.engine.clone();
        tokio::task::spawn_blocking(move || {
            let engine = engine.lock().unwrap();
            let _ = engine.generate(&messages, &params, |tok| {
                let _ = tx.send(tok.to_string());
            });
        });

        let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let created = now_unix();
        let model_for_stream = model_id.clone();

        let stream = async_stream::stream! {
            let role = ChatCompletionChunk::role_chunk(&completion_id, created, &model_for_stream);
            yield Ok::<_, std::convert::Infallible>(Event::default().data(serde_json::to_string(&role).unwrap()));

            while let Some(tok) = rx.recv().await {
                let chunk = ChatCompletionChunk::delta_chunk(&completion_id, created, &model_for_stream, &tok);
                yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()));
            }

            let done = ChatCompletionChunk::finish_chunk(&completion_id, created, &model_for_stream);
            yield Ok(Event::default().data(serde_json::to_string(&done).unwrap()));
            yield Ok(Event::default().data("[DONE]"));
        };

        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        let engine = state.engine.clone();
        let (text, stats) = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
            let engine = engine.lock().unwrap();
            let mut full = String::new();
            let stats = engine.generate(&messages, &params, |tok| full.push_str(tok))?;
            Ok((full, stats))
        })
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))??;

        let resp = ChatCompletionResponse::new(model_id, text, &stats);
        Ok(Json(resp).into_response())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::AppState;
    use crate::engine::Engine;
    use axum::extract::State;
    use axum::Json;
    use std::sync::{Arc, Mutex};

    fn make_state() -> AppState {
        let dir = crate::engine::tests::fixture_dir_for_external_use();
        let engine = Engine::load(dir.path()).unwrap();
        // Keep the tempdir alive for the engine's lifetime by leaking it — fine in tests.
        std::mem::forget(dir);
        AppState {
            engine: Arc::new(Mutex::new(engine)),
            started_at: 0,
        }
    }

    fn request(stream: bool) -> ChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "flour",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 3,
            "temperature": 0.0,
            "stream": stream
        }))
        .unwrap()
    }

    #[tokio::test]
    async fn blocking_request_returns_chat_completion_json() {
        let state = make_state();
        let resp = chat_completions(State(state), Json(request(false)))
            .await
            .unwrap();
        assert_eq!(resp.status(), axum::http::StatusCode::OK);
    }

    #[tokio::test]
    async fn streaming_request_returns_event_stream_content_type() {
        let state = make_state();
        let resp = chat_completions(State(state), Json(request(true)))
            .await
            .unwrap();
        let content_type = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(content_type.starts_with("text/event-stream"));
    }
}
