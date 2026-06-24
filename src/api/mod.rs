pub mod chat;
pub mod error;
pub mod models;
pub mod openai;

use std::sync::{Arc, Mutex};

use crate::engine::Engine;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Mutex<Engine>>,
    pub started_at: u64,
}

use axum::routing::{get, post};
use axum::Router;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/models", get(models::list_models))
        .route("/v1/chat/completions", post(chat::chat_completions))
        .with_state(state)
}

/// Mounts the distributed KV cache Master/Worker routes onto `router` when the
/// node's role requires them. Each sub-router carries its own state, so the
/// engine `AppState` is untouched.
pub fn mount_distkv(
    router: Router,
    master: Option<crate::distkv::http::SharedMaster>,
    worker: Option<crate::distkv::http::SharedWorker>,
) -> Router {
    let mut router = router;
    if let Some(master) = master {
        router = router.merge(crate::distkv::http::master_router(master));
    }
    if let Some(worker) = worker {
        router = router.merge(crate::distkv::http::worker_router(worker));
    }
    router
}

pub async fn serve(state: AppState, addr: std::net::SocketAddr) -> anyhow::Result<()> {
    serve_router(router(state), addr).await
}

/// Serves a fully-built router (used when extra routes such as the co-located
/// worker data path have already been merged in).
pub async fn serve_router(app: Router, addr: std::net::SocketAddr) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Engine;
    use std::sync::{Arc, Mutex};

    async fn spawn_test_server() -> std::net::SocketAddr {
        let dir = crate::engine::tests::fixture_dir_for_external_use();
        let engine = Engine::load(dir.path(), None).unwrap();
        std::mem::forget(dir);
        let state = AppState {
            engine: Arc::new(Mutex::new(engine)),
            started_at: 0,
        };

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = router(state);
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        addr
    }

    #[tokio::test]
    async fn models_endpoint_returns_loaded_model_over_real_http() {
        let addr = spawn_test_server().await;
        let resp = reqwest::get(format!("http://{addr}/v1/models"))
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let json: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn chat_completions_endpoint_returns_message_over_real_http() {
        let addr = spawn_test_server().await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://{addr}/v1/chat/completions"))
            .json(&serde_json::json!({
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 3,
                "temperature": 0.0
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let json: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert!(json["choices"][0]["message"]["content"].is_string());
    }

    #[tokio::test]
    async fn one_server_serves_api_and_worker_data_path() {
        let dir = crate::engine::tests::fixture_dir_for_external_use();
        let engine = Engine::load(dir.path(), None).unwrap();
        std::mem::forget(dir);
        let state = AppState {
            engine: Arc::new(Mutex::new(engine)),
            started_at: 0,
        };
        let store = Arc::new(Mutex::new(crate::distkv::worker::WorkerStore::new(
            "local".into(),
            0,
            1 << 20,
        )));
        let app = mount_distkv(router(state), None, Some(store));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let http = reqwest::Client::new();
        // OpenAI API on this server.
        let models = http
            .get(format!("http://{addr}/v1/models"))
            .send()
            .await
            .unwrap();
        assert_eq!(models.status(), 200);
        // Worker data path on the SAME server: PUT then GET round-trips bytes.
        let put = http
            .put(format!(
                "http://{addr}/v1/distkv/worker/objects/k?generation=1"
            ))
            .body(vec![1u8, 2, 3])
            .send()
            .await
            .unwrap();
        assert!(put.status().is_success());
        let got = http
            .get(format!(
                "http://{addr}/v1/distkv/worker/objects/k?generation=1"
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(got.status(), 200);
        assert_eq!(got.bytes().await.unwrap().as_ref(), &[1u8, 2, 3]);
    }
}
