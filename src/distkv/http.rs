//! Axum HTTP routes for the Master (metadata path) and Worker (data path).
//!
//! The Master router exposes only metadata endpoints and never touches object
//! bytes. The Worker router serves bytes keyed by `(key, generation)`.

use std::sync::{Arc, Mutex, MutexGuard};

use axum::body::Bytes;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};

use crate::distkv::master::MasterState;
use crate::distkv::protocol::*;
use crate::distkv::worker::WorkerStore;

pub type SharedMaster = Arc<Mutex<MasterState>>;
pub type SharedWorker = Arc<Mutex<WorkerStore>>;

/// An error response with an explicit HTTP status and a message body.
///
/// Business/validation rejections (mismatched put_id, no capacity, dead worker)
/// surface as 400; infrastructure failures (e.g. a poisoned state lock) surface
/// as 500. The data path proper uses 404 for misses.
struct AppError {
    status: StatusCode,
    message: String,
}

impl AppError {
    fn internal(message: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.to_string(),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (self.status, self.message).into_response()
    }
}

impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(e: E) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: e.into().to_string(),
        }
    }
}

/// Acquires the shared state lock, turning a poisoned lock into a 500 instead
/// of panicking and dropping the connection.
fn lock<T>(state: &Mutex<T>) -> Result<MutexGuard<'_, T>, AppError> {
    state
        .lock()
        .map_err(|_| AppError::internal("distkv state lock poisoned"))
}

pub fn master_router(state: SharedMaster) -> Router {
    Router::new()
        .route("/v1/distkv/workers/register", post(register))
        .route("/v1/distkv/workers/heartbeat", post(heartbeat))
        .route("/v1/distkv/put_start", post(put_start))
        .route("/v1/distkv/put_commit", post(put_commit))
        .route("/v1/distkv/get_route", get(get_route))
        .with_state(state)
}

pub fn worker_router(state: SharedWorker) -> Router {
    Router::new()
        .route(
            "/v1/distkv/worker/objects/{key}",
            put(put_object).get(get_object).delete(delete_object),
        )
        .with_state(state)
}

// --- Master handlers ---

async fn register(
    State(m): State<SharedMaster>,
    Json(req): Json<RegisterRequest>,
) -> Result<Json<RegisterResponse>, AppError> {
    let epoch = lock(&m)?.register_worker(req.worker_id, req.addr, req.capacity_bytes);
    Ok(Json(RegisterResponse { epoch }))
}

async fn heartbeat(
    State(m): State<SharedMaster>,
    Json(req): Json<HeartbeatRequest>,
) -> Result<StatusCode, AppError> {
    lock(&m)?.heartbeat(&req.worker_id, req.epoch)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn put_start(
    State(m): State<SharedMaster>,
    Json(req): Json<PutStartRequest>,
) -> Result<Json<PutStartResponse>, AppError> {
    let resp = lock(&m)?.put_start(req)?;
    Ok(Json(resp))
}

async fn put_commit(
    State(m): State<SharedMaster>,
    Json(req): Json<PutCommitRequest>,
) -> Result<StatusCode, AppError> {
    lock(&m)?.put_commit(req)?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(serde::Deserialize)]
struct GetRouteQuery {
    key: String,
}

async fn get_route(
    State(m): State<SharedMaster>,
    Query(q): Query<GetRouteQuery>,
) -> Result<Response, AppError> {
    let route = lock(&m)?.get_route(&q.key)?;
    Ok(match route {
        Some(route) => Json(route).into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    })
}

// --- Worker handlers ---

#[derive(serde::Deserialize)]
struct GenerationQuery {
    generation: u64,
}

async fn put_object(
    State(w): State<SharedWorker>,
    Path(key): Path<String>,
    Query(q): Query<GenerationQuery>,
    body: Bytes,
) -> Result<StatusCode, AppError> {
    lock(&w)?.put_bytes(key, q.generation, body.to_vec())?;
    Ok(StatusCode::NO_CONTENT)
}

async fn get_object(
    State(w): State<SharedWorker>,
    Path(key): Path<String>,
    Query(q): Query<GenerationQuery>,
) -> Result<Response, AppError> {
    let bytes = lock(&w)?.get_bytes(&key, q.generation)?;
    Ok(match bytes {
        Some(bytes) => bytes.into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    })
}

async fn delete_object(
    State(w): State<SharedWorker>,
    Path(key): Path<String>,
    Query(q): Query<GenerationQuery>,
) -> Result<StatusCode, AppError> {
    lock(&w)?.delete_generation(&key, q.generation)?;
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    //! Real-HTTP tests for the distributed KV cache data path.
    //!
    //! Proves the metadata/data split: the Master only ever returns routes, and
    //! object bytes flow directly between the client and the Worker. The Master
    //! server exposes no object-byte route at all.

    use super::*;

    async fn spawn(app: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }

    fn master() -> SharedMaster {
        Arc::new(Mutex::new(MasterState::new(|| 1_000)))
    }

    #[tokio::test]
    async fn put_get_data_path_bypasses_master() {
        // --- Master server (metadata only) ---
        let master_url = spawn(master_router(master())).await;

        // --- Worker server (bytes only) ---
        let worker_store = Arc::new(Mutex::new(WorkerStore::new("w1".into(), 1, 1 << 20)));
        let worker_url = spawn(worker_router(worker_store)).await;

        let http = reqwest::Client::new();

        // Register the worker with the master, advertising its data-path URL.
        let reg: RegisterResponse = http
            .post(format!("{master_url}/v1/distkv/workers/register"))
            .json(&RegisterRequest {
                worker_id: "w1".into(),
                addr: worker_url.clone(),
                capacity_bytes: 1 << 20,
            })
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(reg.epoch, 1);

        let key = "kv://v1/model/m/prefix/abc/tokens/64";
        let payload = vec![7u8; 256];

        // 1. put_start -> master picks a worker and opens a write.
        let start: PutStartResponse = http
            .post(format!("{master_url}/v1/distkv/put_start"))
            .json(&PutStartRequest {
                key: key.into(),
                size_bytes: payload.len(),
            })
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(start.worker_id, "w1");
        assert_eq!(start.worker_addr, worker_url);

        // 2. client writes bytes DIRECTLY to the worker.
        let put = http
            .put(format!(
                "{}/v1/distkv/worker/objects/{}?generation={}",
                start.worker_addr,
                urlencoding(key),
                start.object_generation
            ))
            .body(payload.clone())
            .send()
            .await
            .unwrap();
        assert!(put.status().is_success(), "worker PUT failed: {put:?}");

        // 3. commit metadata.
        let commit = http
            .post(format!("{master_url}/v1/distkv/put_commit"))
            .json(&PutCommitRequest {
                key: key.into(),
                put_id: start.put_id,
            })
            .send()
            .await
            .unwrap();
        assert!(commit.status().is_success());

        // 4. get_route -> master returns the worker route.
        let route_resp = http
            .get(format!("{master_url}/v1/distkv/get_route"))
            .query(&[("key", key)])
            .send()
            .await
            .unwrap();
        assert_eq!(route_resp.status(), 200);
        let route: GetRouteResponse = route_resp.json().await.unwrap();
        assert_eq!(route.worker_addr, worker_url);
        assert_eq!(route.object_generation, start.object_generation);

        // 5. client fetches bytes DIRECTLY from the worker.
        let fetched = http
            .get(format!(
                "{}/v1/distkv/worker/objects/{}?generation={}",
                route.worker_addr,
                urlencoding(key),
                route.object_generation
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(fetched.status(), 200);
        let bytes = fetched.bytes().await.unwrap();
        assert_eq!(bytes.as_ref(), payload.as_slice());

        // 6. The master has no byte route: writing/reading objects against it must
        //    not succeed (proves bytes never touch the master).
        let on_master = http
            .get(format!(
                "{master_url}/v1/distkv/worker/objects/{}?generation={}",
                urlencoding(key),
                route.object_generation
            ))
            .send()
            .await
            .unwrap();
        assert!(
            !on_master.status().is_success(),
            "master must not serve object bytes, got {}",
            on_master.status()
        );
    }

    #[tokio::test]
    async fn get_route_misses_for_unknown_key() {
        let master_url = spawn(master_router(master())).await;

        let resp = reqwest::Client::new()
            .get(format!("{master_url}/v1/distkv/get_route"))
            .query(&[("key", "does-not-exist")])
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
    }

    /// Minimal percent-encoding for the path segment (the key contains `/` and `:`).
    fn urlencoding(s: &str) -> String {
        let mut out = String::new();
        for b in s.bytes() {
            match b {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                    out.push(b as char)
                }
                _ => out.push_str(&format!("%{b:02X}")),
            }
        }
        out
    }
}
