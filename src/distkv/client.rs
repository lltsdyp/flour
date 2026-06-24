//! HTTP client used by the Engine/Scheduler to drive the distributed KV cache.
//!
//! `DistKvClient` talks to the Master over the metadata path and to the Worker
//! over the data path. It implements the two-phase PUT (`put_start` → write
//! bytes to the Worker → `put_commit`) and the GET route lookup, keeping all
//! object bytes off the Master.

use crate::distkv::protocol::*;

/// Client for the metadata (Master) and data (Worker) paths.
pub struct DistKvClient {
    master_url: String,
    http: reqwest::Client,
}

impl DistKvClient {
    pub fn new(master_url: impl Into<String>) -> Self {
        Self {
            master_url: master_url.into().trim_end_matches('/').to_string(),
            http: reqwest::Client::new(),
        }
    }

    pub fn with_http(master_url: impl Into<String>, http: reqwest::Client) -> Self {
        Self {
            master_url: master_url.into().trim_end_matches('/').to_string(),
            http,
        }
    }

    /// Phase 1 of PUT: ask the Master to open a write, optionally pinning the
    /// object to `preferred_worker_id` (write locality).
    pub async fn put_start(
        &self,
        key: &str,
        size_bytes: usize,
        preferred_worker_id: Option<&str>,
    ) -> anyhow::Result<PutStartResponse> {
        self.post_json(
            "/v1/distkv/put_start",
            &PutStartRequest {
                key: key.to_string(),
                size_bytes,
                preferred_worker_id: preferred_worker_id.map(|s| s.to_string()),
            },
        )
        .await
    }

    /// Writes object bytes directly to a Worker's data path (never via Master).
    pub async fn write_worker(
        &self,
        worker_addr: &str,
        key: &str,
        generation: u64,
        bytes: Vec<u8>,
    ) -> anyhow::Result<()> {
        let put = self
            .http
            .put(format!(
                "{}/v1/distkv/worker/objects/{}?generation={}",
                worker_addr.trim_end_matches('/'),
                encode_segment(key),
                generation
            ))
            .body(bytes)
            .send()
            .await?;
        if !put.status().is_success() {
            anyhow::bail!("worker write failed with status {}", put.status());
        }
        Ok(())
    }

    /// Phase 2 of PUT: publish the object as `Complete`.
    pub async fn put_commit(&self, key: &str, put_id: PutId) -> anyhow::Result<()> {
        let commit = self
            .http
            .post(format!("{}/v1/distkv/put_commit", self.master_url))
            .json(&PutCommitRequest {
                key: key.to_string(),
                put_id,
            })
            .send()
            .await?;
        if !commit.status().is_success() {
            anyhow::bail!("put_commit failed with status {}", commit.status());
        }
        Ok(())
    }

    /// Looks up a read route for `key`. `Ok(None)` is a clean miss.
    pub async fn get_route(&self, key: &str) -> anyhow::Result<Option<GetRouteResponse>> {
        let route_resp = self
            .http
            .get(format!("{}/v1/distkv/get_route", self.master_url))
            .query(&[("key", key)])
            .send()
            .await?;
        if route_resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !route_resp.status().is_success() {
            anyhow::bail!("get_route failed with status {}", route_resp.status());
        }
        Ok(Some(route_resp.json().await?))
    }

    /// Fetches object bytes directly from a Worker. `Ok(None)` if that
    /// generation no longer exists (a safe miss, never stale bytes).
    pub async fn fetch_worker(
        &self,
        worker_addr: &str,
        key: &str,
        generation: u64,
    ) -> anyhow::Result<Option<Vec<u8>>> {
        let fetched = self
            .http
            .get(format!(
                "{}/v1/distkv/worker/objects/{}?generation={}",
                worker_addr.trim_end_matches('/'),
                encode_segment(key),
                generation
            ))
            .send()
            .await?;
        if fetched.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !fetched.status().is_success() {
            anyhow::bail!("worker fetch failed with status {}", fetched.status());
        }
        Ok(Some(fetched.bytes().await?.to_vec()))
    }

    /// Two-phase PUT over HTTP (no locality). Bytes go straight to the Worker.
    pub async fn put_object(&self, key: &str, bytes: Vec<u8>) -> anyhow::Result<()> {
        let start = self.put_start(key, bytes.len(), None).await?;
        self.write_worker(&start.worker_addr, key, start.object_generation, bytes)
            .await?;
        self.put_commit(key, start.put_id).await
    }

    /// Route lookup + direct Worker fetch over HTTP (no locality).
    pub async fn get_object(&self, key: &str) -> anyhow::Result<Option<Vec<u8>>> {
        let route = match self.get_route(key).await? {
            Some(r) => r,
            None => return Ok(None),
        };
        self.fetch_worker(&route.worker_addr, key, route.object_generation)
            .await
    }

    async fn post_json<Req, Resp>(&self, path: &str, body: &Req) -> anyhow::Result<Resp>
    where
        Req: serde::Serialize,
        Resp: serde::de::DeserializeOwned,
    {
        let resp = self
            .http
            .post(format!("{}{}", self.master_url, path))
            .json(body)
            .send()
            .await?;
        if !resp.status().is_success() {
            anyhow::bail!("{path} failed with status {}", resp.status());
        }
        Ok(resp.json().await?)
    }
}

/// Percent-encodes a single URL path segment. The object key contains `/` and
/// `:`, which must not be interpreted as path structure.
fn encode_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distkv::http::{master_router, worker_router, SharedMaster, SharedWorker};
    use crate::distkv::master::MasterState;
    use crate::distkv::worker::WorkerStore;
    use std::sync::{Arc, Mutex};

    async fn spawn(app: axum::Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }

    /// Brings up a Master + one registered Worker and returns a client.
    async fn cluster() -> DistKvClient {
        let master: SharedMaster = Arc::new(Mutex::new(MasterState::new(|| 1_000)));
        let master_url = spawn(master_router(master)).await;

        let worker: SharedWorker = Arc::new(Mutex::new(WorkerStore::new("w1".into(), 1, 1 << 20)));
        let worker_url = spawn(worker_router(worker)).await;

        let client = DistKvClient::new(master_url);
        client
            .post_json::<_, RegisterResponse>(
                "/v1/distkv/workers/register",
                &RegisterRequest {
                    worker_id: "w1".into(),
                    addr: worker_url,
                    capacity_bytes: 1 << 20,
                },
            )
            .await
            .unwrap();
        client
    }

    #[tokio::test]
    async fn put_object_then_get_object_round_trips_through_real_http() {
        let client = cluster().await;
        let key = "kv://v1/model/m/prefix/abc/tokens/64";
        let payload = vec![9u8; 512];

        client.put_object(key, payload.clone()).await.unwrap();
        let got = client.get_object(key).await.unwrap();
        assert_eq!(got, Some(payload));
    }

    #[tokio::test]
    async fn get_object_miss_returns_none() {
        let client = cluster().await;
        let got = client.get_object("kv://nope").await.unwrap();
        assert_eq!(got, None);
    }

    #[tokio::test]
    async fn put_start_routes_to_preferred_worker() {
        let client = cluster().await; // registers worker "w1"
        let start = client.put_start("kv://k", 128, Some("w1")).await.unwrap();
        assert_eq!(start.worker_id, "w1");
        assert_eq!(start.object_generation, 1);
    }
}
