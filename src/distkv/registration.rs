//! Shared Worker registration + heartbeat against the Master.
//!
//! Used by the standalone `flour-worker` binary and by the co-located node
//! (engine + embedded worker), so both follow identical liveness behavior:
//! register (retrying until the Master is reachable), heartbeat on an interval,
//! and re-register — adopting the new epoch — whenever a heartbeat is rejected
//! (e.g. the Master restarted and forgot us).

use std::time::Duration;

use crate::distkv::protocol::{HeartbeatRequest, RegisterRequest, RegisterResponse};

/// How often to heartbeat. Must stay well under the Master's
/// `HEARTBEAT_TIMEOUT_MS` (10s) so a healthy worker is never marked dead.
pub const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(3);
/// Delay between registration retries while the Master is unreachable.
pub const REGISTER_RETRY: Duration = Duration::from_secs(2);

/// Registers (or re-registers) with the Master, retrying until it succeeds.
/// Returns the epoch the Master assigned.
pub async fn register(
    http: &reqwest::Client,
    master_url: &str,
    req: &RegisterRequest,
) -> anyhow::Result<u64> {
    loop {
        match http
            .post(format!("{master_url}/v1/distkv/workers/register"))
            .json(req)
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                let reg: RegisterResponse = resp.json().await?;
                return Ok(reg.epoch);
            }
            Ok(resp) => tracing::warn!("register rejected: {}", resp.status()),
            Err(e) => tracing::warn!("master unreachable, retrying registration: {e}"),
        }
        tokio::time::sleep(REGISTER_RETRY).await;
    }
}

/// Sends one heartbeat. Returns true iff the Master accepted it.
pub async fn heartbeat_once(
    http: &reqwest::Client,
    master_url: &str,
    worker_id: &str,
    epoch: u64,
) -> bool {
    http.post(format!("{master_url}/v1/distkv/workers/heartbeat"))
        .json(&HeartbeatRequest {
            worker_id: worker_id.to_string(),
            epoch,
        })
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

/// Registers and then heartbeats forever, re-registering on any rejected
/// heartbeat. Intended to be spawned as a background task.
pub async fn run_registration(
    http: reqwest::Client,
    master_url: String,
    req: RegisterRequest,
) -> anyhow::Result<()> {
    let mut epoch = register(&http, &master_url, &req).await?;
    tracing::info!(
        "registered '{}' with master as epoch {epoch}",
        req.worker_id
    );
    loop {
        tokio::time::sleep(HEARTBEAT_INTERVAL).await;
        if !heartbeat_once(&http, &master_url, &req.worker_id, epoch).await {
            tracing::warn!("heartbeat failed, re-registering with master");
            epoch = register(&http, &master_url, &req).await?;
            tracing::info!("re-registered '{}' as epoch {epoch}", req.worker_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distkv::http::master_router;
    use crate::distkv::master::MasterState;
    use crate::distkv::protocol::RegisterRequest;
    use std::sync::{Arc, Mutex};

    async fn spawn_master() -> String {
        let master = Arc::new(Mutex::new(MasterState::new(|| 1_000)));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, master_router(master)).await.unwrap();
        });
        format!("http://{addr}")
    }

    fn req() -> RegisterRequest {
        RegisterRequest {
            worker_id: "w1".into(),
            addr: "http://w1:8090".into(),
            capacity_bytes: 1 << 20,
        }
    }

    #[tokio::test]
    async fn register_returns_incrementing_epoch() {
        let url = spawn_master().await;
        let http = reqwest::Client::new();
        assert_eq!(register(&http, &url, &req()).await.unwrap(), 1);
        // Re-registering the same worker bumps the epoch.
        assert_eq!(register(&http, &url, &req()).await.unwrap(), 2);
    }

    #[tokio::test]
    async fn heartbeat_once_true_for_current_epoch_false_for_stale() {
        let url = spawn_master().await;
        let http = reqwest::Client::new();
        let epoch = register(&http, &url, &req()).await.unwrap();
        assert!(heartbeat_once(&http, &url, "w1", epoch).await);
        assert!(!heartbeat_once(&http, &url, "w1", epoch + 99).await);
        assert!(!heartbeat_once(&http, &url, "unknown", epoch).await);
    }
}
