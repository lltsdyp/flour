//! Standalone Worker server for the distributed KV cache.
//!
//! A Worker owns KV object bytes and serves them directly to requesters over the
//! data path. On startup it registers with the Master (advertising the URL peers
//! should use to reach it) and then heartbeats periodically so the Master keeps
//! routing reads to it. If the Master forgets the worker (e.g. it restarted), the
//! next heartbeat fails and the worker transparently re-registers.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use clap::Parser;
use flour::distkv::http::worker_router;
use flour::distkv::protocol::{HeartbeatRequest, RegisterRequest, RegisterResponse};
use flour::distkv::worker::WorkerStore;

/// How often to heartbeat. Must be well under the Master's heartbeat timeout
/// (`HEARTBEAT_TIMEOUT_MS` = 10s) so a healthy worker is never marked dead.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(3);
/// Delay between registration retries while the Master is still unreachable.
const REGISTER_RETRY: Duration = Duration::from_secs(2);

#[derive(Parser, Debug)]
#[command(name = "flour-worker", about = "Distributed KV cache Worker (object bytes)")]
struct Args {
    /// Stable identity for this worker across restarts.
    #[arg(long)]
    worker_id: String,

    /// Master metadata URL (e.g. http://master:8081).
    #[arg(long)]
    master_url: String,

    /// URL other nodes should use to reach this worker's data path
    /// (e.g. http://worker1:8090). Defaults to http://<host>:<port>.
    #[arg(long)]
    advertise_url: Option<String>,

    /// Address to bind the data-path HTTP server to.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    #[arg(long, default_value_t = 8090)]
    port: u16,

    /// Object-store capacity advertised to the Master, in bytes.
    #[arg(long, default_value_t = 1 << 30)]
    capacity_bytes: usize,
}

/// Registers (or re-registers) with the Master, retrying until it succeeds.
/// Returns the epoch the Master assigned to this registration.
async fn register(
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    let master_url = args.master_url.trim_end_matches('/').to_string();
    let advertise_url = args
        .advertise_url
        .clone()
        .unwrap_or_else(|| format!("http://{}:{}", args.host, args.port));

    // Bring up the data-path server first so we can serve as soon as we're routed.
    let store = Arc::new(Mutex::new(WorkerStore::new(
        args.worker_id.clone(),
        0,
        args.capacity_bytes,
    )));
    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    tracing::info!("distkv worker '{}' data path on http://{addr}", args.worker_id);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let server = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, worker_router(store)).await {
            tracing::error!("worker server exited: {e}");
        }
    });

    // Register with the Master and keep heartbeating in the background.
    let http = reqwest::Client::new();
    let reg = RegisterRequest {
        worker_id: args.worker_id.clone(),
        addr: advertise_url.clone(),
        capacity_bytes: args.capacity_bytes,
    };
    let mut epoch = register(&http, &master_url, &reg).await?;
    tracing::info!("registered '{}' with master as epoch {epoch}", args.worker_id);

    loop {
        tokio::time::sleep(HEARTBEAT_INTERVAL).await;
        let hb = HeartbeatRequest {
            worker_id: args.worker_id.clone(),
            epoch,
        };
        let ok = http
            .post(format!("{master_url}/v1/distkv/workers/heartbeat"))
            .json(&hb)
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false);
        if !ok {
            // Master lost our registration (likely a restart): re-register and
            // adopt the new epoch.
            tracing::warn!("heartbeat failed, re-registering with master");
            epoch = register(&http, &master_url, &reg).await?;
            tracing::info!("re-registered '{}' as epoch {epoch}", args.worker_id);
        }
        if server.is_finished() {
            anyhow::bail!("worker data-path server stopped");
        }
    }
}
