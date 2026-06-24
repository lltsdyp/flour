//! Standalone Master server for the distributed KV cache.
//!
//! The Master owns *only* metadata (worker liveness, capacity, object state,
//! placement, leases, routing) and never touches object bytes. Workers register
//! against it and heartbeat; the Engine/Scheduler drives PUT/GET routing through
//! it. See `docs/plan/plan-distkv.md` for the full specification.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use clap::Parser;
use flour::distkv::http::master_router;
use flour::distkv::master::MasterState;

#[derive(Parser, Debug)]
#[command(name = "flour-master", about = "Distributed KV cache Master (metadata only)")]
struct Args {
    /// Address to bind the metadata HTTP server to.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    #[arg(long, default_value_t = 8081)]
    port: u16,
}

/// Current wall-clock time in milliseconds since the Unix epoch.
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    let state = Arc::new(Mutex::new(MasterState::new(now_ms)));
    let app = master_router(state);

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    tracing::info!("distkv master listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
