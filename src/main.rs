use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use clap::Parser;
use flour::api::{mount_distkv, router, serve_router, AppState};
use flour::distkv::protocol::RegisterRequest;
use flour::distkv::registration::run_registration;
use flour::distkv::worker::WorkerStore;
use flour::engine::Engine;

#[derive(Parser, Debug)]
#[command(
    name = "flour",
    about = "Minimal CPU-only OpenAI-compatible inference server"
)]
struct Args {
    /// Directory containing config.json, tokenizer.json, and safetensors weights.
    #[arg(long)]
    model_dir: PathBuf,

    /// Dtype to load the model in: f32, bf16, or f16. Defaults to the model's `torch_dtype`
    /// from config.json (falling back to f32).
    #[arg(long)]
    dtype: Option<String>,

    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Enable the optional distributed KV cache client (remote-only). Requires
    /// `--remote-kv-master-url`. Mutually exclusive with `--colocated-worker`.
    #[arg(long, default_value_t = false)]
    remote_kv_enabled: bool,

    /// Master URL for the distributed KV cache (e.g. http://127.0.0.1:8081).
    #[arg(long)]
    remote_kv_master_url: Option<String>,

    /// Run an embedded KV-cache worker in this process (co-located mode): writes
    /// prefer the local worker and local routes are read in-process. Requires
    /// `--remote-kv-master-url` and `--worker-id`.
    #[arg(long, default_value_t = false)]
    colocated_worker: bool,

    /// This node's stable worker identity (co-located mode).
    #[arg(long)]
    worker_id: Option<String>,

    /// URL other nodes use to reach this node's worker data path. Defaults to
    /// http://<host>:<port> (the same server as the API).
    #[arg(long)]
    advertise_url: Option<String>,

    /// Embedded worker capacity advertised to the Master, in bytes.
    #[arg(long, default_value_t = 1 << 30)]
    distkv_capacity_bytes: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    if args.colocated_worker && args.remote_kv_enabled {
        anyhow::bail!("--colocated-worker and --remote-kv-enabled are mutually exclusive");
    }

    // Early validation for co-located mode flags
    if args.colocated_worker && args.remote_kv_master_url.is_none() {
        anyhow::bail!("--colocated-worker requires --remote-kv-master-url");
    }
    if args.colocated_worker && args.worker_id.is_none() {
        anyhow::bail!("--colocated-worker requires --worker-id");
    }

    let dtype = args
        .dtype
        .as_deref()
        .map(flour::engine::parse_dtype)
        .transpose()?;
    tracing::info!("loading model from {}", args.model_dir.display());
    let mut engine = Engine::load(&args.model_dir, dtype)?;
    tracing::info!("model loaded: {}", engine.model_id());

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;

    // Co-located worker store, shared between the engine (in-process locality)
    // and the worker data-path routes (so peers can fetch our bytes).
    let mut colocated_store = None;
    if args.colocated_worker {
        let master_url = args
            .remote_kv_master_url
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--colocated-worker requires --remote-kv-master-url"))?;
        let worker_id = args
            .worker_id
            .clone()
            .ok_or_else(|| anyhow::anyhow!("--colocated-worker requires --worker-id"))?;
        let store = Arc::new(Mutex::new(WorkerStore::new(
            worker_id.clone(),
            0,
            args.distkv_capacity_bytes,
        )));
        engine.enable_remote_kv_colocated(master_url, worker_id.clone(), store.clone())?;
        tracing::info!("co-located worker '{worker_id}' enabled against master {master_url}");

        // Register + heartbeat the embedded worker in the background.
        let advertise = args
            .advertise_url
            .clone()
            .unwrap_or_else(|| format!("http://{}:{}", args.host, args.port));
        let req = RegisterRequest {
            worker_id,
            addr: advertise,
            capacity_bytes: args.distkv_capacity_bytes,
        };
        let master_url = master_url.to_string();
        tokio::spawn(async move {
            if let Err(e) = run_registration(reqwest::Client::new(), master_url, req).await {
                tracing::error!("co-located worker registration loop exited: {e}");
            }
        });
        colocated_store = Some(store);
    } else if args.remote_kv_enabled {
        let master_url = args.remote_kv_master_url.as_deref().ok_or_else(|| {
            anyhow::anyhow!("--remote-kv-enabled requires --remote-kv-master-url")
        })?;
        engine.enable_remote_kv(master_url)?;
        tracing::info!("remote KV cache enabled against master {master_url}");
    }

    let started_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let state = AppState {
        engine: Arc::new(Mutex::new(engine)),
        started_at,
    };

    // One server hosts the OpenAI API and, in co-located mode, the worker data path.
    let app = mount_distkv(router(state), None, colocated_store);
    serve_router(app, addr).await
}
