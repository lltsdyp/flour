use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use clap::Parser;
use flour::api::{serve, AppState};
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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    let dtype = args.dtype.as_deref().map(flour::engine::parse_dtype).transpose()?;
    tracing::info!("loading model from {}", args.model_dir.display());
    let engine = Engine::load(&args.model_dir, dtype)?;
    tracing::info!("model loaded: {}", engine.model_id());

    let started_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let state = AppState {
        engine: Arc::new(Mutex::new(engine)),
        started_at,
    };

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    serve(state, addr).await
}
