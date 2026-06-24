use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Context;
use candle_core::{DType, Device, IndexOp, Tensor};

use crate::distkv::worker::WorkerStore;
use crate::kv_cache::distkv_store::DistKvObjectStore;
use crate::kv_cache::local::LocalKvHandle;
use crate::kv_cache::manager::KvCacheManager;
use crate::loader::safetensors::load_var_builder;
use crate::models::{
    self,
    common::{Cache, CausalLM, EosTokenId},
};
use crate::sampling::{apply_repeat_penalty, LogitsSampler, SamplingParams};
use crate::tokenizer::{ChatMessage, ChatTemplate, Tokenizer};

/// Parse a dtype name into a `DType`. Accepts both short forms (`f32`, `bf16`, `f16`) and the
/// torch names found in `config.json` (`float32`, `bfloat16`, `float16`), case-insensitively.
pub fn parse_dtype(s: &str) -> anyhow::Result<DType> {
    match s.trim().to_ascii_lowercase().as_str() {
        "f32" | "float32" => Ok(DType::F32),
        "bf16" | "bfloat16" => Ok(DType::BF16),
        "f16" | "float16" => Ok(DType::F16),
        other => anyhow::bail!("unsupported dtype {other:?}; expected one of f32, bf16, f16"),
    }
}

/// Decide which dtype to load the model in. An explicit override always wins; otherwise fall back
/// to the model's `torch_dtype` from `config.json`, and finally to f32 if that is absent.
fn resolve_dtype(dtype_override: Option<DType>, raw: &serde_json::Value) -> anyhow::Result<DType> {
    if let Some(dtype) = dtype_override {
        return Ok(dtype);
    }
    match raw.get("torch_dtype").and_then(|v| v.as_str()) {
        Some(name) => parse_dtype(name),
        None => Ok(DType::F32),
    }
}

/// Why generation stopped. Maps to the OpenAI `finish_reason` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Hit an EOS token or a configured stop sequence.
    Stop,
    /// Reached the `max_tokens` budget without stopping naturally.
    Length,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
        }
    }
}

pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub reused_prefix_tokens: usize,
    pub finish_reason: FinishReason,
    /// Remote KV cache outcome for this request, when the remote cache is
    /// enabled: `Some(true)` on a remote hit, `Some(false)` on a miss (or a
    /// best-effort error), `None` when the remote cache is disabled.
    pub remote_cache_hit: Option<bool>,
    /// The prefix object key used for the remote cache, when enabled. Exposed
    /// for metrics and observability.
    pub remote_key: Option<String>,
}

pub struct Engine {
    model: CausalLM,
    tokenizer: Tokenizer,
    chat_template: ChatTemplate,
    eos_token_id: EosTokenId,
    device: Device,
    model_id: String,
    cache: std::sync::Mutex<Cache>,
    /// Unified KV cache layer. Owns prefix keying, remote hit/miss, fallback,
    /// and publish policy. Defaults to local-only; remote is strictly
    /// best-effort and never changes inference correctness.
    kv_cache: KvCacheManager,
}

impl Engine {
    pub fn load(model_dir: &Path, dtype_override: Option<DType>) -> anyhow::Result<Self> {
        let config_path = model_dir.join("config.json");
        let raw: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .with_context(|| format!("reading {}", config_path.display()))?,
        )?;
        let family = models::detect_family(&raw)?;
        let cfg = models::load_config(family, &raw)?;

        let dtype = resolve_dtype(dtype_override, &raw)?;
        tracing::info!("using dtype {dtype:?}");

        let device = Device::Cpu;
        let vb = load_var_builder(model_dir, dtype, &device)?;
        let model = CausalLM::load(vb, cfg.clone())?;
        let cache = std::sync::Mutex::new(Cache::new(&cfg, dtype, &device)?);

        let tokenizer = Tokenizer::from_file(&model_dir.join("tokenizer.json"))?;
        let chat_template = ChatTemplate::for_family(family);

        let eos_token_id = cfg.eos_token_id.clone().unwrap_or_else(|| {
            let fallback = match chat_template {
                ChatTemplate::Llama3 => "<|eot_id|>",
                ChatTemplate::ChatMl => "<|im_end|>",
            };
            EosTokenId::Single(tokenizer.token_to_id(fallback).unwrap_or(0))
        });

        let model_id = model_dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "model".to_string());

        Ok(Self {
            model,
            tokenizer,
            chat_template,
            eos_token_id,
            device,
            cache,
            kv_cache: KvCacheManager::local_only(model_id.clone()),
            model_id,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Enables the remote KV cache against the given Master URL. Best-effort:
    /// the Master need not be reachable now, and remote failures never abort
    /// generation.
    pub fn enable_remote_kv(&mut self, master_url: &str) -> anyhow::Result<()> {
        let store = Arc::new(DistKvObjectStore::connect(master_url)?);
        self.kv_cache = KvCacheManager::with_remote(self.model_id.clone(), store);
        Ok(())
    }

    /// Enables the remote KV cache in co-located mode: this node embeds the
    /// worker `worker_id` backed by `store`, so writes prefer it and local
    /// committed routes are read in-process. Best-effort, like `enable_remote_kv`.
    pub fn enable_remote_kv_colocated(
        &mut self,
        master_url: &str,
        worker_id: String,
        store: Arc<Mutex<WorkerStore>>,
    ) -> anyhow::Result<()> {
        let local = LocalKvHandle::new(worker_id, store);
        let store = Arc::new(DistKvObjectStore::connect_colocated(master_url, local)?);
        self.kv_cache = KvCacheManager::with_remote(self.model_id.clone(), store);
        Ok(())
    }

    pub fn generate(
        &self,
        messages: &[ChatMessage],
        params: &SamplingParams,
        mut on_token: impl FnMut(&str),
    ) -> anyhow::Result<GenerationStats> {
        let prompt = self.chat_template.render(messages);
        let prompt_tokens = self.tokenizer.encode(&prompt)?;
        let prompt_len = prompt_tokens.len();

        // Remote KV cache plan (best-effort): just the reusable prefix key and length. The actual
        // remote fetch + import happens inside `KvSession::prefill`, which holds the cache lock.
        let lookup = self.kv_cache.prepare(&prompt_tokens);
        let remote_key = lookup.remote_key.clone();

        let mut cache = self
            .cache
            .lock()
            .map_err(|_| anyhow::anyhow!("engine cache mutex poisoned"))?;
        let mut sampler = LogitsSampler::new(params.seed);
        let mut all_tokens = prompt_tokens.clone();

        let mut kv_session = self
            .kv_cache
            .bind_session(lookup, &mut cache, prompt_tokens.clone());
        let prefill = kv_session.prefill(&self.model, &self.device)?;
        let mut logits = prefill.logits;
        let reused_prefix_tokens = prefill.reused_prefix_tokens;
        // The hit/miss metric is known only after prefill performed (or skipped) the remote fetch.
        let remote_cache_hit = kv_session.remote_cache_hit();
        let mut completion_tokens = 0usize;

        tracing::info!(
            "Finished prefill, prompt tokens count: {}, reused prefix tokens count: {}",
            all_tokens.len(),
            reused_prefix_tokens
        );

        // Defaults to Length: if the loop runs to the token budget without an EOS
        // token or stop sequence, the completion was truncated.
        let mut finish_reason = FinishReason::Length;
        let mut generated = String::new();
        for index_pos in (prompt_len..).take(params.max_tokens) {
            let seq_len = logits.dim(1)?;
            let last = logits.i((0, seq_len - 1))?.to_dtype(DType::F32)?;
            let mut logits_vec = last.to_vec1::<f32>()?;

            if params.repeat_penalty != 1.0 {
                let start = all_tokens.len().saturating_sub(params.repeat_last_n);
                apply_repeat_penalty(&mut logits_vec, params.repeat_penalty, &all_tokens[start..]);
            }

            let next_token = sampler.sample(&logits_vec, params);
            if self.eos_token_id.is_eos(next_token) {
                finish_reason = FinishReason::Stop;
                break;
            }

            let piece = self.tokenizer.decode(&[next_token])?;

            // Stop-sequence check: a stop string may span several tokens, so test it
            // against the full decoded output. The stop text is counted as generated
            // but not emitted to the caller.
            if !params.stop.is_empty() {
                let candidate = format!("{generated}{piece}");
                if params
                    .stop
                    .iter()
                    .any(|s| !s.is_empty() && candidate.ends_with(s.as_str()))
                {
                    all_tokens.push(next_token);
                    completion_tokens += 1;
                    finish_reason = FinishReason::Stop;
                    break;
                }
            }

            generated.push_str(&piece);
            on_token(&piece);
            all_tokens.push(next_token);
            completion_tokens += 1;

            let next_input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            logits = self
                .model
                .forward(&next_input, index_pos, kv_session.cache_mut())?;
        }

        tracing::info!(
            "Generate finished, generated tokens count:{}",
            completion_tokens
        );

        // Collect publish state, then release the cache lock before any remote
        // network I/O. The publish is best-effort: a failure is logged and
        // ignored so inference is never blocked on the remote cache.
        let publish = kv_session.finish();
        drop(cache);
        self.kv_cache.publish_best_effort(publish);

        Ok(GenerationStats {
            prompt_tokens: prompt_len,
            completion_tokens,
            reused_prefix_tokens,
            finish_reason,
            remote_cache_hit,
            remote_key,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tokenizer::ChatMessage;
    use candle_core::{DType, Device, Tensor};
    use std::collections::HashMap;

    #[test]
    fn parse_dtype_accepts_short_and_torch_names() {
        assert_eq!(parse_dtype("f32").unwrap(), DType::F32);
        assert_eq!(parse_dtype("float32").unwrap(), DType::F32);
        assert_eq!(parse_dtype("bf16").unwrap(), DType::BF16);
        assert_eq!(parse_dtype("BFloat16").unwrap(), DType::BF16);
        assert_eq!(parse_dtype("f16").unwrap(), DType::F16);
        assert_eq!(parse_dtype("float16").unwrap(), DType::F16);
        assert!(parse_dtype("int8").is_err());
    }

    #[test]
    fn resolve_dtype_prefers_override_then_config() {
        let cfg: serde_json::Value = serde_json::json!({ "torch_dtype": "bfloat16" });
        // Explicit override wins over config.
        assert_eq!(resolve_dtype(Some(DType::F16), &cfg).unwrap(), DType::F16);
        // Otherwise fall back to config's torch_dtype.
        assert_eq!(resolve_dtype(None, &cfg).unwrap(), DType::BF16);
        // Missing torch_dtype defaults to f32.
        let empty = serde_json::json!({});
        assert_eq!(resolve_dtype(None, &empty).unwrap(), DType::F32);
    }

    pub(crate) fn fixture_dir_for_external_use() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        write_fixture_model(dir.path());
        dir
    }

    /// Builds a tiny but structurally valid Qwen3-family model directory: config.json,
    /// a minimal tokenizer.json (byte-level BPE over a handful of fixed tokens), and
    /// randomly-initialized safetensors weights matching that config.
    fn write_fixture_model(dir: &std::path::Path) {
        let hidden = 8usize;
        let heads = 2usize;
        let kv_heads = 1usize;
        let head_dim = 4usize;
        let intermediate = 16usize;
        let layers = 1usize;
        // 128 = full ASCII byte range, so every byte that can appear in a rendered ChatML
        // prompt (letters, digits, punctuation, "<|...|>", newlines) has a vocab entry.
        let vocab = 128usize;

        let config = serde_json::json!({
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "vocab_size": vocab,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": true,
            "max_position_embeddings": 64
        });
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_string(&config).unwrap(),
        )
        .unwrap();

        fn rand_tensor(shape: &[usize], seed: u64) -> Tensor {
            use rand::rngs::StdRng;
            use rand::{Rng, SeedableRng};
            let numel: usize = shape.iter().product();
            let mut rng = StdRng::seed_from_u64(seed);
            let data: Vec<f32> = (0..numel).map(|_| rng.gen_range(-0.05..0.05)).collect();
            Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
        }

        let size_q = head_dim * heads;
        let size_kv = head_dim * kv_heads;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert(
            "model.embed_tokens.weight".into(),
            rand_tensor(&[vocab, hidden], 1),
        );
        tensors.insert(
            "model.norm.weight".into(),
            Tensor::ones(hidden, DType::F32, &Device::Cpu).unwrap(),
        );
        for l in 0..layers {
            let p = format!("model.layers.{l}");
            tensors.insert(
                format!("{p}.input_layernorm.weight"),
                Tensor::ones(hidden, DType::F32, &Device::Cpu).unwrap(),
            );
            tensors.insert(
                format!("{p}.post_attention_layernorm.weight"),
                Tensor::ones(hidden, DType::F32, &Device::Cpu).unwrap(),
            );
            tensors.insert(
                format!("{p}.self_attn.q_proj.weight"),
                rand_tensor(&[size_q, hidden], 10),
            );
            tensors.insert(
                format!("{p}.self_attn.k_proj.weight"),
                rand_tensor(&[size_kv, hidden], 11),
            );
            tensors.insert(
                format!("{p}.self_attn.v_proj.weight"),
                rand_tensor(&[size_kv, hidden], 12),
            );
            tensors.insert(
                format!("{p}.self_attn.o_proj.weight"),
                rand_tensor(&[hidden, size_q], 13),
            );
            tensors.insert(
                format!("{p}.self_attn.q_norm.weight"),
                Tensor::ones(head_dim, DType::F32, &Device::Cpu).unwrap(),
            );
            tensors.insert(
                format!("{p}.self_attn.k_norm.weight"),
                Tensor::ones(head_dim, DType::F32, &Device::Cpu).unwrap(),
            );
            tensors.insert(
                format!("{p}.mlp.gate_proj.weight"),
                rand_tensor(&[intermediate, hidden], 14),
            );
            tensors.insert(
                format!("{p}.mlp.up_proj.weight"),
                rand_tensor(&[intermediate, hidden], 15),
            );
            tensors.insert(
                format!("{p}.mlp.down_proj.weight"),
                rand_tensor(&[hidden, intermediate], 16),
            );
        }
        candle_core::safetensors::save(&tensors, dir.join("model.safetensors")).unwrap();

        // Minimal byte-level tokenizer: vocab of single bytes 0..vocab, no merges, so any
        // input string encodes to one token per byte and round-trips exactly. The `tokenizers`
        // crate's ByteLevel pre-tokenizer remaps raw bytes through the GPT-2 byte-to-unicode
        // table before lookup (printable bytes map to themselves; everything else, including
        // all of 0..=32, maps into the U+0100+ range) — so the vocab keys must go through the
        // same mapping, not the raw byte values, or encoding silently produces zero tokens.
        fn bytes_to_unicode() -> HashMap<u8, char> {
            let mut bs: Vec<u8> = vec![];
            bs.extend(b'!'..=b'~');
            bs.extend(0xA1u8..=0xACu8);
            bs.extend(0xAEu8..=0xFFu8);
            let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
            let mut n = 0u32;
            for b in 0u8..=255 {
                if !bs.contains(&b) {
                    bs.push(b);
                    cs.push(256 + n);
                    n += 1;
                }
            }
            bs.into_iter()
                .zip(cs)
                .map(|(b, c)| (b, char::from_u32(c).unwrap()))
                .collect()
        }
        let byte_map = bytes_to_unicode();
        let mut vocab_map = serde_json::Map::new();
        for i in 0..vocab {
            let c = byte_map[&(i as u8)];
            vocab_map.insert(c.to_string(), serde_json::json!(i));
        }
        let tokenizer_json = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false },
            "post_processor": null,
            "decoder": { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false },
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "vocab": vocab_map,
                "merges": []
            }
        });
        std::fs::write(
            dir.join("tokenizer.json"),
            serde_json::to_string(&tokenizer_json).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn load_and_generate_end_to_end_with_tiny_random_model() {
        let dir = tempfile::tempdir().unwrap();
        write_fixture_model(dir.path());

        let engine = Engine::load(dir.path(), None).unwrap();
        assert_eq!(
            engine.model_id(),
            dir.path().file_name().unwrap().to_string_lossy()
        );

        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }];
        let params = crate::sampling::SamplingParams {
            max_tokens: 4,
            seed: 1,
            ..Default::default()
        };

        let mut produced = String::new();
        let stats = engine
            .generate(&messages, &params, |tok| produced.push_str(tok))
            .unwrap();

        assert!(stats.prompt_tokens > 0);
        assert!(stats.completion_tokens <= 4);
    }

    #[test]
    fn second_identical_prompt_reuses_prefix_and_is_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        write_fixture_model(dir.path());
        let engine = Engine::load(dir.path(), None).unwrap();
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }];
        let params = crate::sampling::SamplingParams {
            max_tokens: 4,
            seed: 7,
            temperature: 0.0,
            ..Default::default()
        };

        let mut first = String::new();
        let stats1 = engine
            .generate(&messages, &params, |tok| first.push_str(tok))
            .unwrap();
        // Cold cache: the first call computes the whole prompt.
        assert_eq!(stats1.reused_prefix_tokens, 0);

        let mut second = String::new();
        let stats2 = engine
            .generate(&messages, &params, |tok| second.push_str(tok))
            .unwrap();
        // Second identical prompt reuses at least one cached block, and output is unchanged.
        assert!(stats2.reused_prefix_tokens > 0);
        assert_eq!(first, second);
        assert_eq!(stats2.prompt_tokens, stats1.prompt_tokens);
    }

    #[test]
    fn generate_is_deterministic_for_a_fixed_seed() {
        let dir = tempfile::tempdir().unwrap();
        write_fixture_model(dir.path());
        let engine = Engine::load(dir.path(), None).unwrap();
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }];
        let params = crate::sampling::SamplingParams {
            max_tokens: 4,
            seed: 99,
            temperature: 0.0,
            ..Default::default()
        };

        let mut a = String::new();
        engine
            .generate(&messages, &params, |tok| a.push_str(tok))
            .unwrap();
        let mut b = String::new();
        engine
            .generate(&messages, &params, |tok| b.push_str(tok))
            .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn exhausting_max_tokens_reports_finish_reason_length() {
        let dir = tempfile::tempdir().unwrap();
        write_fixture_model(dir.path());
        let engine = Engine::load(dir.path(), None).unwrap();
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }];
        let params = crate::sampling::SamplingParams {
            max_tokens: 3,
            seed: 5,
            temperature: 0.0,
            ..Default::default()
        };

        let stats = engine.generate(&messages, &params, |_| {}).unwrap();
        // With no EOS or stop sequence hit, generation runs to the token budget.
        assert_eq!(stats.completion_tokens, 3);
        assert_eq!(stats.finish_reason, FinishReason::Length);
    }

    // --- Remote KV cache integration (Task 6) ---

    use crate::distkv::client::DistKvClient;
    use crate::distkv::http::{master_router, worker_router};
    use crate::distkv::master::MasterState;
    use crate::distkv::protocol::RegisterRequest;
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

    /// Brings up a Master + one registered Worker and returns the Master URL.
    async fn remote_cluster() -> String {
        let master = Arc::new(Mutex::new(MasterState::new(|| 1_000)));
        let master_url = spawn(master_router(master)).await;
        let worker = Arc::new(Mutex::new(WorkerStore::new("w1".into(), 1, 1 << 20)));
        let worker_url = spawn(worker_router(worker)).await;

        // Register the worker with its data-path URL.
        reqwest::Client::new()
            .post(format!("{master_url}/v1/distkv/workers/register"))
            .json(&crate::distkv::protocol::RegisterRequest {
                worker_id: "w1".into(),
                addr: worker_url,
                capacity_bytes: 1 << 20,
            })
            .send()
            .await
            .unwrap();
        master_url
    }

    fn loaded_engine() -> Engine {
        let dir = tests::fixture_dir_for_external_use();
        let engine = Engine::load(dir.path(), None).unwrap();
        std::mem::forget(dir);
        engine
    }

    fn hi() -> Vec<ChatMessage> {
        vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }]
    }

    /// Runs `generate` on a blocking thread (mirroring the server), so the
    /// engine's owned runtime can `block_on` the async remote calls.
    async fn run_generate(engine: Arc<Mutex<Engine>>) -> GenerationStats {
        let params = crate::sampling::SamplingParams {
            max_tokens: 4,
            seed: 7,
            temperature: 0.0,
            ..Default::default()
        };
        tokio::task::spawn_blocking(move || engine.lock().unwrap().generate(&hi(), &params, |_| {}))
            .await
            .unwrap()
            .unwrap()
    }

    /// Like `run_generate`, but also captures the decoded completion text so two engines'
    /// greedy outputs can be compared.
    async fn run_generate_capture(engine: Arc<Mutex<Engine>>) -> (GenerationStats, String) {
        let params = crate::sampling::SamplingParams {
            max_tokens: 4,
            seed: 7,
            temperature: 0.0,
            ..Default::default()
        };
        tokio::task::spawn_blocking(move || {
            let mut out = String::new();
            let stats = engine
                .lock()
                .unwrap()
                .generate(&hi(), &params, |t| out.push_str(t))
                .unwrap();
            (stats, out)
        })
        .await
        .unwrap()
    }

    /// End-to-end proof of real cross-node KV reuse: engine A cold-prefills and publishes a real
    /// bundle; engine B — a *separate* `Engine` with its own empty `Cache`, sharing only the
    /// DistKV cluster — imports that bundle, reports a remote hit with a non-empty reused prefix,
    /// and produces byte-identical greedy output. The two engines load from the same fixture dir
    /// so they agree on model weights and `model_id` (hence the prefix key), but never share a
    /// local `Cache`, so any reuse on B can only have come from the remote bundle.
    #[tokio::test(flavor = "multi_thread")]
    async fn cross_node_reuses_real_remote_kv() {
        let master_url = remote_cluster().await;
        let dir = tests::fixture_dir_for_external_use();

        // Engine A: cold-generate, publishing a real KV bundle to DistKV.
        let mut engine_a = Engine::load(dir.path(), None).unwrap();
        engine_a.enable_remote_kv(&master_url).unwrap();
        let engine_a = Arc::new(Mutex::new(engine_a));
        let (stats_a, out_a) = run_generate_capture(engine_a.clone()).await;
        assert_eq!(stats_a.remote_cache_hit, Some(false), "A starts cold");
        assert_eq!(stats_a.reused_prefix_tokens, 0, "A computed the whole prompt");

        // Engine B: independent engine + cache; only DistKV is shared.
        let mut engine_b = Engine::load(dir.path(), None).unwrap();
        engine_b.enable_remote_kv(&master_url).unwrap();
        let engine_b = Arc::new(Mutex::new(engine_b));
        let (stats_b, out_b) = run_generate_capture(engine_b.clone()).await;

        assert_eq!(
            stats_b.remote_cache_hit,
            Some(true),
            "B hits the remote KV bundle"
        );
        assert!(
            stats_b.reused_prefix_tokens > 0,
            "B imports a real prefix from the remote bundle"
        );
        assert_eq!(
            stats_a.remote_key, stats_b.remote_key,
            "same model + prompt => same prefix key"
        );
        // Correctness: B's imported-prefix generation matches A's cold reference exactly.
        assert_eq!(out_a, out_b, "imported KV must reproduce the cold output");

        std::mem::forget(dir);
        std::mem::forget(engine_a);
        std::mem::forget(engine_b);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn generation_succeeds_when_master_is_down() {
        let mut engine = loaded_engine();
        // Point at a port where nothing is listening.
        engine.enable_remote_kv("http://127.0.0.1:9").unwrap();
        let engine = Arc::new(Mutex::new(engine));

        let stats = run_generate(engine.clone()).await;
        // Generation still works; the failed remote lookup counts as a miss.
        assert!(stats.completion_tokens > 0);
        assert_eq!(stats.remote_cache_hit, Some(false));

        std::mem::forget(engine);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn generation_puts_remote_object_after_prefill_when_enabled() {
        let master_url = remote_cluster().await;
        let mut engine = loaded_engine();
        engine.enable_remote_kv(&master_url).unwrap();
        let engine = Arc::new(Mutex::new(engine));

        let stats = run_generate(engine.clone()).await;
        let key = stats.remote_key.clone().expect("remote enabled => key set");
        assert_eq!(stats.remote_cache_hit, Some(false), "cold cache is a miss");

        // The object must now be fetchable directly from the worker via a route.
        let client = DistKvClient::new(master_url);
        let bytes = client.get_object(&key).await.unwrap();
        assert!(
            bytes.is_some(),
            "engine should have stored the prefix bundle"
        );

        std::mem::forget(engine);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn second_request_observes_remote_cache_hit_metric() {
        let master_url = remote_cluster().await;
        let mut engine = loaded_engine();
        engine.enable_remote_kv(&master_url).unwrap();
        let engine = Arc::new(Mutex::new(engine));

        let first = run_generate(engine.clone()).await;
        assert_eq!(first.remote_cache_hit, Some(false));

        let second = run_generate(engine.clone()).await;
        assert_eq!(
            second.remote_cache_hit,
            Some(true),
            "second identical prompt should hit the remote cache"
        );
        assert_eq!(first.remote_key, second.remote_key);

        std::mem::forget(engine);
    }

    // --- Co-located KV cache: read/write locality ---

    /// Master + one worker registered under `worker_id`, advertising `advertise`.
    /// Pass an unroutable `advertise` to prove a path never used HTTP.
    async fn colocated_master(worker_id: &str, advertise: &str) -> String {
        let master = Arc::new(Mutex::new(MasterState::new(|| 1_000)));
        let master_url = spawn(master_router(master)).await;
        reqwest::Client::new()
            .post(format!("{master_url}/v1/distkv/workers/register"))
            .json(&RegisterRequest {
                worker_id: worker_id.into(),
                addr: advertise.into(),
                capacity_bytes: 1 << 20,
            })
            .send()
            .await
            .unwrap();
        master_url
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn colocated_put_writes_to_local_store_without_http() {
        let store = Arc::new(Mutex::new(WorkerStore::new("local".into(), 0, 1 << 20)));
        // Unroutable advertise: any HTTP write would fail the test.
        let master_url = colocated_master("local", "http://127.0.0.1:1").await;
        let mut engine = loaded_engine();
        engine
            .enable_remote_kv_colocated(&master_url, "local".into(), store.clone())
            .unwrap();
        let engine = Arc::new(Mutex::new(engine));

        let stats = run_generate(engine.clone()).await;
        let key = stats.remote_key.clone().expect("remote enabled => key set");

        // Bytes landed in the in-process store at generation 1, no HTTP involved.
        let held = store.lock().unwrap().get_bytes(&key, 1).unwrap();
        assert!(held.is_some(), "local store should hold the bundle");

        std::mem::forget(engine);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn colocated_get_hits_local_store_without_http() {
        let store = Arc::new(Mutex::new(WorkerStore::new("local".into(), 0, 1 << 20)));
        let master_url = colocated_master("local", "http://127.0.0.1:1").await;
        let mut engine = loaded_engine();
        engine
            .enable_remote_kv_colocated(&master_url, "local".into(), store.clone())
            .unwrap();
        let engine = Arc::new(Mutex::new(engine));

        let first = run_generate(engine.clone()).await;
        assert_eq!(first.remote_cache_hit, Some(false));
        // Second identical prompt hits the cache via an in-process read (the
        // worker addr is unroutable, so a hit can only come from the local store).
        let second = run_generate(engine.clone()).await;
        assert_eq!(second.remote_cache_hit, Some(true));

        std::mem::forget(engine);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn colocated_second_request_uses_committed_local_fast_path() {
        let store = Arc::new(Mutex::new(WorkerStore::new("local".into(), 0, 1 << 20)));
        let master_url = colocated_master("local", "http://127.0.0.1:1").await;
        let mut engine = loaded_engine();
        engine
            .enable_remote_kv_colocated(&master_url, "local".into(), store.clone())
            .unwrap();
        let engine = Arc::new(Mutex::new(engine));

        let first = run_generate(engine.clone()).await;
        assert_eq!(first.remote_cache_hit, Some(false));

        let second = run_generate(engine.clone()).await;
        assert_eq!(second.remote_cache_hit, Some(true));
        assert_eq!(first.remote_key, second.remote_key);

        std::mem::forget(engine);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn remote_route_still_uses_http() {
        // Local store exists but is NOT registered with the Master; the only
        // registered worker is a real remote HTTP worker. The route therefore
        // points remote and must be served over HTTP.
        let local = Arc::new(Mutex::new(WorkerStore::new("local".into(), 0, 1 << 20)));
        let master_url = remote_cluster().await; // registers a real worker "w1"
        let mut engine = loaded_engine();
        engine
            .enable_remote_kv_colocated(&master_url, "local".into(), local.clone())
            .unwrap();
        let engine = Arc::new(Mutex::new(engine));

        let first = run_generate(engine.clone()).await;
        assert_eq!(first.remote_cache_hit, Some(false));
        let second = run_generate(engine.clone()).await;
        assert_eq!(
            second.remote_cache_hit,
            Some(true),
            "served over HTTP by w1"
        );
        // The local store was never the route target, so it stays empty.
        assert_eq!(local.lock().unwrap().used_bytes(), 0);

        std::mem::forget(engine);
    }

    #[test]
    fn stop_sequence_halts_generation_with_finish_reason_stop() {
        let dir = tempfile::tempdir().unwrap();
        write_fixture_model(dir.path());
        let engine = Engine::load(dir.path(), None).unwrap();
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }];
        let base = crate::sampling::SamplingParams {
            max_tokens: 8,
            seed: 5,
            temperature: 0.0,
            ..Default::default()
        };

        // Capture the deterministic first emitted piece, then make it a stop sequence.
        let mut first_piece = String::new();
        engine
            .generate(&messages, &base, |tok| {
                if first_piece.is_empty() {
                    first_piece.push_str(tok);
                }
            })
            .unwrap();
        assert!(
            !first_piece.is_empty(),
            "model emitted no tokens to stop on"
        );

        let params = crate::sampling::SamplingParams {
            stop: vec![first_piece.clone()],
            ..base
        };
        let mut emitted = String::new();
        let stats = engine
            .generate(&messages, &params, |tok| emitted.push_str(tok))
            .unwrap();

        assert_eq!(stats.finish_reason, FinishReason::Stop);
        // The stop sequence itself is not emitted to the caller.
        assert!(!emitted.contains(&first_piece));
        assert!(stats.completion_tokens < 8);
    }
}
