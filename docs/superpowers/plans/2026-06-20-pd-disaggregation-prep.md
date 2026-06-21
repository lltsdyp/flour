# Prefill/Decode Disaggregation Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the monolithic `Engine::generate` into a `PrefillWorker` and a `DecodeWorker` that exchange a serializable KV-cache bundle through a transport seam, so a future deployment can run prefill and decode on separate nodes.

**Architecture:** Prefill renders + tokenizes the prompt, runs the prompt forward pass, and produces a `KvCacheBundle` (per-layer K/V tensors) plus the last-position logits and the prompt token ids. That payload is serialized to a safetensors byte buffer and handed across a `KvCacheTransport` trait (in-process `LocalTransport` for now; network impl slots in later). Decode reconstructs a `Cache` from the bundle — rebuilding RoPE/mask tables locally from `Config` so only KV crosses the wire — and owns the entire sampling/detokenization loop including the first sampled token. `Engine::generate` keeps its exact public signature and behavior by composing the two workers in-process.

**Tech Stack:** Rust 2021, candle-core/candle-nn 0.9.2, safetensors 0.7, anyhow, tokio/axum (unchanged).

## Global Constraints

- Rust edition: `2021` (matches `Cargo.toml`).
- Tensor dtype for all KV/compute is `candle_core::DType::F32`; device is `candle_core::Device::Cpu`.
- Do NOT change the public signature or observed behavior of `Engine::load`, `Engine::model_id`, or `Engine::generate`. The two existing `engine.rs` tests (`load_and_generate_end_to_end_with_tiny_random_model`, `generate_is_deterministic_for_a_fixed_seed`) are the regression guard and must keep passing unchanged.
- KV-cache shapes are `[batch, num_key_value_heads, seq_len, head_dim]` (candle layout used by `Cache::append_kv`). `seq_len` is recoverable from tensor dim 2 — never ship it as a separate required field.
- **RoPE/QK-norm are baked into the cached K.** `attention.rs:96-101` applies `rope` (and `q_norm`/`k_norm` before that) to K *before* `cache.append_kv`. The transferred KV is therefore already position-encoded — `into_cache` must NEVER re-apply RoPE to it (doing so double-rotates and silently corrupts output). RoPE `cos`/`sin` tables and causal masks are pure functions of `Config` and are rebuilt locally by `Cache::new`, never transferred.
- **Position continuity:** decode must resume RoPE at `index_pos = prompt_len` (== `prompt_tokens.len()`), aligning with the cache's existing `kv_seq_len`. This is the second reason `prompt_tokens` is shipped (beyond repeat-penalty).
- **Config-identity invariant:** prefill and decode nodes MUST load the identical model config (`rope_theta`, `head_dim`, `max_seq_len`, layer/head counts). Locally rebuilt RoPE tables are only bit-identical to the prefill node's when these match; a mismatch corrupts results silently. A future network transport should validate a config fingerprint before accepting a bundle (out of scope here; noted for later).
- Networking is explicitly OUT OF SCOPE. Build the seam (trait + local impl + byte serialization), not the network transport.
- Run `cargo fmt` and `cargo clippy --all-targets -- -D warnings` clean before each commit (the repo's commit `8d32381` established this bar).
- Sampling stays entirely on the decode side. Prefill never constructs a `LogitsSampler`.

---

## File Structure

- `Cargo.toml` — add `safetensors = "0.7"` as a direct dependency (candle already resolves the same 0.7.0; this gives us `safetensors::serialize`).
- `src/lib.rs` — declare `pub mod disagg;`.
- `src/models/common/cache.rs` — add `export_kv` / `install_kv` accessors so a `Cache`'s KV state can leave and re-enter the struct. One responsibility: the KV/RoPE/mask cache.
- `src/disagg/mod.rs` — module root; defines `GenerationStats`; re-exports the worker/bundle/transport types.
- `src/disagg/bundle.rs` — `KvCacheBundle`: the transferable KV representation + safetensors byte (de)serialization. Pure data + (de)serialization, no model/forward logic.
- `src/disagg/transport.rs` — `TransferPayload`, `KvCacheTransport` trait, `LocalTransport`. The node-boundary contract.
- `src/disagg/prefill.rs` — `PrefillWorker` + `PrefillOutput`. Owns prompt → KV bundle + last logits.
- `src/disagg/decode.rs` — `DecodeWorker` + `DecodeInput`. Owns bundle → sampled tokens.
- `src/engine.rs` — refactor `Engine` to compose `PrefillWorker` + `DecodeWorker`; re-export `GenerationStats`; keep public API identical.

---

### Task 1: Cache KV export/install accessors

Add the two methods that let a `Cache`'s per-layer KV tensors be extracted (on the prefill node) and re-installed into a freshly constructed `Cache` (on the decode node). RoPE/mask tables are NOT exported — they are rebuilt by `Cache::new` from `Config`.

**Files:**
- Modify: `src/models/common/cache.rs` (add two methods to `impl Cache`, after `append_kv` ending at line 85)
- Test: `src/models/common/cache.rs` (extend the existing `#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: existing `Cache { kvs: Vec<Option<(Tensor, Tensor)>>, .. }`, `Cache::new(cfg: &Config, device: &Device) -> Result<Self>`, `Cache::append_kv`.
- Produces:
  - `Cache::export_kv(&self) -> candle_core::Result<Vec<(Tensor, Tensor)>>` — one contiguous `(k, v)` per layer, in layer order; errors if any layer is un-prefilled (`None`).
  - `Cache::install_kv(&mut self, layers: Vec<(Tensor, Tensor)>) -> candle_core::Result<()>` — replaces `kvs` with the provided layers; errors if `layers.len()` != number of layers.

- [ ] **Step 1: Write the failing tests**

Add to the `mod tests` block in `src/models/common/cache.rs`:

```rust
    #[test]
    fn export_kv_errors_when_a_layer_is_unprefilled() {
        let cache = Cache::new(&test_config(), &Device::Cpu).unwrap();
        // Fresh cache has all layers None -> export must fail, not panic.
        assert!(cache.export_kv().is_err());
    }

    #[test]
    fn export_then_install_round_trips_kv_tensors() {
        let cfg = test_config();
        let mut cache = Cache::new(&cfg, &Device::Cpu).unwrap();
        // Prefill both layers with distinguishable shapes.
        for layer in 0..cfg.num_hidden_layers {
            let k =
                Tensor::zeros((1, cfg.num_key_value_heads, 3, cfg.head_dim), DType::F32, &Device::Cpu)
                    .unwrap();
            let v = k.clone();
            cache.append_kv(layer, k, v).unwrap();
        }

        let exported = cache.export_kv().unwrap();
        assert_eq!(exported.len(), cfg.num_hidden_layers);
        assert_eq!(exported[0].0.dims(), &[1, cfg.num_key_value_heads, 3, cfg.head_dim]);

        let mut fresh = Cache::new(&cfg, &Device::Cpu).unwrap();
        fresh.install_kv(exported).unwrap();
        let re = fresh.export_kv().unwrap();
        assert_eq!(re[0].0.dims(), &[1, cfg.num_key_value_heads, 3, cfg.head_dim]);
    }

    #[test]
    fn install_kv_rejects_wrong_layer_count() {
        let cfg = test_config();
        let mut cache = Cache::new(&cfg, &Device::Cpu).unwrap();
        let k = Tensor::zeros((1, cfg.num_key_value_heads, 1, cfg.head_dim), DType::F32, &Device::Cpu)
            .unwrap();
        // test_config has 2 layers; supplying 1 must error.
        assert!(cache.install_kv(vec![(k.clone(), k)]).is_err());
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p flour models::common::cache 2>&1 | tail -20`
Expected: FAIL — `no method named export_kv`/`install_kv found for struct Cache`.

- [ ] **Step 3: Implement the accessors**

Insert into `impl Cache` in `src/models/common/cache.rs`, immediately after the closing `}` of `append_kv` (line 85):

```rust
    /// Extract a contiguous `(k, v)` per layer for transfer to another node. Errors if any
    /// layer has not been prefilled yet. RoPE/mask tables are intentionally not exported —
    /// the receiving node rebuilds them from `Config` via `Cache::new`.
    pub fn export_kv(&self) -> Result<Vec<(Tensor, Tensor)>> {
        let mut out = Vec::with_capacity(self.kvs.len());
        for (i, slot) in self.kvs.iter().enumerate() {
            let (k, v) = slot.as_ref().ok_or_else(|| {
                candle_core::Error::Msg(format!("export_kv: layer {i} has no cached KV"))
            })?;
            out.push((k.contiguous()?, v.contiguous()?));
        }
        Ok(out)
    }

    /// Install previously exported KV tensors into a freshly constructed cache. The layer count
    /// must match this cache's configured number of layers.
    pub fn install_kv(&mut self, layers: Vec<(Tensor, Tensor)>) -> Result<()> {
        if layers.len() != self.kvs.len() {
            return Err(candle_core::Error::Msg(format!(
                "install_kv: expected {} layers, got {}",
                self.kvs.len(),
                layers.len()
            )));
        }
        self.kvs = layers.into_iter().map(Some).collect();
        Ok(())
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p flour models::common::cache 2>&1 | tail -20`
Expected: PASS — all cache tests including the three new ones.

- [ ] **Step 5: Commit**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings
git add src/models/common/cache.rs
git commit -m "feat(cache): add export_kv/install_kv for cross-node KV transfer"
```

---

### Task 2: KvCacheBundle in-memory (from_cache / into_cache)

Create the `disagg` module and the `KvCacheBundle` value type that carries KV tensors between workers in-process. Wire serialization comes in Task 3.

**Files:**
- Modify: `Cargo.toml` (add `safetensors = "0.7"` under `[dependencies]`)
- Modify: `src/lib.rs` (add `pub mod disagg;`)
- Create: `src/disagg/mod.rs`
- Create: `src/disagg/bundle.rs`
- Test: `src/disagg/bundle.rs` (inline `#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: `Cache::export_kv`, `Cache::install_kv`, `Cache::new` (Task 1); `crate::models::common::{Cache, Config}`.
- Produces:
  - `struct KvCacheBundle { layers: Vec<(Tensor, Tensor)> }` (field private to the module)
  - `KvCacheBundle::from_cache(cache: &Cache) -> anyhow::Result<Self>`
  - `KvCacheBundle::into_cache(self, cfg: &Config, device: &Device) -> anyhow::Result<Cache>`
  - `KvCacheBundle::num_layers(&self) -> usize`
  - `KvCacheBundle::seq_len(&self) -> anyhow::Result<usize>`
  - `struct GenerationStats { pub prompt_tokens: usize, pub completion_tokens: usize }` (defined in `disagg/mod.rs`; replaces the one currently in `engine.rs`)

- [ ] **Step 1: Add the dependency and module declarations**

In `Cargo.toml`, under `[dependencies]`, add after the `uuid` line:

```toml
safetensors = "0.7"
```

In `src/lib.rs`, add after `pub mod backend;`:

```rust
pub mod disagg;
```

- [ ] **Step 2: Create the module root with GenerationStats**

Create `src/disagg/mod.rs`:

```rust
//! Prefill/Decode disaggregation: workers, a transferable KV-cache bundle, and the
//! transport seam that lets prefill and decode run on separate nodes.

pub mod bundle;
pub mod decode;
pub mod prefill;
pub mod transport;

pub use bundle::KvCacheBundle;
pub use decode::{DecodeInput, DecodeWorker};
pub use prefill::{PrefillOutput, PrefillWorker};
pub use transport::{KvCacheTransport, LocalTransport, TransferPayload};

/// Token accounting returned by a completed generation.
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}
```

NOTE: `decode`, `prefill`, and `transport` modules are created in later tasks. To compile Task 2 in isolation, temporarily comment out the `pub mod decode/prefill/transport;` and matching `pub use` lines, OR create empty stub files. The simplest path: create the three files now as empty placeholders so the module tree resolves:

```bash
printf '' > src/disagg/transport.rs
printf '' > src/disagg/prefill.rs
printf '' > src/disagg/decode.rs
```

Then in `mod.rs` keep only the lines that resolve. For this task, replace the `pub use` block with just:

```rust
pub use bundle::KvCacheBundle;
```

and comment the others back in as their tasks land. (Tasks 4–6 restore them.)

- [ ] **Step 3: Write the failing test**

Create `src/disagg/bundle.rs` with the test first:

```rust
use std::collections::HashMap;

use anyhow::Context;
use candle_core::{Device, Tensor};

use crate::models::common::{Cache, Config};

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    fn test_config() -> Config {
        Config {
            hidden_size: 8,
            intermediate_size: 16,
            vocab_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            max_seq_len: 16,
            use_qkv_bias: false,
            use_qk_norm: false,
            tie_word_embeddings: false,
            eos_token_id: None,
        }
    }

    fn prefilled_cache(cfg: &Config, seq: usize) -> Cache {
        let mut cache = Cache::new(cfg, &Device::Cpu).unwrap();
        for layer in 0..cfg.num_hidden_layers {
            let shape = (1, cfg.num_key_value_heads, seq, cfg.head_dim);
            let k = Tensor::ones(shape, DType::F32, &Device::Cpu).unwrap();
            let v = Tensor::ones(shape, DType::F32, &Device::Cpu).unwrap();
            cache.append_kv(layer, k, v).unwrap();
        }
        cache
    }

    #[test]
    fn from_cache_captures_layers_and_seq_len() {
        let cfg = test_config();
        let cache = prefilled_cache(&cfg, 5);
        let bundle = KvCacheBundle::from_cache(&cache).unwrap();
        assert_eq!(bundle.num_layers(), cfg.num_hidden_layers);
        assert_eq!(bundle.seq_len().unwrap(), 5);
    }

    #[test]
    fn into_cache_rebuilds_an_equivalent_cache() {
        let cfg = test_config();
        let cache = prefilled_cache(&cfg, 5);
        let bundle = KvCacheBundle::from_cache(&cache).unwrap();
        let rebuilt = bundle.into_cache(&cfg, &Device::Cpu).unwrap();
        // Re-exporting from the rebuilt cache yields the same shapes.
        let exported = rebuilt.export_kv().unwrap();
        assert_eq!(exported.len(), cfg.num_hidden_layers);
        assert_eq!(exported[0].0.dims(), &[1, cfg.num_key_value_heads, 5, cfg.head_dim]);
    }
}
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cargo test -p flour disagg::bundle 2>&1 | tail -20`
Expected: FAIL — `cannot find type KvCacheBundle` / unresolved import.

- [ ] **Step 5: Implement KvCacheBundle (in-memory parts)**

Add above the `#[cfg(test)]` block in `src/disagg/bundle.rs` (keep the existing `use` lines at the top of the file):

```rust
/// A transferable snapshot of a model's KV cache: one `(key, value)` tensor pair per layer,
/// each contiguous and shaped `[batch, num_key_value_heads, seq_len, head_dim]`. RoPE/mask
/// tables are deliberately excluded — the receiver rebuilds them from `Config`.
pub struct KvCacheBundle {
    layers: Vec<(Tensor, Tensor)>,
}

impl KvCacheBundle {
    /// Snapshot a prefilled cache. Errors if the cache has un-prefilled layers.
    pub fn from_cache(cache: &Cache) -> anyhow::Result<Self> {
        let layers = cache.export_kv().context("exporting KV from cache")?;
        Ok(Self { layers })
    }

    /// Rebuild a usable `Cache` on the receiving node: fresh RoPE/mask tables from `cfg`,
    /// with the transferred KV installed.
    pub fn into_cache(self, cfg: &Config, device: &Device) -> anyhow::Result<Cache> {
        let mut cache = Cache::new(cfg, device).context("allocating cache for transferred KV")?;
        cache.install_kv(self.layers).context("installing transferred KV")?;
        Ok(cache)
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Number of cached sequence positions, read from the KV tensor layout (dim 2).
    pub fn seq_len(&self) -> anyhow::Result<usize> {
        let (k, _) = self
            .layers
            .first()
            .context("KvCacheBundle has no layers")?;
        Ok(k.dims()[2])
    }
}
```

The top-of-file `use std::collections::HashMap;` is unused until Task 3 — add `#[allow(unused_imports)]` to it for now, or omit it until Task 3 (recommended: omit `HashMap` import here, add it in Task 3).

- [ ] **Step 6: Run test to verify it passes**

Run: `cargo test -p flour disagg::bundle 2>&1 | tail -20`
Expected: PASS — both bundle tests.

- [ ] **Step 7: Commit**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings
git add Cargo.toml Cargo.lock src/lib.rs src/disagg/
git commit -m "feat(disagg): add KvCacheBundle for in-process KV transfer"
```

---

### Task 3: KvCacheBundle wire serialization (to_bytes / from_bytes)

Add safetensors byte (de)serialization so the bundle can cross a real node boundary. Serialize each layer's K/V as named tensors (`layer.{i}.k`, `layer.{i}.v`) into a single safetensors buffer; deserialize with candle's `load_buffer`.

**Files:**
- Modify: `src/disagg/bundle.rs` (add two methods + test)

**Interfaces:**
- Consumes: `safetensors::serialize`, `candle_core::safetensors::load_buffer`. `&Tensor` implements `safetensors::View` (candle 0.9.2, `candle-core/src/safetensors.rs:87`), so it can be passed directly to `serialize`.
- Produces:
  - `KvCacheBundle::to_bytes(&self) -> anyhow::Result<Vec<u8>>`
  - `KvCacheBundle::from_bytes(data: &[u8], device: &Device) -> anyhow::Result<Self>`

- [ ] **Step 1: Write the failing test**

Add inside the existing `mod tests` in `src/disagg/bundle.rs`:

```rust
    #[test]
    fn bytes_round_trip_preserves_layers_and_seq_len() {
        let cfg = test_config();
        let cache = prefilled_cache(&cfg, 7);
        let bundle = KvCacheBundle::from_cache(&cache).unwrap();

        let bytes = bundle.to_bytes().unwrap();
        assert!(!bytes.is_empty());

        let restored = KvCacheBundle::from_bytes(&bytes, &Device::Cpu).unwrap();
        assert_eq!(restored.num_layers(), cfg.num_hidden_layers);
        assert_eq!(restored.seq_len().unwrap(), 7);

        // Values survive the round-trip: ones in, ones out.
        let restored_cache = restored.into_cache(&cfg, &Device::Cpu).unwrap();
        let (k, _) = &restored_cache.export_kv().unwrap()[0];
        let sum = k.sum_all().unwrap().to_scalar::<f32>().unwrap();
        let expected = (1 * cfg.num_key_value_heads * 7 * cfg.head_dim) as f32;
        assert!((sum - expected).abs() < 1e-3);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p flour disagg::bundle::tests::bytes_round_trip 2>&1 | tail -20`
Expected: FAIL — `no method named to_bytes found`.

- [ ] **Step 3: Implement to_bytes / from_bytes**

At the top of `src/disagg/bundle.rs`, ensure these imports are present:

```rust
use std::collections::HashMap;

use anyhow::{anyhow, Context};
use candle_core::{Device, Tensor};
```

Add these methods to `impl KvCacheBundle` (after `seq_len`):

```rust
    /// Serialize all layers into a single safetensors byte buffer. Tensors are keyed
    /// `layer.{i}.k` / `layer.{i}.v`; a small metadata header records the format version.
    pub fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        let mut data: Vec<(String, &Tensor)> = Vec::with_capacity(self.layers.len() * 2);
        for (i, (k, v)) in self.layers.iter().enumerate() {
            data.push((format!("layer.{i}.k"), k));
            data.push((format!("layer.{i}.v"), v));
        }
        let mut meta = HashMap::new();
        meta.insert("format".to_string(), "flour-kv-v1".to_string());
        meta.insert("num_layers".to_string(), self.layers.len().to_string());
        safetensors::serialize(data, Some(meta))
            .map_err(|e| anyhow!("serializing KV bundle: {e}"))
    }

    /// Reconstruct a bundle from a safetensors byte buffer produced by `to_bytes`.
    pub fn from_bytes(data: &[u8], device: &Device) -> anyhow::Result<Self> {
        let map = candle_core::safetensors::load_buffer(data, device)
            .context("loading KV bundle buffer")?;
        if map.len() % 2 != 0 {
            return Err(anyhow!(
                "KV bundle has {} tensors, expected an even count (k+v per layer)",
                map.len()
            ));
        }
        let num_layers = map.len() / 2;
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let k = map
                .get(&format!("layer.{i}.k"))
                .ok_or_else(|| anyhow!("KV bundle missing layer.{i}.k"))?
                .clone();
            let v = map
                .get(&format!("layer.{i}.v"))
                .ok_or_else(|| anyhow!("KV bundle missing layer.{i}.v"))?
                .clone();
            layers.push((k, v));
        }
        Ok(Self { layers })
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p flour disagg::bundle 2>&1 | tail -20`
Expected: PASS — all three bundle tests.

- [ ] **Step 5: Commit**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings
git add src/disagg/bundle.rs
git commit -m "feat(disagg): safetensors wire (de)serialization for KvCacheBundle"
```

---

### Task 4: Transport seam (KvCacheTransport + TransferPayload + LocalTransport)

Define what crosses the node boundary (`TransferPayload`) and the trait a transport implements (`KvCacheTransport`), plus an in-process `LocalTransport` so the seam is exercised end-to-end without networking.

**Files:**
- Create/replace: `src/disagg/transport.rs` (was an empty placeholder from Task 2)
- Modify: `src/disagg/mod.rs` (restore the `pub mod transport;` + `pub use transport::...` lines)
- Test: `src/disagg/transport.rs` (inline tests)

**Interfaces:**
- Consumes: nothing from other tasks (pure data + std).
- Produces:
  - `struct TransferPayload { pub bundle_bytes: Vec<u8>, pub last_logits: Vec<f32>, pub prompt_tokens: Vec<u32> }`
  - `trait KvCacheTransport { fn send(&self, payload: TransferPayload) -> anyhow::Result<()>; fn recv(&self) -> anyhow::Result<TransferPayload>; }`
  - `struct LocalTransport` with `LocalTransport::new() -> Self` implementing `KvCacheTransport`.

- [ ] **Step 1: Restore module wiring**

In `src/disagg/mod.rs`, ensure these lines are present (uncomment if they were commented in Task 2):

```rust
pub mod transport;
pub use transport::{KvCacheTransport, LocalTransport, TransferPayload};
```

- [ ] **Step 2: Write the failing test**

Replace the empty `src/disagg/transport.rs` with:

```rust
use std::collections::VecDeque;
use std::sync::Mutex;

/// Everything that must cross the prefill -> decode node boundary for one request:
/// the serialized KV cache, the last-position logits the decoder samples its first token
/// from, and the prompt token ids (needed for repeat-penalty over the prompt).
pub struct TransferPayload {
    pub bundle_bytes: Vec<u8>,
    pub last_logits: Vec<f32>,
    pub prompt_tokens: Vec<u32>,
}

/// The seam between prefill and decode. A networked implementation will frame and ship
/// `TransferPayload` over a socket; `LocalTransport` keeps it in-process.
pub trait KvCacheTransport {
    fn send(&self, payload: TransferPayload) -> anyhow::Result<()>;
    fn recv(&self) -> anyhow::Result<TransferPayload>;
}

/// In-process FIFO transport. `recv` is non-blocking and errors when empty — sufficient for
/// colocated tests and as the reference impl; a network transport will block instead.
pub struct LocalTransport {
    queue: Mutex<VecDeque<TransferPayload>>,
}

impl LocalTransport {
    pub fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
        }
    }
}

impl Default for LocalTransport {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_transport_round_trips_a_payload() {
        let t = LocalTransport::new();
        t.send(TransferPayload {
            bundle_bytes: vec![1, 2, 3],
            last_logits: vec![0.5, -0.5],
            prompt_tokens: vec![7, 8, 9],
        })
        .unwrap();

        let got = t.recv().unwrap();
        assert_eq!(got.bundle_bytes, vec![1, 2, 3]);
        assert_eq!(got.last_logits, vec![0.5, -0.5]);
        assert_eq!(got.prompt_tokens, vec![7, 8, 9]);
    }

    #[test]
    fn recv_on_empty_transport_errors() {
        let t = LocalTransport::new();
        assert!(t.recv().is_err());
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p flour disagg::transport 2>&1 | tail -20`
Expected: FAIL — `KvCacheTransport` not implemented for `LocalTransport` (the `impl` block is missing).

- [ ] **Step 4: Implement the trait for LocalTransport**

Add to `src/disagg/transport.rs`, after the `impl Default` block:

```rust
impl KvCacheTransport for LocalTransport {
    fn send(&self, payload: TransferPayload) -> anyhow::Result<()> {
        self.queue
            .lock()
            .map_err(|_| anyhow::anyhow!("LocalTransport queue mutex poisoned"))?
            .push_back(payload);
        Ok(())
    }

    fn recv(&self) -> anyhow::Result<TransferPayload> {
        self.queue
            .lock()
            .map_err(|_| anyhow::anyhow!("LocalTransport queue mutex poisoned"))?
            .pop_front()
            .ok_or_else(|| anyhow::anyhow!("LocalTransport: no payload available"))
    }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p flour disagg::transport 2>&1 | tail -20`
Expected: PASS — both transport tests.

- [ ] **Step 6: Commit**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings
git add src/disagg/transport.rs src/disagg/mod.rs
git commit -m "feat(disagg): KvCacheTransport seam with in-process LocalTransport"
```

---

### Task 5: PrefillWorker

The prefill half: render + tokenize the prompt, run the prompt forward pass, and produce a `PrefillOutput` (KV bundle + last-position logits + prompt token ids). No sampling here.

**Files:**
- Create/replace: `src/disagg/prefill.rs` (was an empty placeholder from Task 2)
- Modify: `src/disagg/mod.rs` (restore `pub mod prefill;` + `pub use prefill::...`)
- Test: `src/disagg/prefill.rs` (inline test, reusing the engine fixture model)

**Interfaces:**
- Consumes: `KvCacheBundle::from_cache` (Task 2); `TransferPayload` (Task 4); `crate::models::common::{Cache, CausalLM}`; `crate::tokenizer::{ChatMessage, ChatTemplate, Tokenizer}`; `crate::engine::tests::fixture_dir_for_external_use` (test only).
- Produces:
  - `struct PrefillOutput { pub bundle: KvCacheBundle, pub last_logits: Vec<f32>, pub prompt_tokens: Vec<u32> }`
  - `struct PrefillWorker { model: Arc<CausalLM>, tokenizer: Arc<Tokenizer>, chat_template: ChatTemplate, device: Device }`
  - `PrefillWorker::new(model: Arc<CausalLM>, tokenizer: Arc<Tokenizer>, chat_template: ChatTemplate, device: Device) -> Self`
  - `PrefillWorker::prefill(&self, messages: &[ChatMessage]) -> anyhow::Result<PrefillOutput>`
  - `PrefillWorker::prefill_to_payload(&self, messages: &[ChatMessage]) -> anyhow::Result<TransferPayload>`

- [ ] **Step 1: Restore module wiring**

In `src/disagg/mod.rs`, ensure present:

```rust
pub mod prefill;
pub use prefill::{PrefillOutput, PrefillWorker};
```

- [ ] **Step 2: Write the failing test**

Create `src/disagg/prefill.rs`:

```rust
use std::sync::Arc;

use anyhow::Context;
use candle_core::{DType, Device, IndexOp, Tensor};

use crate::disagg::bundle::KvCacheBundle;
use crate::disagg::transport::TransferPayload;
use crate::models::common::{Cache, CausalLM};
use crate::tokenizer::{ChatMessage, ChatTemplate, Tokenizer};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::safetensors::load_var_builder;
    use crate::models::{self};

    fn build_worker(dir: &std::path::Path) -> PrefillWorker {
        let raw: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
                .unwrap();
        let family = models::detect_family(&raw).unwrap();
        let cfg = models::load_config(family, &raw).unwrap();
        let device = Device::Cpu;
        let vb = load_var_builder(dir, DType::F32, &device).unwrap();
        let model = Arc::new(CausalLM::load(vb, cfg).unwrap());
        let tokenizer = Arc::new(Tokenizer::from_file(&dir.join("tokenizer.json")).unwrap());
        let chat_template = ChatTemplate::for_family(family);
        PrefillWorker::new(model, tokenizer, chat_template, device)
    }

    #[test]
    fn prefill_produces_bundle_logits_and_prompt_tokens() {
        let dir = crate::engine::tests::fixture_dir_for_external_use();
        let worker = build_worker(dir.path());
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }];

        let out = worker.prefill(&messages).unwrap();
        assert!(!out.prompt_tokens.is_empty());
        assert_eq!(out.bundle.seq_len().unwrap(), out.prompt_tokens.len());
        // last_logits is one row over the vocab (fixture vocab_size = 128).
        assert_eq!(out.last_logits.len(), 128);
    }

    #[test]
    fn prefill_to_payload_serializes_the_bundle() {
        let dir = crate::engine::tests::fixture_dir_for_external_use();
        let worker = build_worker(dir.path());
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }];
        let payload = worker.prefill_to_payload(&messages).unwrap();
        assert!(!payload.bundle_bytes.is_empty());
        assert_eq!(payload.last_logits.len(), 128);
        assert!(!payload.prompt_tokens.is_empty());
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p flour disagg::prefill 2>&1 | tail -20`
Expected: FAIL — `cannot find type PrefillWorker`.

- [ ] **Step 4: Implement PrefillWorker**

Add above the `#[cfg(test)]` block in `src/disagg/prefill.rs`:

```rust
/// What prefill hands to decode: the populated KV cache, the logits at the last prompt
/// position (the decoder samples its first token from these), and the prompt token ids
/// (so the decoder can apply repeat-penalty over the prompt).
pub struct PrefillOutput {
    pub bundle: KvCacheBundle,
    pub last_logits: Vec<f32>,
    pub prompt_tokens: Vec<u32>,
}

/// Runs the prompt forward pass and emits a transferable prefill result. Owns no sampler.
pub struct PrefillWorker {
    model: Arc<CausalLM>,
    tokenizer: Arc<Tokenizer>,
    chat_template: ChatTemplate,
    device: Device,
}

impl PrefillWorker {
    pub fn new(
        model: Arc<CausalLM>,
        tokenizer: Arc<Tokenizer>,
        chat_template: ChatTemplate,
        device: Device,
    ) -> Self {
        Self {
            model,
            tokenizer,
            chat_template,
            device,
        }
    }

    /// Render the chat template, tokenize, run the prompt through the model at position 0,
    /// and capture the resulting KV cache plus the final-position logits.
    pub fn prefill(&self, messages: &[ChatMessage]) -> anyhow::Result<PrefillOutput> {
        let prompt = self.chat_template.render(messages);
        let prompt_tokens = self.tokenizer.encode(&prompt).context("encoding prompt")?;

        let mut cache = Cache::new(self.model.config(), &self.device)
            .context("allocating prefill cache")?;
        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0, &mut cache)?;

        let seq_len = logits.dim(1)?;
        let last = logits.i((0, seq_len - 1))?.to_dtype(DType::F32)?;
        let last_logits = last.to_vec1::<f32>()?;

        let bundle = KvCacheBundle::from_cache(&cache)?;
        Ok(PrefillOutput {
            bundle,
            last_logits,
            prompt_tokens,
        })
    }

    /// Convenience: run `prefill` and serialize the result into a wire-ready `TransferPayload`.
    pub fn prefill_to_payload(&self, messages: &[ChatMessage]) -> anyhow::Result<TransferPayload> {
        let out = self.prefill(messages)?;
        Ok(TransferPayload {
            bundle_bytes: out.bundle.to_bytes()?,
            last_logits: out.last_logits,
            prompt_tokens: out.prompt_tokens,
        })
    }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p flour disagg::prefill 2>&1 | tail -20`
Expected: PASS — both prefill tests.

- [ ] **Step 6: Commit**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings
git add src/disagg/prefill.rs src/disagg/mod.rs
git commit -m "feat(disagg): PrefillWorker producing transferable KV + last logits"
```

---

### Task 6: DecodeWorker

The decode half: reconstruct the cache from a bundle and run the full sampling/detokenization loop — including the first token, sampled from the prefill's last logits. This is where `GenerationStats` is produced.

**Files:**
- Create/replace: `src/disagg/decode.rs` (was an empty placeholder from Task 2)
- Modify: `src/disagg/mod.rs` (restore `pub mod decode;` + `pub use decode::...`)
- Test: `src/disagg/decode.rs` (inline test)

**Interfaces:**
- Consumes: `KvCacheBundle::{into_cache, from_bytes}` (Tasks 2–3); `TransferPayload` (Task 4); `PrefillWorker` (Task 5, test only); `GenerationStats` (`crate::disagg::GenerationStats`, Task 2); `crate::models::common::{CausalLM, EosTokenId}`; `crate::sampling::{apply_repeat_penalty, LogitsSampler, SamplingParams}`; `crate::tokenizer::Tokenizer`.
- Produces:
  - `struct DecodeInput { pub bundle: KvCacheBundle, pub last_logits: Vec<f32>, pub prompt_tokens: Vec<u32> }`
  - `struct DecodeWorker { model: Arc<CausalLM>, tokenizer: Arc<Tokenizer>, eos_token_id: EosTokenId, device: Device }`
  - `DecodeWorker::new(model: Arc<CausalLM>, tokenizer: Arc<Tokenizer>, eos_token_id: EosTokenId, device: Device) -> Self`
  - `DecodeWorker::decode(&self, input: DecodeInput, params: &SamplingParams, on_token: impl FnMut(&str)) -> anyhow::Result<GenerationStats>`
  - `DecodeWorker::decode_from_payload(&self, payload: TransferPayload, params: &SamplingParams, on_token: impl FnMut(&str)) -> anyhow::Result<GenerationStats>`

- [ ] **Step 1: Restore module wiring**

In `src/disagg/mod.rs`, ensure present:

```rust
pub mod decode;
pub use decode::{DecodeInput, DecodeWorker};
```

- [ ] **Step 2: Write the failing test**

Create `src/disagg/decode.rs`:

```rust
use std::sync::Arc;

use anyhow::Context;
use candle_core::{DType, Device, IndexOp, Tensor};

use crate::disagg::bundle::KvCacheBundle;
use crate::disagg::transport::TransferPayload;
use crate::disagg::GenerationStats;
use crate::models::common::{CausalLM, EosTokenId};
use crate::sampling::{apply_repeat_penalty, LogitsSampler, SamplingParams};
use crate::tokenizer::Tokenizer;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disagg::prefill::PrefillWorker;
    use crate::loader::safetensors::load_var_builder;
    use crate::models::{self};
    use crate::tokenizer::{ChatMessage, ChatTemplate};

    struct Built {
        prefill: PrefillWorker,
        decode: DecodeWorker,
    }

    fn build(dir: &std::path::Path) -> Built {
        let raw: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
                .unwrap();
        let family = models::detect_family(&raw).unwrap();
        let cfg = models::load_config(family, &raw).unwrap();
        let device = Device::Cpu;
        let vb = load_var_builder(dir, DType::F32, &device).unwrap();
        let model = Arc::new(CausalLM::load(vb, cfg.clone()).unwrap());
        let tokenizer = Arc::new(Tokenizer::from_file(&dir.join("tokenizer.json")).unwrap());
        let chat_template = ChatTemplate::for_family(family);
        let eos = cfg
            .eos_token_id
            .clone()
            .unwrap_or(EosTokenId::Single(tokenizer.token_to_id("<|im_end|>").unwrap_or(0)));
        let prefill =
            PrefillWorker::new(model.clone(), tokenizer.clone(), chat_template, device.clone());
        let decode = DecodeWorker::new(model, tokenizer, eos, device);
        Built { prefill, decode }
    }

    #[test]
    fn decode_continues_from_a_prefill_bundle() {
        let dir = crate::engine::tests::fixture_dir_for_external_use();
        let built = build(dir.path());
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }];
        let out = built.prefill.prefill(&messages).unwrap();
        let prompt_len = out.prompt_tokens.len();
        let params = SamplingParams {
            max_tokens: 4,
            seed: 1,
            ..Default::default()
        };
        let mut produced = String::new();
        let stats = built
            .decode
            .decode(
                DecodeInput {
                    bundle: out.bundle,
                    last_logits: out.last_logits,
                    prompt_tokens: out.prompt_tokens,
                },
                &params,
                |t| produced.push_str(t),
            )
            .unwrap();
        assert_eq!(stats.prompt_tokens, prompt_len);
        assert!(stats.completion_tokens <= 4);
    }

    #[test]
    fn decode_from_payload_matches_decode_for_a_fixed_seed() {
        let dir = crate::engine::tests::fixture_dir_for_external_use();
        let built = build(dir.path());
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }];
        let params = SamplingParams {
            max_tokens: 4,
            seed: 7,
            temperature: 0.0,
            ..Default::default()
        };

        // In-memory path.
        let out = built.prefill.prefill(&messages).unwrap();
        let mut a = String::new();
        built
            .decode
            .decode(
                DecodeInput {
                    bundle: out.bundle,
                    last_logits: out.last_logits,
                    prompt_tokens: out.prompt_tokens,
                },
                &params,
                |t| a.push_str(t),
            )
            .unwrap();

        // Serialized payload path.
        let payload = built.prefill.prefill_to_payload(&messages).unwrap();
        let mut b = String::new();
        built
            .decode
            .decode_from_payload(payload, &params, |t| b.push_str(t))
            .unwrap();

        assert_eq!(a, b);
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p flour disagg::decode 2>&1 | tail -20`
Expected: FAIL — `cannot find type DecodeWorker`.

- [ ] **Step 4: Implement DecodeWorker**

Add above the `#[cfg(test)]` block in `src/disagg/decode.rs`. This mirrors the decode loop currently in `engine.rs:88-119` exactly (same sampling order, repeat-penalty window, EOS check, and stats), so colocated behavior is preserved:

```rust
/// What decode needs to continue generation from a prefilled cache.
pub struct DecodeInput {
    pub bundle: KvCacheBundle,
    pub last_logits: Vec<f32>,
    pub prompt_tokens: Vec<u32>,
}

/// Runs the autoregressive sampling loop. Owns the sampler, the tokenizer (for detokenization),
/// and the EOS rule. The first token is sampled from `DecodeInput::last_logits`.
pub struct DecodeWorker {
    model: Arc<CausalLM>,
    tokenizer: Arc<Tokenizer>,
    eos_token_id: EosTokenId,
    device: Device,
}

impl DecodeWorker {
    pub fn new(
        model: Arc<CausalLM>,
        tokenizer: Arc<Tokenizer>,
        eos_token_id: EosTokenId,
        device: Device,
    ) -> Self {
        Self {
            model,
            tokenizer,
            eos_token_id,
            device,
        }
    }

    pub fn decode(
        &self,
        input: DecodeInput,
        params: &SamplingParams,
        mut on_token: impl FnMut(&str),
    ) -> anyhow::Result<GenerationStats> {
        let DecodeInput {
            bundle,
            last_logits,
            prompt_tokens,
        } = input;

        let prompt_len = prompt_tokens.len();
        let mut cache = bundle
            .into_cache(self.model.config(), &self.device)
            .context("reconstructing cache for decode")?;
        let mut sampler = LogitsSampler::new(params.seed);
        let mut all_tokens = prompt_tokens;
        let mut logits_vec = last_logits;
        let mut completion_tokens = 0usize;

        for index_pos in (prompt_len..).take(params.max_tokens) {
            if params.repeat_penalty != 1.0 {
                let start = all_tokens.len().saturating_sub(params.repeat_last_n);
                apply_repeat_penalty(&mut logits_vec, params.repeat_penalty, &all_tokens[start..]);
            }

            let next_token = sampler.sample(&logits_vec, params);
            if self.eos_token_id.is_eos(next_token) {
                break;
            }

            let piece = self.tokenizer.decode(&[next_token])?;
            on_token(&piece);
            all_tokens.push(next_token);
            completion_tokens += 1;

            let next_input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&next_input, index_pos, &mut cache)?;
            let seq_len = logits.dim(1)?;
            let last = logits.i((0, seq_len - 1))?.to_dtype(DType::F32)?;
            logits_vec = last.to_vec1::<f32>()?;
        }

        Ok(GenerationStats {
            prompt_tokens: prompt_len,
            completion_tokens,
        })
    }

    /// Deserialize a wire payload and run `decode`.
    pub fn decode_from_payload(
        &self,
        payload: TransferPayload,
        params: &SamplingParams,
        on_token: impl FnMut(&str),
    ) -> anyhow::Result<GenerationStats> {
        let bundle = KvCacheBundle::from_bytes(&payload.bundle_bytes, &self.device)?;
        self.decode(
            DecodeInput {
                bundle,
                last_logits: payload.last_logits,
                prompt_tokens: payload.prompt_tokens,
            },
            params,
            on_token,
        )
    }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p flour disagg::decode 2>&1 | tail -20`
Expected: PASS — both decode tests (including the payload-equals-in-memory determinism check).

- [ ] **Step 6: Commit**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings
git add src/disagg/decode.rs src/disagg/mod.rs
git commit -m "feat(disagg): DecodeWorker owning the sampling loop from a KV bundle"
```

---

### Task 7: Compose Engine from the workers

Refactor `Engine` so `generate` delegates to `PrefillWorker` then `DecodeWorker`, keeping the exact public API and behavior. Re-export `GenerationStats` from its new home. Add an end-to-end regression test proving the serialized transport path matches colocated `generate`.

**Files:**
- Modify: `src/engine.rs` (struct fields, `load`, `generate`; remove the local `GenerationStats` definition; keep both existing tests; add one new test)
- Verify: `src/api/openai.rs:3` (`use crate::engine::GenerationStats;`) still resolves via re-export.

**Interfaces:**
- Consumes: `PrefillWorker`, `DecodeWorker`, `DecodeInput`, `KvCacheTransport`, `LocalTransport`, `GenerationStats` from `crate::disagg`.
- Produces (unchanged public API):
  - `Engine::load(model_dir: &Path) -> anyhow::Result<Self>`
  - `Engine::model_id(&self) -> &str`
  - `Engine::generate(&self, messages: &[ChatMessage], params: &SamplingParams, on_token: impl FnMut(&str)) -> anyhow::Result<GenerationStats>`
  - `pub use crate::disagg::GenerationStats;` (so `crate::engine::GenerationStats` keeps resolving)

- [ ] **Step 1: Write the failing test**

Add to `mod tests` in `src/engine.rs` (after `generate_is_deterministic_for_a_fixed_seed`):

```rust
    #[test]
    fn disaggregated_transport_path_matches_colocated_generate() {
        use crate::disagg::{KvCacheTransport, LocalTransport};

        let dir = tempfile::tempdir().unwrap();
        write_fixture_model(dir.path());
        let engine = Engine::load(dir.path()).unwrap();
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }];
        let params = crate::sampling::SamplingParams {
            max_tokens: 4,
            seed: 123,
            temperature: 0.0,
            ..Default::default()
        };

        // Colocated reference.
        let mut colocated = String::new();
        engine
            .generate(&messages, &params, |t| colocated.push_str(t))
            .unwrap();

        // Same engine, but the bundle goes through a serialized LocalTransport hop.
        let transport = LocalTransport::new();
        let payload = engine.prefill_worker().prefill_to_payload(&messages).unwrap();
        transport.send(payload).unwrap();
        let received = transport.recv().unwrap();
        let mut disagg = String::new();
        engine
            .decode_worker()
            .decode_from_payload(received, &params, |t| disagg.push_str(t))
            .unwrap();

        assert_eq!(colocated, disagg);
    }
```

This requires two new test-support accessors `Engine::prefill_worker()` / `Engine::decode_worker()` — add them in Step 3.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p flour engine::tests::disaggregated_transport_path 2>&1 | tail -20`
Expected: FAIL — `no method named prefill_worker` (and the refactor below is not yet in place).

- [ ] **Step 3: Refactor Engine to compose the workers**

In `src/engine.rs`, replace the imports, the local `GenerationStats` struct, the `Engine` struct, `load`, and `generate` as follows.

Replace the top `use` block (lines 1-12) with:

```rust
use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use candle_core::{DType, Device};

use crate::disagg::{DecodeInput, DecodeWorker, PrefillWorker};
use crate::loader::safetensors::load_var_builder;
use crate::models::{
    self,
    common::{CausalLM, EosTokenId},
};
use crate::sampling::SamplingParams;
use crate::tokenizer::{ChatMessage, ChatTemplate, Tokenizer};

pub use crate::disagg::GenerationStats;
```

Delete the local `GenerationStats` struct (old lines 14-17) — it now lives in `crate::disagg`.

Replace the `Engine` struct (old lines 19-26) with:

```rust
pub struct Engine {
    prefill: PrefillWorker,
    decode: DecodeWorker,
    model_id: String,
}
```

Replace `Engine::load` (old lines 29-66) with the version that builds shared `Arc`s and constructs both workers:

```rust
    pub fn load(model_dir: &Path) -> anyhow::Result<Self> {
        let config_path = model_dir.join("config.json");
        let raw: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .with_context(|| format!("reading {}", config_path.display()))?,
        )?;
        let family = models::detect_family(&raw)?;
        let cfg = models::load_config(family, &raw)?;

        let device = Device::Cpu;
        let vb = load_var_builder(model_dir, DType::F32, &device)?;
        let model = Arc::new(CausalLM::load(vb, cfg.clone())?);

        let tokenizer = Arc::new(Tokenizer::from_file(&model_dir.join("tokenizer.json"))?);
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

        let prefill = PrefillWorker::new(
            model.clone(),
            tokenizer.clone(),
            chat_template,
            device.clone(),
        );
        let decode = DecodeWorker::new(model, tokenizer, eos_token_id, device);

        Ok(Self {
            prefill,
            decode,
            model_id,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }
```

Replace `Engine::generate` (old lines 72-120) with the composing version:

```rust
    pub fn generate(
        &self,
        messages: &[ChatMessage],
        params: &SamplingParams,
        on_token: impl FnMut(&str),
    ) -> anyhow::Result<GenerationStats> {
        let out = self.prefill.prefill(messages)?;
        tracing::info!(
            "Finished prefill, input token count: {}",
            out.prompt_tokens.len()
        );
        self.decode.decode(
            DecodeInput {
                bundle: out.bundle,
                last_logits: out.last_logits,
                prompt_tokens: out.prompt_tokens,
            },
            params,
            on_token,
        )
    }

    #[cfg(test)]
    pub(crate) fn prefill_worker(&self) -> &PrefillWorker {
        &self.prefill
    }

    #[cfg(test)]
    pub(crate) fn decode_worker(&self) -> &DecodeWorker {
        &self.decode
    }
```

NOTE on test imports: the `mod tests` block currently imports `use candle_core::{DType, Device, Tensor};` and `use super::*;`. The fixture code uses `DType`, `Device`, `Tensor`, `HashMap` — all still imported locally in the test module, so they remain valid. `IndexOp` is no longer used in `engine.rs` (it moved to the workers) — confirm `cargo clippy` does not flag an unused import in the non-test code (the new `use` block above does not import `IndexOp`, so this is already handled).

- [ ] **Step 4: Run the full test suite to verify everything passes**

Run: `cargo test -p flour 2>&1 | tail -30`
Expected: PASS — all tests, specifically:
- `engine::tests::load_and_generate_end_to_end_with_tiny_random_model` (unchanged regression)
- `engine::tests::generate_is_deterministic_for_a_fixed_seed` (unchanged regression)
- `engine::tests::disaggregated_transport_path_matches_colocated_generate` (new)
- all `disagg::*` and `models::common::cache::*` tests

- [ ] **Step 5: Verify the API layer still compiles against the re-export**

Run: `cargo build -p flour 2>&1 | tail -20`
Expected: clean build. `src/api/openai.rs`'s `use crate::engine::GenerationStats;` resolves through the `pub use` re-export.

- [ ] **Step 6: Commit**

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings
git add src/engine.rs
git commit -m "refactor(engine): compose generate from Prefill/Decode workers"
```

---

## Self-Review

**1. Spec coverage**

- "把 Prefill 和 Decode 部分分开" (split prefill and decode): Tasks 5 (`PrefillWorker`) + 6 (`DecodeWorker`) + 7 (Engine composes them). ✓
- "让 KV Cache 在节点间传输的准备工作" (prepare KV cache for cross-node transfer): Task 1 (export/install on `Cache`) + Tasks 2–3 (`KvCacheBundle` in-memory + safetensors bytes) + Task 4 (`KvCacheTransport` seam + `LocalTransport`). ✓
- Networking out of scope, seam in place: `KvCacheTransport` trait with `LocalTransport`; a network impl implements the same trait and frames `TransferPayload`. ✓
- Behavior preserved: Task 7 keeps `Engine::generate`'s signature; the two original engine tests are untouched; Task 7's new test asserts the transport path equals colocated output for a fixed seed. ✓

**2. Placeholder scan**

No "TBD/TODO/handle edge cases/similar to Task N" left. The only intentional transient placeholders are the empty `transport.rs`/`prefill.rs`/`decode.rs` files created in Task 2 so the module tree resolves; Tasks 4–6 replace them with full implementations, and Task 2's `mod.rs` only `pub use`s `KvCacheBundle` until those tasks restore the rest. Each code step shows complete code.

**3. Type consistency**

- `KvCacheBundle` field `layers: Vec<(Tensor, Tensor)>`; methods `from_cache`/`into_cache`/`num_layers`/`seq_len`/`to_bytes`/`from_bytes` used identically in Tasks 2, 3, 5, 6. ✓
- `Cache::export_kv` / `install_kv` signatures (candle `Result`) consumed by `from_cache`/`into_cache`. ✓
- `TransferPayload { bundle_bytes, last_logits, prompt_tokens }` constructed in `prefill_to_payload` (Task 5) and consumed in `decode_from_payload` (Task 6) and `LocalTransport` (Task 4) — field names match. ✓
- `PrefillOutput { bundle, last_logits, prompt_tokens }` → mapped into `DecodeInput { bundle, last_logits, prompt_tokens }` in Tasks 6 and 7 — names match. ✓
- `GenerationStats { prompt_tokens, completion_tokens }` defined once in `disagg/mod.rs`, produced by `DecodeWorker::decode`, re-exported from `engine.rs` for `api/openai.rs`. ✓
- `DecodeWorker::new(model, tokenizer, eos_token_id, device)` and `PrefillWorker::new(model, tokenizer, chat_template, device)` argument order matches construction in `Engine::load` and in the worker tests. ✓
