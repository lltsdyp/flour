# Co-located Engine + Worker Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let each inference node run the `Engine` (DistKV client) and a `WorkerStore` in one process/port, with write-locality (PUT prefers the local worker) and read-locality (GET to the local worker short-circuits HTTP via an in-process read).

**Architecture:** The Master stays a separate single process that manages only Workers. A co-located node embeds a `WorkerStore` behind the same axum server as the OpenAI API, registers/heartbeats it with the Master, and the Engine compares each route's `worker_id` to its own local worker id to decide between an in-process store access and the existing HTTP data path. The metadata safety state machine (two-phase PUT, leases, epoch/generation guards) is untouched.

**Tech Stack:** Rust 2021, axum, tokio, reqwest, serde/serde_json, anyhow.

## Global Constraints

- Master never touches object bytes; the data path stays Worker↔requester. (unchanged)
- The Master manages Stores only — never Engines. No engine registry, no engine liveness.
- Locality is a pure client-side optimization: the Engine knows its local worker id only from shared process config, never by querying the Master.
- Remote cache is strictly optional: a Master that is down or a failed get/put degrades to a cache miss + local prefill, never an inference error.
- Objects are stored by `(key, generation)`; the worker data path does not check epoch. GET safety (Complete + Alive + epoch match) is enforced only in the Master and is left unchanged.
- KV paging block size `BLOCK_SIZE = 16` (in `src/distkv/scheduler.rs`); store policy stores prompts of `>= 2 * BLOCK_SIZE` tokens with new content.
- Existing standalone binaries `flour-master` and `flour-worker` must keep working (disaggregated mode). Co-location is additive.
- Run `cargo fmt` before each commit; keep `cargo clippy` clean.

---

### Task 1: Add `preferred_worker_id` to `PutStartRequest`

**Files:**
- Modify: `src/distkv/protocol.rs:50-54`
- Test: `src/distkv/protocol.rs` (tests module)

**Interfaces:**
- Consumes: nothing.
- Produces: `PutStartRequest { key: ObjectKey, size_bytes: usize, preferred_worker_id: Option<WorkerId> }`. `preferred_worker_id` is `#[serde(default)]` so legacy JSON without the field deserializes to `None`.

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `src/distkv/protocol.rs`:

```rust
    #[test]
    fn put_start_request_round_trips_with_preferred_worker() {
        round_trip(&PutStartRequest {
            key: "k".to_string(),
            size_bytes: 4096,
            preferred_worker_id: Some("w1".to_string()),
        });
    }

    #[test]
    fn put_start_request_legacy_json_defaults_preferred_to_none() {
        // JSON written by an older peer that predates preferred_worker_id.
        let legacy = r#"{"key":"k","size_bytes":4096}"#;
        let req: PutStartRequest = serde_json::from_str(legacy).expect("deserialize legacy");
        assert_eq!(req.preferred_worker_id, None);
    }
```

Also update the existing `put_start_request_round_trips` test to set the new field:

```rust
    #[test]
    fn put_start_request_round_trips() {
        round_trip(&PutStartRequest {
            key: "kv://v1/model/m/prefix/abc/tokens/64".to_string(),
            size_bytes: 4096,
            preferred_worker_id: None,
        });
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib distkv::protocol`
Expected: FAIL — `PutStartRequest` has no field `preferred_worker_id`.

- [ ] **Step 3: Add the field**

In `src/distkv/protocol.rs`, change the struct:

```rust
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PutStartRequest {
    pub key: ObjectKey,
    pub size_bytes: usize,
    #[serde(default)]
    pub preferred_worker_id: Option<WorkerId>,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib distkv::protocol`
Expected: PASS. (Other crates referencing `PutStartRequest` will not compile yet — fixed in Tasks 2 and 3. Run only this module's tests here.)

- [ ] **Step 5: Commit**

```bash
cargo fmt
git add src/distkv/protocol.rs
git commit -m "feat(distkv): add preferred_worker_id to PutStartRequest"
```

---

### Task 2: Master honors `preferred_worker_id` in `put_start`

**Files:**
- Modify: `src/distkv/master.rs:147-156`
- Test: `src/distkv/master.rs` (tests module)

**Interfaces:**
- Consumes: `PutStartRequest.preferred_worker_id` from Task 1.
- Produces: `put_start` selects the preferred worker when it is alive with enough free space, else falls back to the existing max-free-space selection. No signature change.

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `src/distkv/master.rs`. Note the existing tests construct `PutStartRequest` without the new field — update those call sites to add `preferred_worker_id: None` (there are calls in `put_start_creates_writing_object_not_readable`, `put_commit_makes_object_readable`, `late_commit_with_wrong_put_id_is_rejected`, `get_route_skips_dead_worker`, `capacity_accounting_rejects_when_no_worker_has_space`).

```rust
    #[test]
    fn put_start_honors_preferred_worker() {
        let mut m = master_at(1000);
        // w_big has more free space, so without a preference it would be chosen.
        m.register_worker("w_big".into(), "http://w_big".into(), 1 << 20);
        m.register_worker("w_pref".into(), "http://w_pref".into(), 1 << 18);

        let resp = m
            .put_start(PutStartRequest {
                key: "k".into(),
                size_bytes: 128,
                preferred_worker_id: Some("w_pref".into()),
            })
            .unwrap();
        assert_eq!(resp.worker_id, "w_pref");
    }

    #[test]
    fn put_start_falls_back_when_preferred_full_or_dead() {
        let mut m = master_at(1000);
        m.register_worker("w_other".into(), "http://w_other".into(), 1 << 20);
        m.register_worker("w_pref".into(), "http://w_pref".into(), 100);

        // Preferred worker cannot fit the object -> fall back to capacity choice.
        let resp = m
            .put_start(PutStartRequest {
                key: "k".into(),
                size_bytes: 200,
                preferred_worker_id: Some("w_pref".into()),
            })
            .unwrap();
        assert_eq!(resp.worker_id, "w_other");

        // Unknown preferred worker -> fall back as well.
        let resp2 = m
            .put_start(PutStartRequest {
                key: "k2".into(),
                size_bytes: 128,
                preferred_worker_id: Some("ghost".into()),
            })
            .unwrap();
        assert_eq!(resp2.worker_id, "w_other");
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib distkv::master`
Expected: FAIL — `put_start` ignores `preferred_worker_id` (and the new-field call sites won't compile until updated).

- [ ] **Step 3: Implement preferred selection**

In `src/distkv/master.rs`, replace the worker-selection block in `put_start` (the `// Pick the alive worker ...` section) with:

```rust
        // Write locality: honor a usable preferred worker; otherwise pick the
        // alive worker with the most free space that can fit the object.
        let preferred = req.preferred_worker_id.as_ref().and_then(|id| {
            self.workers.get(id).and_then(|w| {
                (self.worker_is_alive(w, now) && w.free_bytes() >= req.size_bytes)
                    .then(|| id.clone())
            })
        });
        let chosen = preferred.or_else(|| {
            self.workers
                .iter()
                .filter(|(_, w)| self.worker_is_alive(w, now) && w.free_bytes() >= req.size_bytes)
                .max_by_key(|(_, w)| w.free_bytes())
                .map(|(id, _)| id.clone())
        });

        let worker_id = chosen
            .ok_or_else(|| anyhow::anyhow!("no alive worker has {} free bytes", req.size_bytes))?;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib distkv::master`
Expected: PASS (all master tests, including the two new ones).

- [ ] **Step 5: Commit**

```bash
cargo fmt
git add src/distkv/master.rs
git commit -m "feat(distkv): master honors preferred_worker_id for write locality"
```

---

### Task 3: Refactor `DistKvClient` into granular steps + preferred PUT

**Files:**
- Modify: `src/distkv/client.rs:36-133`
- Test: `src/distkv/client.rs` (tests module)

**Interfaces:**
- Consumes: `PutStartRequest.preferred_worker_id` (Task 1); Master preferred selection (Task 2).
- Produces, as public async methods on `DistKvClient`:
  - `put_start(&self, key: &str, size_bytes: usize, preferred_worker_id: Option<&str>) -> anyhow::Result<PutStartResponse>`
  - `write_worker(&self, worker_addr: &str, key: &str, generation: u64, bytes: Vec<u8>) -> anyhow::Result<()>`
  - `put_commit(&self, key: &str, put_id: PutId) -> anyhow::Result<()>`
  - `get_route(&self, key: &str) -> anyhow::Result<Option<GetRouteResponse>>`
  - `fetch_worker(&self, worker_addr: &str, key: &str, generation: u64) -> anyhow::Result<Option<Vec<u8>>>`
  - existing `put_object` / `get_object` retained, now composing the steps with `preferred = None`.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `src/distkv/client.rs` (the existing `cluster()` helper registers a worker `w1`):

```rust
    #[tokio::test]
    async fn put_start_routes_to_preferred_worker() {
        let client = cluster().await; // registers worker "w1"
        let start = client
            .put_start("kv://k", 128, Some("w1"))
            .await
            .unwrap();
        assert_eq!(start.worker_id, "w1");
        assert_eq!(start.object_generation, 1);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib distkv::client`
Expected: FAIL — no method `put_start` on `DistKvClient`.

- [ ] **Step 3: Refactor the client**

In `src/distkv/client.rs`, replace the `put_object` and `get_object` method bodies with granular steps plus thin compositions. Keep the existing `post_json` helper and `encode_segment` function.

```rust
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
```

Ensure `use crate::distkv::protocol::*;` already imports `PutId` (it does, via the glob).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib distkv::client`
Expected: PASS — `put_start_routes_to_preferred_worker` plus the existing `put_object_then_get_object_round_trips_through_real_http` and `get_object_miss_returns_none`.

- [ ] **Step 5: Commit**

```bash
cargo fmt
git add src/distkv/client.rs
git commit -m "refactor(distkv): expose granular client steps with preferred PUT"
```

---

### Task 4: Co-located `RemoteKvCache` with write + read locality

**Files:**
- Modify: `src/engine.rs:16-48` (RemoteKvCache), `src/engine.rs:105-174` (Engine fields/constructors)
- Test: `src/engine.rs` (tests module)

**Interfaces:**
- Consumes: granular `DistKvClient` methods (Task 3); `WorkerStore::{put_bytes,get_bytes}` (existing).
- Produces:
  - `Engine::enable_remote_kv_colocated(&mut self, master_url: &str, worker_id: String, store: Arc<Mutex<WorkerStore>>) -> anyhow::Result<()>`
  - `RemoteKvCache::connect_colocated(master_url, worker_id, store)`; `get`/`put` short-circuit to the local store when the route's `worker_id` equals the local one.

- [ ] **Step 1: Write the failing tests**

In `src/engine.rs`, add imports at the top of the test module (the test module already has `use std::sync::{Arc, Mutex}` and brings in `MasterState`, `master_router`, `worker_router`, `WorkerStore`, `RegisterRequest`, `DistKvClient`, and `spawn`/`remote_cluster`/`loaded_engine`/`run_generate` helpers — reuse them). Add this co-located harness and three tests:

```rust
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
        assert_eq!(second.remote_cache_hit, Some(true), "served over HTTP by w1");
        // The local store was never the route target, so it stays empty.
        assert_eq!(local.lock().unwrap().used_bytes(), 0);

        std::mem::forget(engine);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib engine::`
Expected: FAIL — no method `enable_remote_kv_colocated`.

- [ ] **Step 3: Implement the co-located cache**

In `src/engine.rs`, add imports near the top (after the existing `use` lines):

```rust
use std::sync::{Arc, Mutex};

use crate::distkv::worker::WorkerStore;
```

Replace the `RemoteKvCache` struct and impl (lines ~22-48) with:

```rust
pub struct RemoteKvCache {
    client: DistKvClient,
    runtime: tokio::runtime::Runtime,
    /// Present in co-located mode: this node's own worker, accessed in-process
    /// when a route resolves to it (read/write locality).
    local: Option<LocalWorker>,
}

struct LocalWorker {
    worker_id: String,
    store: Arc<Mutex<WorkerStore>>,
}

impl RemoteKvCache {
    /// Remote-only cache: every get/put flows over HTTP. Does not connect now;
    /// failures surface lazily so the remote cache stays optional.
    pub fn connect(master_url: &str) -> anyhow::Result<Self> {
        Self::build(master_url, None)
    }

    /// Co-located cache: writes prefer the local worker and routes that resolve
    /// to it are read directly from the in-process store.
    pub fn connect_colocated(
        master_url: &str,
        worker_id: String,
        store: Arc<Mutex<WorkerStore>>,
    ) -> anyhow::Result<Self> {
        Self::build(master_url, Some(LocalWorker { worker_id, store }))
    }

    fn build(master_url: &str, local: Option<LocalWorker>) -> anyhow::Result<Self> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        Ok(Self {
            client: DistKvClient::new(master_url),
            runtime,
            local,
        })
    }

    fn get(&self, key: &str) -> anyhow::Result<Option<Vec<u8>>> {
        self.runtime.block_on(async {
            let route = match self.client.get_route(key).await? {
                Some(r) => r,
                None => return Ok(None),
            };
            match &self.local {
                // Read locality: route points at our own worker -> in-process read.
                Some(l) if l.worker_id == route.worker_id => l
                    .store
                    .lock()
                    .map_err(|_| anyhow::anyhow!("local store poisoned"))?
                    .get_bytes(key, route.object_generation),
                _ => {
                    self.client
                        .fetch_worker(&route.worker_addr, key, route.object_generation)
                        .await
                }
            }
        })
    }

    fn put(&self, key: &str, bytes: Vec<u8>) -> anyhow::Result<()> {
        self.runtime.block_on(async {
            let preferred = self.local.as_ref().map(|l| l.worker_id.as_str());
            let start = self.client.put_start(key, bytes.len(), preferred).await?;
            match &self.local {
                // Write locality: master pinned it to us -> in-process write.
                Some(l) if l.worker_id == start.worker_id => {
                    l.store
                        .lock()
                        .map_err(|_| anyhow::anyhow!("local store poisoned"))?
                        .put_bytes(key.to_string(), start.object_generation, bytes)?;
                }
                _ => {
                    self.client
                        .write_worker(&start.worker_addr, key, start.object_generation, bytes)
                        .await?;
                }
            }
            self.client.put_commit(key, start.put_id).await
        })
    }
}
```

Then add the co-located enable method next to `enable_remote_kv` (after line ~174):

```rust
    /// Enables the remote KV cache in co-located mode: this node embeds the
    /// worker `worker_id` backed by `store`, so writes prefer it and local
    /// routes are read in-process. Best-effort, like `enable_remote_kv`.
    pub fn enable_remote_kv_colocated(
        &mut self,
        master_url: &str,
        worker_id: String,
        store: Arc<Mutex<WorkerStore>>,
    ) -> anyhow::Result<()> {
        self.remote_kv = Some(RemoteKvCache::connect_colocated(master_url, worker_id, store)?);
        Ok(())
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib engine::`
Expected: PASS — the three new co-located tests plus existing `generation_*` remote tests.

- [ ] **Step 5: Commit**

```bash
cargo fmt
git add src/engine.rs
git commit -m "feat(engine): co-located RemoteKvCache with read/write locality"
```

---

### Task 5: Extract reusable registration/heartbeat helper

**Files:**
- Create: `src/distkv/registration.rs`
- Modify: `src/distkv/mod.rs:7-13`, `src/bin/flour-worker.rs:18-138`
- Test: `src/distkv/registration.rs` (tests module)

**Interfaces:**
- Consumes: `RegisterRequest`, `RegisterResponse`, `HeartbeatRequest` (existing protocol).
- Produces:
  - `pub async fn register(http: &reqwest::Client, master_url: &str, req: &RegisterRequest) -> anyhow::Result<u64>` — retries until success, returns epoch.
  - `pub async fn heartbeat_once(http: &reqwest::Client, master_url: &str, worker_id: &str, epoch: u64) -> bool`
  - `pub async fn run_registration(http: reqwest::Client, master_url: String, req: RegisterRequest) -> anyhow::Result<()>` — register, then heartbeat loop, re-registering on rejection.
  - `pub const HEARTBEAT_INTERVAL: Duration`, `pub const REGISTER_RETRY: Duration`.

- [ ] **Step 1: Write the failing tests**

Create `src/distkv/registration.rs` with only the test module first (implementation in Step 3):

```rust
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib distkv::registration`
Expected: FAIL — module not declared / functions missing.

- [ ] **Step 3: Implement the helper**

Add `pub mod registration;` to `src/distkv/mod.rs` (keep the list alphabetical: after `protocol`).

Prepend the implementation to `src/distkv/registration.rs` (above the test module):

```rust
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
    tracing::info!("registered '{}' with master as epoch {epoch}", req.worker_id);
    loop {
        tokio::time::sleep(HEARTBEAT_INTERVAL).await;
        if !heartbeat_once(&http, &master_url, &req.worker_id, epoch).await {
            tracing::warn!("heartbeat failed, re-registering with master");
            epoch = register(&http, &master_url, &req).await?;
            tracing::info!("re-registered '{}' as epoch {epoch}", req.worker_id);
        }
    }
}

```

- [ ] **Step 4: Refactor `flour-worker` to use the shared units**

In `src/bin/flour-worker.rs`: delete the local `register` function (lines ~52-75) and the two `const HEARTBEAT_INTERVAL`/`REGISTER_RETRY` (lines ~20-22). Replace the imports and heartbeat loop to call the shared helpers, keeping the binary's extra "server died" guard:

Change the `use` block to add:

```rust
use flour::distkv::registration::{heartbeat_once, register, HEARTBEAT_INTERVAL};
```

Replace the registration + heartbeat loop in `main` (from `let mut epoch = register(...)` to the end of the `loop { ... }`) with:

```rust
    let mut epoch = register(&http, &master_url, &reg).await?;
    tracing::info!("registered '{}' with master as epoch {epoch}", args.worker_id);

    loop {
        tokio::time::sleep(HEARTBEAT_INTERVAL).await;
        if !heartbeat_once(&http, &master_url, &args.worker_id, epoch).await {
            tracing::warn!("heartbeat failed, re-registering with master");
            epoch = register(&http, &master_url, &reg).await?;
            tracing::info!("re-registered '{}' as epoch {epoch}", args.worker_id);
        }
        if server.is_finished() {
            anyhow::bail!("worker data-path server stopped");
        }
    }
```

- [ ] **Step 5: Run tests and build to verify**

Run: `cargo test --lib distkv::registration && cargo build --bin flour-worker`
Expected: tests PASS; `flour-worker` builds.

- [ ] **Step 6: Commit**

```bash
cargo fmt
git add src/distkv/registration.rs src/distkv/mod.rs src/bin/flour-worker.rs
git commit -m "refactor(distkv): extract shared registration/heartbeat helper"
```

---

### Task 6: Wire co-located mode into the `flour` server

**Files:**
- Modify: `src/api/mod.rs:44-49` (add `serve_router`)
- Modify: `src/main.rs:1-76`
- Test: `src/api/mod.rs` (tests module)

**Interfaces:**
- Consumes: `mount_distkv` (existing), `Engine::enable_remote_kv_colocated` (Task 4), `run_registration` + `RegisterRequest` (Task 5), `WorkerStore` (existing).
- Produces: `pub async fn serve_router(app: Router, addr: SocketAddr) -> anyhow::Result<()>`; new `flour` CLI flags `--colocated-worker`, `--worker-id`, `--advertise-url`, `--distkv-capacity-bytes`.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `src/api/mod.rs`:

```rust
    #[tokio::test]
    async fn one_server_serves_api_and_worker_data_path() {
        let dir = crate::engine::tests::fixture_dir_for_external_use();
        let engine = Engine::load(dir.path(), None).unwrap();
        std::mem::forget(dir);
        let state = AppState {
            engine: Arc::new(Mutex::new(engine)),
            started_at: 0,
        };
        let store = Arc::new(Mutex::new(
            crate::distkv::worker::WorkerStore::new("local".into(), 0, 1 << 20),
        ));
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
            .put(format!("http://{addr}/v1/distkv/worker/objects/k?generation=1"))
            .body(vec![1u8, 2, 3])
            .send()
            .await
            .unwrap();
        assert!(put.status().is_success());
        let got = http
            .get(format!("http://{addr}/v1/distkv/worker/objects/k?generation=1"))
            .send()
            .await
            .unwrap();
        assert_eq!(got.status(), 200);
        assert_eq!(got.bytes().await.unwrap().as_ref(), &[1u8, 2, 3]);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib api::tests::one_server_serves_api_and_worker_data_path`
Expected: FAIL — the test won't compile until `mount_distkv` is in scope in tests (it is, via `use super::*`) — if it already compiles, it should still PASS only after Step 3 adds nothing for this test... Actually `mount_distkv` already exists, so this test may PASS immediately. If it PASSES, that confirms the co-mount works; proceed to add `serve_router` and the CLI wiring below (still required for `main.rs`).

- [ ] **Step 3: Add `serve_router` to the API**

In `src/api/mod.rs`, refactor `serve` to delegate to a router-level entry point:

```rust
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
```

- [ ] **Step 4: Add CLI flags and wiring to `main.rs`**

Replace `src/main.rs` with:

```rust
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
        let master_url = args.remote_kv_master_url.as_deref().ok_or_else(|| {
            anyhow::anyhow!("--colocated-worker requires --remote-kv-master-url")
        })?;
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
```

- [ ] **Step 5: Run the test and a full build**

Run: `cargo test --lib api:: && cargo build`
Expected: API tests PASS; the whole workspace (all three binaries) builds.

- [ ] **Step 6: Commit**

```bash
cargo fmt
git add src/api/mod.rs src/main.rs
git commit -m "feat: co-located engine+worker server wiring and CLI"
```

---

### Task 7: Co-located cluster example + docs

**Files:**
- Modify: `examples/cluster/docker-compose.yml`
- Modify: `examples/cluster/README.md`

**Interfaces:**
- Consumes: the `flour` CLI flags from Task 6.
- Produces: a `colocated` compose profile demonstrating engine+worker nodes against one Master.

- [ ] **Step 1: Add a co-located profile to docker-compose**

Append to `examples/cluster/services` in `examples/cluster/docker-compose.yml` (keep the existing `master`, `worker1`, `worker2`, `engine` services). Two co-located nodes share the one Master:

```yaml
  # Co-located nodes: each runs the engine AND an embedded worker in one
  # process/port. Enabled only with `--profile colocated`. Mount a model dir.
  node1:
    <<: *flour-build
    profiles: ["colocated"]
    command:
      - flour
      - --model-dir=/models
      - --host=0.0.0.0
      - --port=8080
      - --colocated-worker
      - --worker-id=node1
      - --advertise-url=http://node1:8080
      - --remote-kv-master-url=http://master:8081
      - --distkv-capacity-bytes=1073741824
    environment:
      RUST_LOG: info
    volumes:
      - ${MODEL_DIR:-./model}:/models:ro
    ports:
      - "8080:8080"
    depends_on:
      - master

  node2:
    <<: *flour-build
    profiles: ["colocated"]
    command:
      - flour
      - --model-dir=/models
      - --host=0.0.0.0
      - --port=8080
      - --colocated-worker
      - --worker-id=node2
      - --advertise-url=http://node2:8080
      - --remote-kv-master-url=http://master:8081
      - --distkv-capacity-bytes=1073741824
    environment:
      RUST_LOG: info
    volumes:
      - ${MODEL_DIR:-./model}:/models:ro
    depends_on:
      - master
```

- [ ] **Step 2: Document the co-located mode in the README**

Add a section to `examples/cluster/README.md` explaining co-located mode. Include exactly this run command and explanation:

```markdown
## Co-located mode (engine + worker per node)

Each node runs the engine and an embedded KV-cache worker in one process and on
one port. Writes prefer the node's own worker, and reads of locally-produced KV
skip the network entirely; cross-node reuse still flows over the data path. The
Master stays a separate process that manages only the workers (never the engines).

Bring up one Master plus two co-located nodes:

    MODEL_DIR=/abs/path/to/model \
      docker compose -f examples/cluster/docker-compose.yml --profile colocated up --build

node1 is published on http://localhost:8080. Each node registers its embedded
worker with the Master under `--worker-id` and advertises `--advertise-url` so
peers can fetch its bytes.
```

- [ ] **Step 3: Validate the compose file**

Run: `docker compose -f examples/cluster/docker-compose.yml --profile colocated config >/dev/null && echo OK`
Expected: prints `OK` (compose file parses; no build/run needed here).

- [ ] **Step 4: Commit**

```bash
git add examples/cluster/docker-compose.yml examples/cluster/README.md
git commit -m "docs(cluster): add co-located engine+worker compose profile"
```

---

### Task 8: Full verification

**Files:** none (verification only).

- [ ] **Step 1: Run the full test suite**

Run: `cargo test`
Expected: PASS — all unit/integration tests, including the new protocol, master, client, engine, registration, and api tests, plus the unchanged DistKV integration tests and deep-bug simulator.

- [ ] **Step 2: Lint and format check**

Run: `cargo clippy --all-targets -- -D warnings && cargo fmt --check`
Expected: no warnings; formatting clean.

- [ ] **Step 3: Build all binaries**

Run: `cargo build --bins`
Expected: `flour`, `flour-master`, `flour-worker` all build.

- [ ] **Step 4: Commit any formatting fixups**

```bash
cargo fmt
git diff --quiet || (git add -A && git commit -m "style: cargo fmt")
```

---

## Self-Review

**Spec coverage:**
- §1 write locality → Tasks 1-2 (protocol + master), Task 4 (engine put). ✓
- §1 read locality short-circuit → Task 4 (engine get). ✓
- §2 "Master manages stores not engines" → no engine registry added; `get_route` unchanged (Task 3 only composes it). ✓
- §2 locality is client-side → comparison done in `RemoteKvCache` (Task 4); Master unaware. ✓
- §3 single server/port → Task 6 `mount_distkv` + `one_server_serves_api_and_worker_data_path`. ✓
- §3 background register/heartbeat → Task 5 helper + Task 6 spawn. ✓
- §5 protocol change (only `preferred_worker_id`, `get_route` untouched) → Task 1. ✓
- §6 CLI flags → Task 6. ✓
- §7 fallback semantics (master down, preferred full, local evicted, lock poisoned) → Task 2 fallback, Task 4 poisoned-lock-as-error→miss, existing `generation_succeeds_when_master_is_down`. ✓
- §8 tests → Tasks 1-6 each carry their tests; Task 8 runs the regression suite + simulator. ✓
- §9 non-goals → no replicas/HA/registry introduced. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. ✓

**Type consistency:** `enable_remote_kv_colocated(master_url, worker_id: String, store: Arc<Mutex<WorkerStore>>)`, `connect_colocated`, `put_start(key, size, Option<&str>)`, `fetch_worker`/`write_worker`, `run_registration`/`register`/`heartbeat_once` — names and signatures match across Tasks 3-6. `PutStartRequest.preferred_worker_id: Option<WorkerId>` used consistently. ✓

## Execution Handoff

(Filled in after the plan is saved.)
