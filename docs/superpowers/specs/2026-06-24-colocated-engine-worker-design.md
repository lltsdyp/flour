# Co-located Engine + Worker Mode for `flour` DistKV — Design

**Date:** 2026-06-24
**Status:** Approved design, pending implementation plan
**Builds on:** `docs/plan/plan-distkv.md` (distributed KV cache: Master metadata + Worker object store + cache-aware Engine client)

## 1. Goal

Add a **co-located** deployment mode to the existing DistKV layer: each inference node
runs the `Engine` (the DistKV *Client*) and a `WorkerStore` (the KV object store) **in the
same process, behind a single axum server and port**. The Master stays a separate single
process that manages all Workers.

The point of co-location is **data locality**:

- **Write locality** — KV bundles produced by an Engine are PUT to *its own* local Worker,
  pinning freshly generated KV on the node that produced it and avoiding a network hop on
  write.
- **Read locality (in-process short-circuit)** — when a GET route resolves to the local
  Worker, the Engine reads the bytes directly from the in-process `WorkerStore`, skipping
  HTTP and serialization entirely.

The common case (a node reusing a prefix it generated itself) then does zero network I/O for
both read and write. Cross-node reuse still flows over the existing data path unchanged.

## 2. Architectural constraints (must hold)

- **The Master manages Stores, not Engines.** Only the `WorkerStore` registers and
  heartbeats with the Master and appears in its worker registry / capacity / liveness /
  placement tables. The Engine is always a *stateless client* of the Master (`put_start`,
  `get_route`, `put_commit`); the Master maintains no identity or liveness for Engines.
- **Locality is a pure client-side optimization.** In a co-located node the Engine knows its
  "local worker id" only because it shares process configuration (the same `worker_id` arg
  and the same `Arc<Mutex<WorkerStore>>`), **not** by querying the Master. The Engine itself
  compares `route.worker_id == local worker_id` and short-circuits. The Master neither knows
  nor needs to know that the short-circuit exists.
- **Master never touches object bytes.** Unchanged. The data path stays Worker↔requester.
- **Existing DistKV safety is untouched.** Objects are stored by `(key, generation)`; GET
  returns only a `Complete` object on an `Alive` worker whose `placement.worker_epoch`
  matches the worker's current epoch; the two-phase PUT (`put_start` → write → `put_commit`)
  and the four deep-bug guards (dirty read, lease/reuse, ABA commit, ghost placement) are
  preserved verbatim. Co-location changes only *which* worker is chosen and *how* bytes are
  fetched, never the metadata state machine.
- **Remote cache stays strictly optional.** If the Master is unreachable, registration is
  retried in the background and inference proceeds via local prefill; every remote get/put
  failure degrades to a cache miss, never an inference error.

## 3. Process topology

```
            ┌─────────────┐
            │   Master    │  separate single process (flour-master)
            │ metadata /  │  worker registry, object state, placement, leases
            │  routing    │  put_start / put_commit / get_route (metadata only)
            └──────┬──────┘
       register/   │
       heartbeat   │   (no bytes ever flow through the Master)
        ┌──────────┼──────────┐
        │          │          │
 ┌──────┴─────┐ ┌──┴─────────┐ ...   each is a co-located node
 │  Node N1   │ │  Node N2   │
 │ ┌────────┐ │ │ ┌────────┐ │   one axum server / one port:
 │ │ engine │ │ │ │ engine │ │     /v1/chat/completions
 │ │+Worker │ │ │ │+Worker │ │     /v1/distkv/worker/objects/{key}
 │ │ Store  │ │ │ │ Store  │ │
 │ └────────┘ │ │ └────────┘ │
 └─────┬──────┘ └─────┬──────┘
       └─ cross-node KV byte transfer (data path, HTTP) ─┘
```

### Components inside a co-located node (one process)

- **`Engine`** — inference plus the DistKV client role.
- **`WorkerStore`** wrapped in `Arc<Mutex<…>>`, shared by two consumers:
  1. the axum `worker_router`, so *other* nodes can fetch this node's bytes over HTTP;
  2. the `Engine`, which reads/writes its own local objects in-process.
- **Background task** — registers the store with the Master and heartbeats periodically,
  re-registering (and adopting a new epoch) if a heartbeat fails. This logic is extracted
  from `src/bin/flour-worker.rs` into a reusable `distkv` helper so the standalone worker
  binary and the embedded worker share one implementation.

## 4. Data flow

### Write path (best-effort PUT after generation)

1. `RemoteKvCache.put` calls `put_start`, passing this node's `worker_id` as
   `preferred_worker_id`.
2. Master: if the preferred worker is `Alive` with sufficient remaining capacity, select it;
   otherwise fall back to the existing capacity-based selection (a remote worker). Returns
   `worker_addr` + `object_generation`.
3. Write bytes: if the selected worker is the **local** one, call `store.put_bytes(...)`
   in-process (no HTTP); otherwise HTTP PUT to the remote worker (existing path).
4. `put_commit`.

### Read path (best-effort GET before generation)

1. `get_route(key)` → Master returns `{worker_id, worker_addr, object_generation, lease}` or
   a miss.
2. If `worker_id == local worker_id` → `store.get_bytes(key, generation)` in-process, skipping
   the HTTP round-trip and serialization.
3. Otherwise → HTTP GET to the remote worker (existing path).
4. Miss / generation no longer present → safe miss → fall back to local prefill (unchanged).

## 5. Changes by file

### Protocol (`src/distkv/protocol.rs`)

The only metadata-protocol change is the write-locality hint:

```rust
pub struct PutStartRequest {
    pub key: ObjectKey,
    pub size_bytes: usize,
    pub preferred_worker_id: Option<WorkerId>, // new; None preserves prior behavior
}
```

`#[serde(default)]` on `preferred_worker_id` so existing JSON (without the field)
deserializes to `None` and old/new peers interoperate.

`get_route` and `GetRouteResponse` are **unchanged** — they already carry `worker_id`, which
is all the Engine needs to decide locality. This keeps the Master purely "manages stores, not
engines."

### Master (`src/distkv/master.rs`)

`put_start` worker selection:

- if `preferred_worker_id == Some(w)` and `w` is `Alive` with remaining capacity ≥
  `size_bytes` → choose `w`;
- otherwise fall back to the current capacity-based selection.

No change to `get_route`, lease handling, epoch/generation validation, or any invariant.

### Registration helper (new, e.g. `src/distkv/registration.rs`)

Extract the register + heartbeat + re-register loop currently inlined in
`src/bin/flour-worker.rs` into a reusable async helper. `flour-worker.rs` and the embedded
co-located worker both call it. Behavior (intervals, retry, re-register-on-heartbeat-failure)
is preserved.

### Engine (`src/engine.rs`)

- `RemoteKvCache` gains, in the co-located case: the local `worker_id` and a clone of the
  local `Arc<Mutex<WorkerStore>>`.
- New constructor path, e.g. `Engine::enable_remote_kv_colocated(master_url, worker_id, store)`,
  alongside the existing `enable_remote_kv(master_url)` (which keeps the remote-only behavior).
- `put` passes `preferred_worker_id = Some(local_worker_id)` and, when the chosen worker is
  local, writes to `store` directly.
- `get` compares the route's `worker_id` to the local one and reads `store` directly on a hit.
- The `DistKvClient` is extended so `put_object` accepts a `preferred_worker_id`, and so the
  local-vs-remote branch is expressible. (Exact split between `RemoteKvCache` and
  `DistKvClient` to be settled in the plan; the data-path HTTP code already exists.)

### API wiring (`src/api/mod.rs`, `src/main.rs`)

- Reuse the existing `mount_distkv(router, master, worker)` to merge the `worker_router` onto
  the engine router so both API and worker data path share one server/port.
- `serve` (or `main`) is adjusted to pass the embedded worker through `mount_distkv` when
  co-location is enabled (today `serve` calls `router(state)` directly and ignores
  `mount_distkv`).
- `main.rs` gains the CLI flags in §6 and the startup wiring: build the store, mount routes,
  spawn the registration helper, enable the co-located remote KV on the engine.

## 6. CLI / entry point (`flour` binary)

New, all optional; defaults preserve today's pure-engine behavior:

| flag | meaning |
|------|---------|
| `--colocated-worker` | enable the in-process embedded worker |
| `--worker-id <id>` | this node's store identity (stable across restarts) |
| `--advertise-url <url>` | URL other nodes use to fetch this node's bytes; default `http://<host>:<port>` (same port as the engine) |
| `--distkv-capacity-bytes <n>` | local store capacity, default 1 GiB |

Validation: `--colocated-worker` requires `--remote-kv-master-url` and `--worker-id`.

`flour-master` and `flour-worker` standalone binaries are retained unchanged — the existing
disaggregated deployment still works; co-location is an additive mode.

## 7. Fallback semantics (remote cache strictly optional)

- Master unreachable / registration failing → background task keeps retrying (same as the
  standalone worker); inference is never blocked; Engine get/put failures degrade to a cache
  miss and local prefill.
- Preferred worker lacks capacity → Master falls back to a remote worker; not an error.
- Local-route GET but the generation was already evicted from the local store → safe miss →
  local prefill.
- Local store lock poisoned on a local-route access → treated as a miss, never a panic.

## 8. Testing

- **Protocol:** `PutStartRequest` serde round-trip with and without `preferred_worker_id`;
  backward-compat (legacy JSON → `None`).
- **Master unit:** `put_start_honors_preferred_worker`;
  `put_start_falls_back_when_preferred_full_or_dead`.
- **Engine / integration:**
  - `colocated_put_writes_to_local_store_without_http` — assert the local store holds the
    bytes; prove no network was used (e.g. a remote worker stub that panics if hit).
  - `colocated_get_reads_local_store` — route resolves local → in-process read.
  - `remote_route_still_uses_http` — route to another worker still uses the data path.
  - `generation_succeeds_when_master_down` — preserves the optional-cache guarantee.
- **Single-server:** one axum server answers both `/v1/chat/completions` and
  `/v1/distkv/worker/objects/{key}`.
- **Regression:** existing DistKV integration tests and the deep-bug simulator keep passing
  (the metadata safety state machine is untouched).

## 9. Non-goals

- No multi-replica, no SSD offload, no RDMA, no HA Master (same as the base DistKV plan).
- No Master-side awareness of Engines/Clients; no engine registry or engine liveness.
- No change to the inference API behavior or to generation determinism.
- No change to the DistKV metadata safety invariants or the four deep-bug guards.
