//! Deterministic fault-injection model for deep-bug search.
//!
//! The simulator drives the *real* `MasterState` and `WorkerStore` (not a
//! reimplementation) through fault traces — crashes, restarts, partial writes,
//! duplicated/stale commits, overwrites, evictions — and asserts the system's
//! safety invariants after every event. Because it composes the production
//! types, a regression that weakens Master routing or Worker generation keying
//! is caught here.
//!
//! Properties checked (see `docs/plan/plan-distkv.md` §6/§7):
//! - I1 NoDirtyRead: a returned route always names a `Complete` generation.
//! - I2 PlacementHealth / Deep Bug 4: a route never names a crashed worker or a
//!   stale epoch.
//! - Deep Bugs 1/2/3: a fetch on a routed generation returns either the exact
//!   committed bytes or a clean miss — never partial, reused, or stale bytes.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::distkv::master::MasterState;
use crate::distkv::protocol::*;
use crate::distkv::worker::WorkerStore;

/// One worker's physical state in the model.
struct WorkerNode {
    store: WorkerStore,
    capacity: usize,
    epoch: u64,
    crashed: bool,
}

/// The latest write opened for a key but not yet committed.
struct InFlight {
    put_id: PutId,
    generation: u64,
    worker: WorkerId,
    bytes: Vec<u8>,
}

/// The last successfully committed generation for a key (shadow of the Master).
struct Committed {
    generation: u64,
    bytes: Vec<u8>,
}

/// Transport/operation events the random driver can emit. Message duplication is
/// modeled by applying an event twice; message drop by skipping it; reorder by
/// replaying a remembered stale `put_id` via `CommitStale`.
#[derive(Debug, Clone)]
pub enum Event {
    Register { worker: WorkerId, capacity: usize },
    PutStart { key: ObjectKey, size: usize },
    WorkerWrite { key: ObjectKey },
    CommitLatest { key: ObjectKey },
    CommitStale { key: ObjectKey },
    GetAndFetch { key: ObjectKey },
    Crash { worker: WorkerId },
    Restart { worker: WorkerId, capacity: usize },
}

pub struct Sim {
    clock: Arc<AtomicU64>,
    master: MasterState,
    workers: HashMap<WorkerId, WorkerNode>,
    inflight: HashMap<ObjectKey, InFlight>,
    committed: HashMap<ObjectKey, Committed>,
    /// Past `put_id`s per key, used to inject stale/reordered commits.
    history: HashMap<ObjectKey, Vec<PutId>>,
    marker: u32,
}

impl Sim {
    pub fn new() -> Self {
        let clock = Arc::new(AtomicU64::new(1_000));
        let c = clock.clone();
        Self {
            clock,
            master: MasterState::new(move || c.load(Ordering::SeqCst)),
            workers: HashMap::new(),
            inflight: HashMap::new(),
            committed: HashMap::new(),
            history: HashMap::new(),
            marker: 0,
        }
    }

    /// Advances the model clock (used to expire leases / time out heartbeats).
    pub fn advance_clock(&mut self, ms: u64) {
        self.clock.fetch_add(ms, Ordering::SeqCst);
    }

    pub fn register_worker(&mut self, worker: &str, capacity: usize) {
        let epoch = self
            .master
            .register_worker(worker.to_string(), worker.to_string(), capacity);
        // Re-registration (restart) gives a fresh, empty store at a new epoch,
        // which is exactly what invalidates ghost placements (Deep Bug 4).
        self.workers.insert(
            worker.to_string(),
            WorkerNode {
                store: WorkerStore::new(worker.to_string(), epoch, capacity),
                capacity,
                epoch,
                crashed: false,
            },
        );
    }

    /// Opens a write. Returns the Master's response on success (errors, e.g. no
    /// capacity, are surfaced as `None` and treated as no-ops by the driver).
    pub fn put_start(&mut self, key: &str, size: usize) -> Option<PutStartResponse> {
        let resp = self
            .master
            .put_start(PutStartRequest {
                key: key.to_string(),
                size_bytes: size,
                preferred_worker_id: None,
            })
            .ok()?;
        self.marker = self.marker.wrapping_add(1);
        let bytes = vec![self.marker as u8; size.max(1)];
        self.history
            .entry(key.to_string())
            .or_default()
            .push(resp.put_id);
        self.inflight.insert(
            key.to_string(),
            InFlight {
                put_id: resp.put_id,
                generation: resp.object_generation,
                worker: resp.worker_id.clone(),
                bytes,
            },
        );
        Some(resp)
    }

    /// Writes the in-flight bytes to the chosen worker (the data path).
    pub fn worker_write(&mut self, key: &str) {
        let Some(inflight) = self.inflight.get(key) else {
            return;
        };
        let Some(node) = self.workers.get_mut(&inflight.worker) else {
            return;
        };
        if node.crashed {
            return;
        }
        // A capacity rejection here simply means the bytes never land, which the
        // data-path miss check then treats as a safe miss.
        let _ = node
            .store
            .put_bytes(key.to_string(), inflight.generation, inflight.bytes.clone());
    }

    /// Commits the in-flight write with its own (current) `put_id`.
    pub fn commit_latest(&mut self, key: &str) -> bool {
        let Some(inflight) = self.inflight.get(key) else {
            return false;
        };
        let put_id = inflight.put_id;
        self.commit(key, put_id)
    }

    /// Commits with an arbitrary (possibly stale) `put_id`. Used to inject
    /// duplicated/reordered commits and the ABA scenario (Deep Bug 3).
    pub fn commit(&mut self, key: &str, put_id: PutId) -> bool {
        let ok = self
            .master
            .put_commit(PutCommitRequest {
                key: key.to_string(),
                put_id,
            })
            .is_ok();
        if ok {
            // The Master accepted: this must be the current in-flight put_id, so
            // its generation becomes the readable one.
            if let Some(inflight) = self.inflight.get(key) {
                if inflight.put_id == put_id {
                    self.committed.insert(
                        key.to_string(),
                        Committed {
                            generation: inflight.generation,
                            bytes: inflight.bytes.clone(),
                        },
                    );
                }
            }
        }
        ok
    }

    /// Commits using the oldest remembered `put_id` for the key (a stale commit).
    pub fn commit_stale(&mut self, key: &str) {
        let stale = self.history.get(key).and_then(|v| v.first().copied());
        if let Some(put_id) = stale {
            self.commit(key, put_id);
        }
    }

    pub fn get_route(&mut self, key: &str) -> Option<GetRouteResponse> {
        self.master.get_route(key).expect("get_route is infallible")
    }

    /// Fetches bytes directly from a worker, keyed by `(key, generation)`.
    pub fn worker_fetch(&self, worker: &str, key: &str, generation: u64) -> Option<Vec<u8>> {
        let node = self.workers.get(worker)?;
        if node.crashed {
            return None;
        }
        node.store.get_bytes(key, generation).ok().flatten()
    }

    /// Crash: the worker loses its memory and is marked dead at the Master.
    pub fn crash(&mut self, worker: &str) {
        if let Some(node) = self.workers.get_mut(worker) {
            node.crashed = true;
            node.store = WorkerStore::new(worker.to_string(), node.epoch, node.capacity);
        }
        self.master.mark_worker_dead(worker);
    }

    /// Restart with the same id: a fresh empty store at a higher epoch.
    pub fn restart(&mut self, worker: &str, capacity: usize) {
        self.register_worker(worker, capacity);
    }

    /// Evict a specific generation from a worker (storage reclamation).
    pub fn evict(&mut self, worker: &str, key: &str, generation: u64) {
        if let Some(node) = self.workers.get_mut(worker) {
            let _ = node.store.delete_generation(key, generation);
        }
    }

    pub fn apply(&mut self, ev: Event) {
        match ev {
            Event::Register { worker, capacity } => self.register_worker(&worker, capacity),
            Event::PutStart { key, size } => {
                self.put_start(&key, size);
            }
            Event::WorkerWrite { key } => self.worker_write(&key),
            Event::CommitLatest { key } => {
                self.commit_latest(&key);
            }
            Event::CommitStale { key } => self.commit_stale(&key),
            Event::GetAndFetch { key } => {
                if let Some(route) = self.get_route(&key) {
                    let _ = self.worker_fetch(&route.worker_id, &key, route.object_generation);
                }
            }
            Event::Crash { worker } => self.crash(&worker),
            Event::Restart { worker, capacity } => self.restart(&worker, capacity),
        }
    }

    /// Asserts every safety invariant against the current state. Returns an
    /// `Err` describing the first violation, so the random driver can report the
    /// offending seed.
    pub fn check_invariants(&mut self) -> Result<(), String> {
        let keys: Vec<ObjectKey> = self
            .committed
            .keys()
            .chain(self.inflight.keys())
            .cloned()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();

        for key in keys {
            let Some(route) = self.get_route(&key) else {
                continue;
            };

            // I2 / Deep Bug 4: never route to a crashed worker.
            let node = self
                .workers
                .get(&route.worker_id)
                .ok_or_else(|| format!("route to unknown worker {} for {key}", route.worker_id))?;
            if node.crashed {
                return Err(format!(
                    "route to crashed worker {} for {key}",
                    route.worker_id
                ));
            }

            // I1 NoDirtyRead: a route's generation must be a committed one.
            let committed = self
                .committed
                .get(&key)
                .ok_or_else(|| format!("route for {key} but no committed generation"))?;
            if committed.generation != route.object_generation {
                return Err(format!(
                    "route for {key} names gen {} but latest committed is {}",
                    route.object_generation, committed.generation
                ));
            }

            // Deep Bugs 1/2/3: a fetch returns the exact committed bytes or a
            // clean miss — never partial/reused/stale bytes.
            if let Some(bytes) = self.worker_fetch(&route.worker_id, &key, route.object_generation)
            {
                if bytes != committed.bytes {
                    return Err(format!(
                        "fetch for {key} gen {} returned corrupt/reused bytes",
                        route.object_generation
                    ));
                }
            }
        }
        Ok(())
    }
}

impl Default for Sim {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Deep-bug regression tests (the four fixed traces) ---

    /// Deep Bug 1: a partial worker write plus a late/duplicate commit must never
    /// surface partial bytes; the data path returns a clean miss instead.
    #[test]
    fn late_or_duplicate_commit_cannot_publish_partial_object() {
        let mut sim = Sim::new();
        sim.register_worker("w1", 1 << 20);

        let start = sim.put_start("k", 64).unwrap();
        // Partial/failed write: bytes never reach the worker.
        // (no worker_write call)

        // Commit succeeds at the metadata level...
        assert!(sim.commit("k", start.put_id));
        // ...but a duplicate commit is rejected (I3 idempotence).
        assert!(
            !sim.commit("k", start.put_id),
            "duplicate commit must be rejected"
        );

        // The Master may expose a route, but fetching returns NO partial bytes.
        let route = sim.get_route("k").expect("committed object is routable");
        assert_eq!(
            sim.worker_fetch(&route.worker_id, "k", route.object_generation),
            None,
            "partial object must read back as a miss, never partial bytes"
        );
        sim.check_invariants().unwrap();
    }

    /// Deep Bug 2: after a lease "expires" and the generation is evicted and the
    /// slot reused by a newer generation, a delayed fetch on the old generation
    /// must miss — never return the new object's bytes.
    #[test]
    fn expired_lease_never_returns_reused_generation() {
        let mut sim = Sim::new();
        sim.register_worker("w1", 1 << 20);

        // Generation 1 stored and committed.
        sim.put_start("k", 64).unwrap();
        sim.worker_write("k");
        assert!(sim.commit_latest("k"));
        let route_g1 = sim.get_route("k").unwrap();
        assert_eq!(route_g1.object_generation, 1);
        let g1_bytes = sim
            .worker_fetch("w1", "k", 1)
            .expect("gen1 readable before eviction");

        // Lease expires; the object is evicted and the slot reused by gen 2.
        sim.evict("w1", "k", 1);
        sim.put_start("k", 64).unwrap(); // generation 2
        sim.worker_write("k");
        assert!(sim.commit_latest("k"));
        let g2_bytes = sim.worker_fetch("w1", "k", 2).unwrap();
        assert_ne!(g1_bytes, g2_bytes, "generations have distinct bytes");

        // The delayed fetch on the OLD generation must miss, not read gen2 bytes.
        assert_eq!(
            sim.worker_fetch("w1", "k", 1),
            None,
            "evicted generation must not return reused bytes"
        );
        sim.check_invariants().unwrap();
    }

    /// Deep Bug 3: a stale `PutCommit` for an overwritten generation must not
    /// publish the old version over the newer one.
    #[test]
    fn stale_put_commit_cannot_overwrite_newer_version() {
        let mut sim = Sim::new();
        sim.register_worker("w1", 1 << 20);

        let a = sim.put_start("k", 64).unwrap(); // generation 1, put_id A
                                                 // Overwrite with a new version before A ever commits.
        let _b = sim.put_start("k", 64).unwrap(); // generation 2, put_id B
        sim.worker_write("k");
        assert!(sim.commit_latest("k")); // commit B -> gen2 Complete

        // The delayed commit for A must be rejected (I4 NoABA).
        assert!(
            !sim.commit("k", a.put_id),
            "stale put_id from an overwritten generation must not commit"
        );

        let route = sim.get_route("k").expect("gen2 is routable");
        assert_eq!(
            route.object_generation, 2,
            "route must name the newer version"
        );
        sim.check_invariants().unwrap();
    }

    /// Deep Bug 4: a worker that restarts with the same id but a higher epoch
    /// invalidates its old placements; the Master must not route to the ghost.
    #[test]
    fn worker_restart_invalidates_old_placements() {
        let mut sim = Sim::new();
        sim.register_worker("w1", 1 << 20);
        sim.put_start("k", 64).unwrap();
        sim.worker_write("k");
        assert!(sim.commit_latest("k"));
        assert!(
            sim.get_route("k").is_some(),
            "object readable before restart"
        );

        // Restart with the same id: empty store, higher epoch.
        sim.restart("w1", 1 << 20);
        assert!(
            sim.get_route("k").is_none(),
            "old placement on a stale epoch must not be routable"
        );
        sim.check_invariants().unwrap();
    }

    // --- Randomized fault-trace search ---

    /// A tiny xorshift RNG so traces are reproducible from a seed without adding
    /// a dependency.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn random_event(rng: &mut Rng) -> Event {
        let worker = if rng.below(2) == 0 { "w1" } else { "w2" }.to_string();
        let key = format!("k{}", rng.below(4));
        match rng.below(10) {
            0 => Event::PutStart {
                key,
                size: 1 + rng.below(64) as usize,
            },
            1 | 2 => Event::WorkerWrite { key },
            3 | 4 => Event::CommitLatest { key },
            5 => Event::CommitStale { key },
            6 | 7 => Event::GetAndFetch { key },
            8 => Event::Crash { worker },
            _ => Event::Restart {
                worker,
                capacity: 1 << 20,
            },
        }
    }

    #[test]
    fn randomized_traces_never_violate_invariants() {
        // >= 1000 distinct fault traces (acceptance criterion §10).
        for seed in 0..1200u64 {
            let mut sim = Sim::new();
            sim.register_worker("w1", 1 << 20);
            sim.register_worker("w2", 1 << 20);
            let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);

            for step in 0..40 {
                let ev = random_event(&mut rng);
                sim.apply(ev.clone());
                if let Err(why) = sim.check_invariants() {
                    panic!("seed {seed} step {step} event {ev:?} violated invariant: {why}");
                }
            }
        }
    }
}
