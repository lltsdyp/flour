//! In-memory Master metadata state machine.
//!
//! The Master owns *only* metadata: worker liveness/capacity, object state,
//! placement, and read leases. It never stores or transfers object bytes.
//!
//! Safety invariants enforced here (see `docs/plan/plan-distkv.md` §4):
//! - I1 NoDirtyRead: `get_route` returns a route only for `Complete` objects.
//! - I2 PlacementHealth: route only to an `Alive` worker whose epoch matches
//!   the placement's `worker_epoch`.
//! - I3 PutCommitIdempotence: commit only when `Writing` with matching `put_id`.
//! - I4 NoABA: a stale `put_id` (from an overwritten generation) cannot commit.
//! - I6 CapacityAccounting: reservations never exceed a worker's capacity.

use std::collections::HashMap;

use crate::distkv::protocol::*;

/// How long a granted read lease stays valid.
pub const LEASE_TTL_MS: u64 = 5_000;
/// A worker with no heartbeat within this window is treated as not alive.
pub const HEARTBEAT_TIMEOUT_MS: u64 = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkerStatus {
    Alive,
    Dead,
}

struct WorkerMeta {
    addr: String,
    capacity_bytes: usize,
    used_bytes: usize,
    epoch: u64,
    last_heartbeat_ms: u64,
    status: WorkerStatus,
}

impl WorkerMeta {
    fn free_bytes(&self) -> usize {
        self.capacity_bytes.saturating_sub(self.used_bytes)
    }
}

struct ObjectMeta {
    /// Monotonic generation, bumped on every `put_start` for the key.
    generation: u64,
    state: ObjectState,
    put_id: Option<PutId>,
    size_bytes: usize,
    worker_id: Option<WorkerId>,
    /// Worker epoch captured at allocation time (for I2 / ghost-placement).
    worker_epoch: u64,
    /// Whether `size_bytes` is currently reserved against the worker.
    reserved: bool,
}

pub struct MasterState {
    now_ms: Box<dyn Fn() -> u64 + Send + Sync>,
    workers: HashMap<WorkerId, WorkerMeta>,
    objects: HashMap<ObjectKey, ObjectMeta>,
}

impl MasterState {
    pub fn new(now_ms: impl Fn() -> u64 + Send + Sync + 'static) -> Self {
        Self {
            now_ms: Box::new(now_ms),
            workers: HashMap::new(),
            objects: HashMap::new(),
        }
    }

    fn now(&self) -> u64 {
        (self.now_ms)()
    }

    /// Registers (or re-registers) a worker. Re-registration increments the
    /// epoch and clears used capacity, invalidating old placements (I4 / Deep
    /// Bug 4). Returns the worker's new epoch.
    pub fn register_worker(
        &mut self,
        worker_id: WorkerId,
        addr: String,
        capacity_bytes: usize,
    ) -> u64 {
        let now = self.now();
        let entry = self
            .workers
            .entry(worker_id)
            .and_modify(|w| {
                w.epoch += 1;
                w.addr = addr.clone();
                w.capacity_bytes = capacity_bytes;
                w.used_bytes = 0;
                w.last_heartbeat_ms = now;
                w.status = WorkerStatus::Alive;
            })
            .or_insert_with(|| WorkerMeta {
                addr,
                capacity_bytes,
                used_bytes: 0,
                epoch: 1,
                last_heartbeat_ms: now,
                status: WorkerStatus::Alive,
            });
        entry.epoch
    }

    pub fn heartbeat(&mut self, worker_id: &str, epoch: u64) -> anyhow::Result<()> {
        let now = self.now();
        let w = self
            .workers
            .get_mut(worker_id)
            .ok_or_else(|| anyhow::anyhow!("unknown worker {worker_id}"))?;
        if w.epoch != epoch {
            anyhow::bail!(
                "stale heartbeat for {worker_id}: epoch {epoch} != current {}",
                w.epoch
            );
        }
        w.last_heartbeat_ms = now;
        w.status = WorkerStatus::Alive;
        Ok(())
    }

    /// Whether a worker is currently usable for routing/placement.
    fn worker_is_alive(&self, w: &WorkerMeta, now: u64) -> bool {
        w.status == WorkerStatus::Alive
            && now.saturating_sub(w.last_heartbeat_ms) <= HEARTBEAT_TIMEOUT_MS
    }

    pub fn put_start(&mut self, req: PutStartRequest) -> anyhow::Result<PutStartResponse> {
        let now = self.now();

        // Release any reservation held by a previous generation of this key so
        // capacity accounting stays correct across overwrites.
        if let Some(obj) = self.objects.get(&req.key) {
            if obj.reserved {
                if let Some(prev_worker) = obj.worker_id.clone() {
                    let size = obj.size_bytes;
                    if let Some(w) = self.workers.get_mut(&prev_worker) {
                        w.used_bytes = w.used_bytes.saturating_sub(size);
                    }
                }
            }
        }

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

        let (worker_addr, worker_epoch) = {
            let w = self
                .workers
                .get_mut(&worker_id)
                .ok_or_else(|| anyhow::anyhow!("internal: chosen worker {worker_id} vanished"))?;
            w.used_bytes += req.size_bytes;
            (w.addr.clone(), w.epoch)
        };

        let put_id = uuid::Uuid::new_v4();
        let generation = match self.objects.get(&req.key) {
            Some(obj) => obj.generation + 1,
            None => 1,
        };

        self.objects.insert(
            req.key.clone(),
            ObjectMeta {
                generation,
                state: ObjectState::Writing,
                put_id: Some(put_id),
                size_bytes: req.size_bytes,
                worker_id: Some(worker_id.clone()),
                worker_epoch,
                reserved: true,
            },
        );

        Ok(PutStartResponse {
            put_id,
            worker_id,
            worker_addr,
            object_generation: generation,
        })
    }

    pub fn put_commit(&mut self, req: PutCommitRequest) -> anyhow::Result<()> {
        let obj = self
            .objects
            .get_mut(&req.key)
            .ok_or_else(|| anyhow::anyhow!("unknown object {}", req.key))?;

        // I3 + I4: only commit a currently-Writing object whose put_id matches.
        if obj.state != ObjectState::Writing {
            anyhow::bail!("object {} is not Writing (state {:?})", req.key, obj.state);
        }
        if obj.put_id != Some(req.put_id) {
            anyhow::bail!("stale or mismatched put_id for {}", req.key);
        }
        obj.state = ObjectState::Complete;
        Ok(())
    }

    pub fn get_route(&mut self, key: &str) -> anyhow::Result<Option<GetRouteResponse>> {
        let now = self.now();
        let obj = match self.objects.get(key) {
            Some(o) => o,
            None => return Ok(None),
        };

        // I1 NoDirtyRead.
        if obj.state != ObjectState::Complete {
            return Ok(None);
        }
        let worker_id = match &obj.worker_id {
            Some(id) => id.clone(),
            None => return Ok(None),
        };
        let object_generation = obj.generation;
        let placement_epoch = obj.worker_epoch;

        // I2 PlacementHealth + Deep Bug 4: worker alive and epoch matches.
        let worker = match self.workers.get(&worker_id) {
            Some(w) => w,
            None => return Ok(None),
        };
        if !self.worker_is_alive(worker, now) || worker.epoch != placement_epoch {
            return Ok(None);
        }

        Ok(Some(GetRouteResponse {
            key: key.to_string(),
            worker_id,
            worker_addr: worker.addr.clone(),
            object_generation,
            lease_id: uuid::Uuid::new_v4(),
            lease_expires_ms: now + LEASE_TTL_MS,
        }))
    }

    pub fn mark_worker_dead(&mut self, worker_id: &str) {
        if let Some(w) = self.workers.get_mut(worker_id) {
            w.status = WorkerStatus::Dead;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn master_at(now_ms: u64) -> MasterState {
        MasterState::new(move || now_ms)
    }

    #[test]
    fn put_start_creates_writing_object_not_readable() {
        let mut m = master_at(1000);
        m.register_worker("w1".into(), "http://w1".into(), 1 << 20);

        let resp = m
            .put_start(PutStartRequest {
                key: "k".into(),
                size_bytes: 128,
                preferred_worker_id: None,
            })
            .unwrap();
        assert_eq!(resp.worker_id, "w1");
        assert_eq!(resp.object_generation, 1);

        // Writing object must not be routable (I1).
        assert!(m.get_route("k").unwrap().is_none());
    }

    #[test]
    fn put_commit_makes_object_readable() {
        let mut m = master_at(1000);
        m.register_worker("w1".into(), "http://w1".into(), 1 << 20);
        let start = m
            .put_start(PutStartRequest {
                key: "k".into(),
                size_bytes: 128,
                preferred_worker_id: None,
            })
            .unwrap();

        m.put_commit(PutCommitRequest {
            key: "k".into(),
            put_id: start.put_id,
        })
        .unwrap();

        let route = m.get_route("k").unwrap().expect("route after commit");
        assert_eq!(route.worker_id, "w1");
        assert_eq!(route.object_generation, 1);
        assert_eq!(route.worker_addr, "http://w1");
    }

    #[test]
    fn late_commit_with_wrong_put_id_is_rejected() {
        let mut m = master_at(1000);
        m.register_worker("w1".into(), "http://w1".into(), 1 << 20);
        m.put_start(PutStartRequest {
            key: "k".into(),
            size_bytes: 128,
            preferred_worker_id: None,
        })
        .unwrap();

        let wrong = uuid::Uuid::new_v4();
        assert!(m
            .put_commit(PutCommitRequest {
                key: "k".into(),
                put_id: wrong,
            })
            .is_err());

        // Object must remain unreadable.
        assert!(m.get_route("k").unwrap().is_none());
    }

    #[test]
    fn get_route_skips_dead_worker() {
        let mut m = master_at(1000);
        m.register_worker("w1".into(), "http://w1".into(), 1 << 20);
        let start = m
            .put_start(PutStartRequest {
                key: "k".into(),
                size_bytes: 128,
                preferred_worker_id: None,
            })
            .unwrap();
        m.put_commit(PutCommitRequest {
            key: "k".into(),
            put_id: start.put_id,
        })
        .unwrap();
        assert!(m.get_route("k").unwrap().is_some());

        m.mark_worker_dead("w1");
        assert!(m.get_route("k").unwrap().is_none());
    }

    #[test]
    fn capacity_accounting_rejects_when_no_worker_has_space() {
        let mut m = master_at(1000);
        m.register_worker("w1".into(), "http://w1".into(), 100);

        let err = m
            .put_start(PutStartRequest {
                key: "k".into(),
                size_bytes: 200,
                preferred_worker_id: None,
            })
            .unwrap_err();
        assert!(err.to_string().contains("free bytes"), "got: {err}");
    }

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
}
