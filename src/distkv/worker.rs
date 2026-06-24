//! In-memory Worker object store.
//!
//! A Worker owns KV bytes keyed by `(object_key, generation)`. Keying by
//! generation (not by a reusable slot) is what makes Deep Bugs 2 and 3 safe:
//! a stale GET for an evicted generation finds nothing instead of reading
//! reused bytes.

use std::collections::HashMap;

use crate::distkv::protocol::{ObjectKey, WorkerId};

pub struct WorkerStore {
    worker_id: WorkerId,
    epoch: u64,
    capacity_bytes: usize,
    used_bytes: usize,
    objects: HashMap<(ObjectKey, u64), Vec<u8>>,
}

impl WorkerStore {
    pub fn new(worker_id: WorkerId, epoch: u64, capacity_bytes: usize) -> Self {
        Self {
            worker_id,
            epoch,
            capacity_bytes,
            used_bytes: 0,
            objects: HashMap::new(),
        }
    }

    pub fn worker_id(&self) -> &str {
        &self.worker_id
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn put_bytes(
        &mut self,
        key: ObjectKey,
        generation: u64,
        bytes: Vec<u8>,
    ) -> anyhow::Result<()> {
        let slot = (key, generation);
        let prev_len = self.objects.get(&slot).map(|b| b.len()).unwrap_or(0);
        let new_used = self.used_bytes - prev_len + bytes.len();
        if new_used > self.capacity_bytes {
            anyhow::bail!(
                "object ({}, gen {}) of {} bytes exceeds capacity (used {}, cap {})",
                slot.0,
                slot.1,
                bytes.len(),
                self.used_bytes - prev_len,
                self.capacity_bytes
            );
        }
        self.objects.insert(slot, bytes);
        self.used_bytes = new_used;
        Ok(())
    }

    pub fn get_bytes(&self, key: &str, generation: u64) -> anyhow::Result<Option<Vec<u8>>> {
        Ok(self.objects.get(&(key.to_string(), generation)).cloned())
    }

    pub fn delete_generation(&mut self, key: &str, generation: u64) -> anyhow::Result<()> {
        if let Some(bytes) = self.objects.remove(&(key.to_string(), generation)) {
            self.used_bytes = self.used_bytes.saturating_sub(bytes.len());
        }
        Ok(())
    }

    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> WorkerStore {
        WorkerStore::new("w1".into(), 1, 1024)
    }

    #[test]
    fn put_then_get_returns_same_bytes() {
        let mut s = store();
        s.put_bytes("k".into(), 1, vec![1, 2, 3, 4]).unwrap();
        assert_eq!(s.get_bytes("k", 1).unwrap(), Some(vec![1, 2, 3, 4]));
        assert_eq!(s.used_bytes(), 4);
    }

    #[test]
    fn get_with_wrong_generation_returns_none() {
        let mut s = store();
        s.put_bytes("k".into(), 1, vec![1, 2, 3, 4]).unwrap();
        assert_eq!(s.get_bytes("k", 2).unwrap(), None);
        assert_eq!(s.get_bytes("missing", 1).unwrap(), None);
    }

    #[test]
    fn capacity_limit_rejects_large_object() {
        let mut s = store();
        let err = s.put_bytes("k".into(), 1, vec![0u8; 2048]).unwrap_err();
        assert!(err.to_string().contains("capacity"), "got: {err}");
        // Failed write must not be observable.
        assert_eq!(s.get_bytes("k", 1).unwrap(), None);
        assert_eq!(s.used_bytes(), 0);
    }

    #[test]
    fn delete_generation_removes_only_that_generation() {
        let mut s = store();
        s.put_bytes("k".into(), 1, vec![0u8; 10]).unwrap();
        s.put_bytes("k".into(), 2, vec![0u8; 20]).unwrap();
        assert_eq!(s.used_bytes(), 30);

        s.delete_generation("k", 1).unwrap();
        assert_eq!(s.get_bytes("k", 1).unwrap(), None);
        assert_eq!(s.get_bytes("k", 2).unwrap(), Some(vec![0u8; 20]));
        assert_eq!(s.used_bytes(), 20);

        // Deleting a missing generation is a no-op.
        s.delete_generation("k", 99).unwrap();
        assert_eq!(s.used_bytes(), 20);
    }
}
