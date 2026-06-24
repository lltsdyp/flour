use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::distkv::worker::WorkerStore;

/// Co-located worker handle with a committed index in front of the raw store.
///
/// `WorkerStore` holds bytes for any `(key, generation)` that was written,
/// including staged writes that were never committed by the Master. To avoid
/// dirty reads, this handle only serves a key once it has been explicitly
/// marked committed at a specific generation.
#[derive(Clone)]
pub struct LocalKvHandle {
    worker_id: String,
    store: Arc<Mutex<WorkerStore>>,
    committed: Arc<Mutex<HashMap<String, u64>>>,
}

impl LocalKvHandle {
    pub fn new(worker_id: String, store: Arc<Mutex<WorkerStore>>) -> Self {
        Self {
            worker_id,
            store,
            committed: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn worker_id(&self) -> &str {
        &self.worker_id
    }

    pub fn try_get_committed(&self, key: &str) -> anyhow::Result<Option<Vec<u8>>> {
        let generation = match self
            .committed
            .lock()
            .map_err(|_| anyhow::anyhow!("local committed index poisoned"))?
            .get(key)
            .copied()
        {
            Some(g) => g,
            None => return Ok(None),
        };
        self.store
            .lock()
            .map_err(|_| anyhow::anyhow!("local worker store poisoned"))?
            .get_bytes(key, generation)
    }

    pub fn mark_committed(&self, key: String, generation: u64) -> anyhow::Result<()> {
        self.committed
            .lock()
            .map_err(|_| anyhow::anyhow!("local committed index poisoned"))?
            .insert(key, generation);
        Ok(())
    }

    pub(crate) fn store(&self) -> &Arc<Mutex<WorkerStore>> {
        &self.store
    }

    #[cfg(test)]
    pub(crate) fn store_for_tests(&self) -> Arc<Mutex<WorkerStore>> {
        self.store.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distkv::worker::WorkerStore;
    use std::sync::{Arc, Mutex};

    fn local() -> LocalKvHandle {
        let store = Arc::new(Mutex::new(WorkerStore::new("local".into(), 1, 1 << 20)));
        LocalKvHandle::new("local".into(), store)
    }

    #[test]
    fn staged_bytes_are_not_returned_until_marked_committed() {
        let local = local();
        local
            .store_for_tests()
            .lock()
            .unwrap()
            .put_bytes("k".into(), 1, vec![1, 2, 3])
            .unwrap();

        assert_eq!(local.try_get_committed("k").unwrap(), None);

        local.mark_committed("k".into(), 1).unwrap();
        assert_eq!(local.try_get_committed("k").unwrap(), Some(vec![1, 2, 3]));
    }

    #[test]
    fn committed_index_points_to_exact_generation() {
        let local = local();
        {
            let store = local.store_for_tests();
            let mut store = store.lock().unwrap();
            store.put_bytes("k".into(), 1, vec![1]).unwrap();
            store.put_bytes("k".into(), 2, vec![2]).unwrap();
        }

        local.mark_committed("k".into(), 1).unwrap();
        assert_eq!(local.try_get_committed("k").unwrap(), Some(vec![1]));

        local.mark_committed("k".into(), 2).unwrap();
        assert_eq!(local.try_get_committed("k").unwrap(), Some(vec![2]));
    }
}
