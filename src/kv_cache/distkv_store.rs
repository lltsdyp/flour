use crate::distkv::client::DistKvClient;
use crate::kv_cache::local::LocalKvHandle;
use crate::kv_cache::object_store::KvObjectStore;

/// `KvObjectStore` backed by the DistKV Master/Worker protocol.
///
/// Hides all Master interaction (route lookup, put start/commit) and worker
/// data-path I/O behind the object-store interface. In co-located mode it
/// owns a `LocalKvHandle` and serves committed local objects in-process
/// before asking the Master, and reads/writes routes that resolve to itself
/// without HTTP.
pub struct DistKvObjectStore {
    client: DistKvClient,
    runtime: tokio::runtime::Runtime,
    local: Option<LocalKvHandle>,
}

impl DistKvObjectStore {
    pub fn connect(master_url: &str) -> anyhow::Result<Self> {
        Self::build(master_url, None)
    }

    pub fn connect_colocated(master_url: &str, local: LocalKvHandle) -> anyhow::Result<Self> {
        Self::build(master_url, Some(local))
    }

    fn build(master_url: &str, local: Option<LocalKvHandle>) -> anyhow::Result<Self> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        Ok(Self {
            client: DistKvClient::new(master_url),
            runtime,
            local,
        })
    }
}

impl KvObjectStore for DistKvObjectStore {
    fn get_object(&self, key: &str) -> anyhow::Result<Option<Vec<u8>>> {
        // Local-first: a committed local object is served without touching the
        // Master. Only committed-index entries qualify, so staged writes never
        // leak as a hit.
        if let Some(local) = &self.local {
            if let Some(bytes) = local.try_get_committed(key)? {
                return Ok(Some(bytes));
            }
        }

        self.runtime.block_on(async {
            let route = match self.client.get_route(key).await? {
                Some(r) => r,
                None => return Ok(None),
            };
            match &self.local {
                // Read locality: the route points at our own worker -> in-process
                // read, and we record the committed generation for future fast paths.
                Some(local) if local.worker_id() == route.worker_id => {
                    let bytes = local
                        .store()
                        .lock()
                        .map_err(|_| anyhow::anyhow!("local worker store poisoned"))?
                        .get_bytes(key, route.object_generation)?;
                    if bytes.is_some() {
                        local.mark_committed(key.to_string(), route.object_generation)?;
                    }
                    Ok(bytes)
                }
                _ => {
                    self.client
                        .fetch_worker(&route.worker_addr, key, route.object_generation)
                        .await
                }
            }
        })
    }

    fn put_object(&self, key: &str, bytes: Vec<u8>) -> anyhow::Result<()> {
        self.runtime.block_on(async {
            let preferred = self.local.as_ref().map(|l| l.worker_id());
            let start = self.client.put_start(key, bytes.len(), preferred).await?;
            match &self.local {
                // Write locality: the Master pinned this object to us -> in-process write.
                Some(local) if local.worker_id() == start.worker_id => {
                    local
                        .store()
                        .lock()
                        .map_err(|_| anyhow::anyhow!("local worker store poisoned"))?
                        .put_bytes(key.to_string(), start.object_generation, bytes)?;
                }
                _ => {
                    self.client
                        .write_worker(&start.worker_addr, key, start.object_generation, bytes)
                        .await?;
                }
            }
            self.client.put_commit(key, start.put_id).await?;
            // Only after the Master commits is it safe to expose the object via
            // the local committed fast path.
            if let Some(local) = &self.local {
                if local.worker_id() == start.worker_id {
                    local.mark_committed(key.to_string(), start.object_generation)?;
                }
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distkv::worker::WorkerStore;
    use crate::kv_cache::local::LocalKvHandle;
    use crate::kv_cache::object_store::KvObjectStore;
    use std::sync::{Arc, Mutex};

    #[test]
    fn local_first_returns_committed_object_without_master() {
        let worker = Arc::new(Mutex::new(WorkerStore::new("local".into(), 1, 1 << 20)));
        worker
            .lock()
            .unwrap()
            .put_bytes("k".into(), 1, vec![1, 2, 3])
            .unwrap();
        let local = LocalKvHandle::new("local".into(), worker);
        local.mark_committed("k".into(), 1).unwrap();

        let store = DistKvObjectStore::connect_colocated("http://127.0.0.1:9", local).unwrap();
        assert_eq!(store.get_object("k").unwrap(), Some(vec![1, 2, 3]));
    }

    #[test]
    fn local_first_does_not_return_staged_object() {
        let worker = Arc::new(Mutex::new(WorkerStore::new("local".into(), 1, 1 << 20)));
        worker
            .lock()
            .unwrap()
            .put_bytes("k".into(), 1, vec![1, 2, 3])
            .unwrap();
        let local = LocalKvHandle::new("local".into(), worker);

        let store = DistKvObjectStore::connect_colocated("http://127.0.0.1:9", local).unwrap();
        let err = store.get_object("k").unwrap_err();
        assert!(
            err.to_string().contains("error") || err.to_string().contains("Connection"),
            "staged object must not be returned as a local hit; got {err}"
        );
    }
}
