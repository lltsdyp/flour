use std::collections::HashMap;
use std::sync::Mutex;

pub trait KvObjectStore: Send + Sync {
    fn get_object(&self, key: &str) -> anyhow::Result<Option<Vec<u8>>>;
    fn put_object(&self, key: &str, bytes: Vec<u8>) -> anyhow::Result<()>;
}

#[derive(Debug, Default)]
pub struct MemoryKvObjectStore {
    objects: Mutex<HashMap<String, Vec<u8>>>,
}

impl KvObjectStore for MemoryKvObjectStore {
    fn get_object(&self, key: &str) -> anyhow::Result<Option<Vec<u8>>> {
        Ok(self
            .objects
            .lock()
            .map_err(|_| anyhow::anyhow!("memory kv store poisoned"))?
            .get(key)
            .cloned())
    }

    fn put_object(&self, key: &str, bytes: Vec<u8>) -> anyhow::Result<()> {
        self.objects
            .lock()
            .map_err(|_| anyhow::anyhow!("memory kv store poisoned"))?
            .insert(key.to_string(), bytes);
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct NoopKvObjectStore;

impl KvObjectStore for NoopKvObjectStore {
    fn get_object(&self, _key: &str) -> anyhow::Result<Option<Vec<u8>>> {
        Ok(None)
    }

    fn put_object(&self, _key: &str, _bytes: Vec<u8>) -> anyhow::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_store_round_trips_bytes() {
        let store = MemoryKvObjectStore::default();
        store.put_object("k", vec![1, 2, 3]).unwrap();
        assert_eq!(store.get_object("k").unwrap(), Some(vec![1, 2, 3]));
        assert_eq!(store.get_object("missing").unwrap(), None);
    }

    #[test]
    fn noop_store_always_misses_and_ignores_puts() {
        let store = NoopKvObjectStore;
        assert_eq!(store.get_object("k").unwrap(), None);
        store.put_object("k", vec![1]).unwrap();
        assert_eq!(store.get_object("k").unwrap(), None);
    }
}
