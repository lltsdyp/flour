use std::sync::Arc;

use candle_core::{Device, Tensor};

use crate::kv_cache::key::{PrefixKeyBuilder, BLOCK_SIZE};
use crate::kv_cache::object_store::KvObjectStore;
use crate::kv_cache::policy::PublishPolicy;
use crate::models::common::{Cache, CausalLM};

/// Owns prefix keying, remote hit/miss lookup, publish policy, and the prefill
/// session lifecycle. `Engine` talks only to this type; it never sees the
/// underlying object store or DistKV protocol.
pub struct KvCacheManager {
    key_builder: PrefixKeyBuilder,
    publish_policy: PublishPolicy,
    remote: Option<Arc<dyn KvObjectStore>>,
}

/// Result of a pre-prefill remote lookup. `remote_*` are `None` when the remote
/// cache is disabled, matching the engine's existing metric semantics.
#[derive(Debug, Clone)]
pub struct KvLookup {
    pub remote_key: Option<String>,
    pub remote_cache_hit: Option<bool>,
}

pub struct KvSession<'a> {
    cache: &'a mut Cache,
    lookup: KvLookup,
    prompt_tokens: Vec<u32>,
    reused_prefix_tokens: usize,
}

#[derive(Debug)]
pub struct PrefillOutput {
    pub logits: Tensor,
    pub reused_prefix_tokens: usize,
}

#[derive(Debug)]
pub struct KvPublish {
    pub key: Option<String>,
    pub prompt_tokens: Vec<u32>,
    pub reused_prefix_tokens: usize,
}

impl KvCacheManager {
    pub fn local_only(model_id: impl Into<String>) -> Self {
        Self {
            key_builder: PrefixKeyBuilder::new(model_id, BLOCK_SIZE),
            publish_policy: PublishPolicy::new(2, BLOCK_SIZE),
            remote: None,
        }
    }

    pub fn with_remote(model_id: impl Into<String>, store: Arc<dyn KvObjectStore>) -> Self {
        Self {
            key_builder: PrefixKeyBuilder::new(model_id, BLOCK_SIZE),
            publish_policy: PublishPolicy::new(2, BLOCK_SIZE),
            remote: Some(store),
        }
    }

    pub fn prepare(&self, prompt_tokens: &[u32]) -> KvLookup {
        let Some(remote) = &self.remote else {
            return KvLookup {
                remote_key: None,
                remote_cache_hit: None,
            };
        };
        let key = self.key_builder.key_for_tokens(prompt_tokens);
        let remote_cache_hit = match &key {
            Some(k) => Some(match remote.get_object(k) {
                Ok(Some(_bytes)) => true,
                Ok(None) => false,
                Err(e) => {
                    tracing::warn!("remote kv get failed, falling back to local prefill: {e}");
                    false
                }
            }),
            None => Some(false),
        };
        KvLookup {
            remote_key: key,
            remote_cache_hit,
        }
    }

    pub fn bind_session<'a>(
        &self,
        lookup: KvLookup,
        cache: &'a mut Cache,
        prompt_tokens: Vec<u32>,
    ) -> KvSession<'a> {
        KvSession {
            cache,
            lookup,
            prompt_tokens,
            reused_prefix_tokens: 0,
        }
    }

    pub fn publish_best_effort(&self, publish: KvPublish) {
        let (Some(remote), Some(key)) = (&self.remote, publish.key) else {
            return;
        };
        if !self
            .publish_policy
            .should_store(publish.prompt_tokens.len(), publish.reused_prefix_tokens)
        {
            return;
        }
        let bytes: Vec<u8> = publish
            .prompt_tokens
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .collect();
        if let Err(e) = remote.put_object(&key, bytes) {
            tracing::warn!("remote kv put failed (ignored): {e}");
        }
    }
}

impl<'a> KvSession<'a> {
    pub fn prefill(&mut self, model: &CausalLM, device: &Device) -> anyhow::Result<PrefillOutput> {
        let input = Tensor::new(self.prompt_tokens.as_slice(), device)?.unsqueeze(0)?;
        let (logits, reused_prefix_tokens) = model.prefill_cached(&input, self.cache)?;
        self.reused_prefix_tokens = reused_prefix_tokens;
        Ok(PrefillOutput {
            logits,
            reused_prefix_tokens,
        })
    }

    pub fn cache_mut(&mut self) -> &mut Cache {
        self.cache
    }

    pub fn remote_cache_hit(&self) -> Option<bool> {
        self.lookup.remote_cache_hit
    }

    pub fn remote_key(&self) -> Option<String> {
        self.lookup.remote_key.clone()
    }

    pub fn finish(self) -> KvPublish {
        KvPublish {
            key: self.lookup.remote_key,
            prompt_tokens: self.prompt_tokens,
            reused_prefix_tokens: self.reused_prefix_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::object_store::KvObjectStore;
    use std::sync::{Arc, Mutex};

    struct FailingStore;

    impl KvObjectStore for FailingStore {
        fn get_object(&self, _key: &str) -> anyhow::Result<Option<Vec<u8>>> {
            anyhow::bail!("boom")
        }

        fn put_object(&self, _key: &str, _bytes: Vec<u8>) -> anyhow::Result<()> {
            anyhow::bail!("boom")
        }
    }

    #[test]
    fn prepare_treats_remote_error_as_miss() {
        let manager = KvCacheManager::with_remote("m", Arc::new(FailingStore));
        let tokens: Vec<u32> = (0..40).collect();
        let lookup = manager.prepare(&tokens);
        assert_eq!(lookup.remote_cache_hit, Some(false));
        assert!(lookup.remote_key.is_some());
    }

    #[test]
    fn local_only_prepare_has_no_remote_metrics() {
        let manager = KvCacheManager::local_only("m");
        let tokens: Vec<u32> = (0..40).collect();
        let lookup = manager.prepare(&tokens);
        assert_eq!(lookup.remote_cache_hit, None);
        assert_eq!(lookup.remote_key, None);
    }

    #[derive(Default)]
    struct RecordingStore {
        puts: Mutex<Vec<(String, Vec<u8>)>>,
    }

    impl KvObjectStore for RecordingStore {
        fn get_object(&self, _key: &str) -> anyhow::Result<Option<Vec<u8>>> {
            Ok(None)
        }

        fn put_object(&self, key: &str, bytes: Vec<u8>) -> anyhow::Result<()> {
            self.puts.lock().unwrap().push((key.to_string(), bytes));
            Ok(())
        }
    }

    #[test]
    fn publish_skips_short_prompts() {
        let store = Arc::new(RecordingStore::default());
        let manager = KvCacheManager::with_remote("m", store.clone());
        manager.publish_best_effort(KvPublish {
            key: Some("k".into()),
            prompt_tokens: vec![1, 2, 3],
            reused_prefix_tokens: 0,
        });
        assert!(store.puts.lock().unwrap().is_empty());
    }
}
