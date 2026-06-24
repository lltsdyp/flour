use std::sync::Arc;

use candle_core::{Device, Tensor};

use crate::kv_cache::bundle::KvBundleCodec;
use crate::kv_cache::key::{PrefixKeyBuilder, BLOCK_SIZE};
use crate::kv_cache::object_store::KvObjectStore;
use crate::kv_cache::policy::PublishPolicy;
use crate::models::common::{Cache, CausalLM};

/// Owns prefix keying, remote hit/miss lookup, publish policy, and the prefill
/// session lifecycle. `Engine` talks only to this type; it never sees the
/// underlying object store or DistKV protocol.
pub struct KvCacheManager {
    model_id: String,
    key_builder: PrefixKeyBuilder,
    publish_policy: PublishPolicy,
    remote: Option<Arc<dyn KvObjectStore>>,
}

/// Pre-prefill plan computed from the prompt alone (no network I/O). `remote_key` and
/// `reusable_tokens` are derived from the prompt; `remote_cache_hit` is filled in later, during
/// `KvSession::prefill`, where the actual remote fetch happens. `remote_*` stay `None` when the
/// remote cache is disabled.
#[derive(Debug, Clone)]
pub struct KvLookup {
    pub remote_key: Option<String>,
    pub reusable_tokens: usize,
    pub remote_cache_hit: Option<bool>,
}

pub struct KvSession<'a> {
    cache: &'a mut Cache,
    model_id: String,
    prompt_tokens: Vec<u32>,
    reusable_tokens: usize,
    remote_key: Option<String>,
    remote_store: Option<Arc<dyn KvObjectStore>>,
    reused_prefix_tokens: usize,
    remote_cache_hit: Option<bool>,
    remote_imported_tokens: usize,
    remote_error: Option<String>,
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
        let model_id = model_id.into();
        Self {
            key_builder: PrefixKeyBuilder::new(model_id.clone(), BLOCK_SIZE),
            publish_policy: PublishPolicy::new(2, BLOCK_SIZE),
            remote: None,
            model_id,
        }
    }

    pub fn with_remote(model_id: impl Into<String>, store: Arc<dyn KvObjectStore>) -> Self {
        let model_id = model_id.into();
        Self {
            key_builder: PrefixKeyBuilder::new(model_id.clone(), BLOCK_SIZE),
            publish_policy: PublishPolicy::new(2, BLOCK_SIZE),
            remote: Some(store),
            model_id,
        }
    }

    /// Compute the prefill plan from the prompt alone: the reusable prefix key and token count.
    /// No network I/O happens here — the remote fetch is deferred to `KvSession::prefill`, which
    /// has mutable `Cache` access and so can actually import. `remote_cache_hit` is therefore
    /// `None` until prefill runs (or always `None` when the remote cache is disabled).
    pub fn prepare(&self, prompt_tokens: &[u32]) -> KvLookup {
        let reusable_tokens = self.key_builder.reusable_token_count(prompt_tokens.len());
        if self.remote.is_none() {
            return KvLookup {
                remote_key: None,
                reusable_tokens,
                remote_cache_hit: None,
            };
        }
        let remote_key = self
            .key_builder
            .key_for_reusable_prefix(prompt_tokens)
            .map(|(k, _)| k);
        KvLookup {
            remote_key,
            reusable_tokens,
            remote_cache_hit: None,
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
            model_id: self.model_id.clone(),
            prompt_tokens,
            reusable_tokens: lookup.reusable_tokens,
            remote_key: lookup.remote_key,
            remote_store: self.remote.clone(),
            reused_prefix_tokens: 0,
            remote_cache_hit: None,
            remote_imported_tokens: 0,
            remote_error: None,
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
    /// Reset the sequence, take the best available prefix, and run suffix-only prefill.
    ///
    /// Prefix precedence: a remote bundle is fetched whenever the remote cache is enabled and a
    /// key exists (the fetch determines `remote_cache_hit`). It is only *imported* when the local
    /// match is shorter than the reusable prefix — if the local paged cache already holds the
    /// whole reusable prefix, the remote object still counts as a hit but importing it would add
    /// nothing. Any remote decode/import failure is logged, counted as a miss, and falls back to
    /// the local/cold path: correctness never depends on the remote cache.
    pub fn prefill(&mut self, model: &CausalLM, device: &Device) -> anyhow::Result<PrefillOutput> {
        let prompt = self.prompt_tokens.clone();
        let input = Tensor::new(prompt.as_slice(), device)?.unsqueeze(0)?;

        self.cache.reset_sequence();
        let mut reused = self.cache.match_prefix(&prompt);

        if let (Some(store), Some(key)) = (self.remote_store.clone(), self.remote_key.clone()) {
            match store.get_object(&key) {
                Ok(Some(bytes)) => {
                    if reused >= self.reusable_tokens {
                        // Local cache already covers the full reusable prefix; the object exists
                        // (hit) but importing it would not extend the reuse.
                        self.remote_cache_hit = Some(true);
                    } else {
                        match self.import_remote_bundle(&bytes, &prompt) {
                            Ok(imported) => {
                                reused = imported;
                                self.remote_cache_hit = Some(true);
                                self.remote_imported_tokens = imported;
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "remote kv import failed, falling back to local prefill: {e}"
                                );
                                self.remote_error = Some(e.to_string());
                                // import rolls back, but re-establish a clean local match anyway.
                                self.cache.reset_sequence();
                                reused = self.cache.match_prefix(&prompt);
                                self.remote_cache_hit = Some(false);
                            }
                        }
                    }
                }
                Ok(None) => self.remote_cache_hit = Some(false),
                Err(e) => {
                    tracing::warn!("remote kv get failed, falling back to local prefill: {e}");
                    self.remote_error = Some(e.to_string());
                    self.remote_cache_hit = Some(false);
                }
            }
        }

        let logits = model.prefill_suffix(&input, reused, self.cache)?;
        self.cache.register_prefix(&prompt);
        self.reused_prefix_tokens = reused;

        Ok(PrefillOutput {
            logits,
            reused_prefix_tokens: reused,
        })
    }

    /// Decode `bytes`, reject a wrong-model bundle, and import it into the cache. Returns the
    /// imported token count. Any failure leaves the cache safe to fall back from (import rolls
    /// back its own partial state).
    fn import_remote_bundle(&mut self, bytes: &[u8], prompt: &[u32]) -> anyhow::Result<usize> {
        let bundle = KvBundleCodec::decode(bytes)?;
        if bundle.meta.model_id != self.model_id {
            anyhow::bail!(
                "bundle model_id {:?} != engine model_id {:?}",
                bundle.meta.model_id,
                self.model_id
            );
        }
        Ok(self.cache.import_prefix_bundle(&bundle, prompt)?)
    }

    pub fn cache_mut(&mut self) -> &mut Cache {
        self.cache
    }

    pub fn remote_cache_hit(&self) -> Option<bool> {
        self.remote_cache_hit
    }

    pub fn remote_key(&self) -> Option<String> {
        self.remote_key.clone()
    }

    /// Tokens imported from a remote bundle for this request (0 when none were imported).
    pub fn remote_imported_tokens(&self) -> usize {
        self.remote_imported_tokens
    }

    /// The remote decode/import error, if one occurred and the request fell back.
    pub fn remote_error(&self) -> Option<String> {
        self.remote_error.clone()
    }

    pub fn finish(self) -> KvPublish {
        KvPublish {
            key: self.remote_key,
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

    use crate::kv_cache::bundle::KvBundleCodec;
    use crate::kv_cache::object_store::MemoryKvObjectStore;
    use crate::models::common::model::test_support::{make_vb, prefix_test_config};
    use crate::models::common::{Cache, CausalLM};
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn prepare_computes_key_and_reusable_tokens_without_io() {
        // `prepare` no longer probes the store; remote_cache_hit is decided during prefill.
        let manager = KvCacheManager::with_remote("m", Arc::new(FailingStore));
        let tokens: Vec<u32> = (0..40).collect();
        let lookup = manager.prepare(&tokens);
        assert_eq!(lookup.remote_cache_hit, None);
        assert_eq!(lookup.reusable_tokens, 32);
        assert!(lookup.remote_key.is_some());
    }

    #[test]
    fn local_only_prepare_has_no_remote_metrics() {
        let manager = KvCacheManager::local_only("m");
        let tokens: Vec<u32> = (0..40).collect();
        let lookup = manager.prepare(&tokens);
        assert_eq!(lookup.remote_cache_hit, None);
        assert_eq!(lookup.remote_key, None);
        // The reusable count is still computed locally for local-only publish decisions.
        assert_eq!(lookup.reusable_tokens, 32);
    }

    /// Build a tiny model, a 40-token prompt, and a valid bundle (encoded bytes) for its first
    /// 32 tokens, produced from a donor cache. Returns (model, ids, bundle_bytes).
    fn model_prompt_and_bundle_bytes() -> (CausalLM, Vec<u32>, Vec<u8>) {
        let cfg = prefix_test_config();
        let model = CausalLM::load(make_vb(&cfg), cfg.clone()).unwrap();
        let ids: Vec<u32> = (0..40u32).map(|i| i % cfg.vocab_size as u32).collect();
        let ids_t = Tensor::from_vec(ids.clone(), (1, 40), &Device::Cpu).unwrap();
        let mut donor = Cache::new(&cfg, DType::F32, &Device::Cpu).unwrap();
        model.prefill_cached(&ids_t, &mut donor).unwrap();
        let bundle = donor.export_prefix_bundle("m", &ids, 32).unwrap();
        let bytes = KvBundleCodec::encode(&bundle).unwrap();
        (model, ids, bytes)
    }

    #[test]
    fn session_imports_remote_bundle_before_suffix_prefill() {
        let cfg = prefix_test_config();
        let (model, ids, bytes) = model_prompt_and_bundle_bytes();

        let store = Arc::new(MemoryKvObjectStore::default());
        let manager = KvCacheManager::with_remote("m", store.clone());
        let lookup = manager.prepare(&ids);
        let key = lookup.remote_key.clone().unwrap();
        store.put_object(&key, bytes).unwrap();

        // Fresh cache: no local prefix exists, so reuse can only come from the remote import.
        let mut cache = Cache::new(&cfg, DType::F32, &Device::Cpu).unwrap();
        let mut session = manager.bind_session(lookup, &mut cache, ids.clone());
        let out = session.prefill(&model, &Device::Cpu).unwrap();

        assert_eq!(out.reused_prefix_tokens, 32);
        assert_eq!(session.remote_cache_hit(), Some(true));
        assert_eq!(session.remote_imported_tokens(), 32);
        assert_eq!(session.remote_error(), None);
    }

    #[test]
    fn session_falls_back_to_cold_prefill_on_store_error() {
        let cfg = prefix_test_config();
        let (model, ids, _bytes) = model_prompt_and_bundle_bytes();

        let manager = KvCacheManager::with_remote("m", Arc::new(FailingStore));
        let lookup = manager.prepare(&ids);

        let mut cache = Cache::new(&cfg, DType::F32, &Device::Cpu).unwrap();
        let mut session = manager.bind_session(lookup, &mut cache, ids.clone());
        let out = session.prefill(&model, &Device::Cpu).unwrap();

        // No local prefix, store errors -> cold prefill, nothing reused, counted as a miss.
        assert_eq!(out.reused_prefix_tokens, 0);
        assert_eq!(session.remote_cache_hit(), Some(false));
        assert!(session.remote_error().is_some());
    }

    #[test]
    fn session_rejects_wrong_model_bundle_and_falls_back() {
        let cfg = prefix_test_config();
        let (model, ids, bytes) = model_prompt_and_bundle_bytes();

        // Manager is for a *different* model id; the bundle's model_id is "m".
        let store = Arc::new(MemoryKvObjectStore::default());
        let manager = KvCacheManager::with_remote("other-model", store.clone());
        let lookup = manager.prepare(&ids);
        let key = lookup.remote_key.clone().unwrap();
        store.put_object(&key, bytes).unwrap();

        let mut cache = Cache::new(&cfg, DType::F32, &Device::Cpu).unwrap();
        let mut session = manager.bind_session(lookup, &mut cache, ids.clone());
        let out = session.prefill(&model, &Device::Cpu).unwrap();

        assert_eq!(out.reused_prefix_tokens, 0);
        assert_eq!(session.remote_cache_hit(), Some(false));
        assert!(session.remote_error().is_some());
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
